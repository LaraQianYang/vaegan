from collections import Counter
import dill
import linecache
import numpy as np
import random
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_glove(glove_path, indexer, embed_size=300):
    """Load pretrained GloVe embeddings (Pennington et al. 2014).
    
    Args:
        glove_path: path to the glove .txt file downloaded from
                    `https://nlp.stanford.edu/projects/glove/`.
        indexer: Indexer object. Words are added by reading data by now.
        embed_size: embedding size.
    Returns:
        embeddings: a numpy ndarray of shape <vocab-size, embed-size>.
    """
    embeddings = np.zeros((indexer.size, embed_size))
    number_loaded = 0
    print("Loading glove embeddings (size = %d)\n" % embed_size)
    with open(glove_path, "r") as glove_file:
        for i, line in enumerate(glove_file):
            line = line.split()
            word, embedding = line[0], np.array(line[1:], dtype=float)
            if indexer.contains(word):
                word_index = indexer.get_index(word, add=False)
                embeddings[word_index] = embedding
                number_loaded += 1
            if i != 0 and i % 100000 == 0:
                print("... processed %d lines." % i)
    print("\nDone!\n")
    print("Loaded %d | OOV size = %d\n" % (number_loaded, indexer.size-number_loaded))
    return embeddings


class Indexer(object):
    """Word to index bidirectional mapping."""
    
    def __init__(self, start_symbol="<s>", end_symbol="</s>"):
        """Initializing dictionaries and (hard coded) special symbols."""
        self.word_to_index = {}
        self.index_to_word = {}
        self.size = 0
        # Hard-code special symbols.
        self.get_index("PAD", add=True) 
        self.get_index("UNK", add=True) 
        self.get_index(start_symbol, add=True)
        self.get_index(end_symbol, add=True)
    
    def __repr__(self):
        """Print size info."""
        return "This indexer currently has %d words" % self.size
    
    def get_word(self, index):   
        """Get word by index if its in range. Otherwise return `UNK`."""
        return self.index_to_word[index] if index < self.size and index >= 0 else "UNK"

    def get_index(self, word, add):
        """Get index by word. If `add` is on, also append word to the dictionaries."""
        if self.contains(word):
            return self.word_to_index[word]
        elif add:
            self.word_to_index[word] = self.size
            self.index_to_word[self.size] = word
            self.size += 1
            return self.word_to_index[word]
        return self.word_to_index["UNK"]
        
    def contains(self, word):
        """Return True/False to indicate whether a word is in the dictionaries."""
        return word in self.word_to_index
    
    def add_sentence(self, sentence, add):
        """Add all the words in a sentence (a string) to the dictionary."""
        indices = [self.get_index(word, add) for word in sentence.split()]
        return indices

    def add_document(self, document_path, add):
        """Add all the words in a document (a path to a text file) to the dictionary."""
        indices_list = []
        with open(document_path, "r") as document:
            for line in document:
                indices = self.add_sentence(line, add)
                indices_list.append(indices)
        return indices_list
    
    def to_words(self, indices):
        """Indices (ints) -> words (strings) conversion."""
        return [self.get_word(index) for index in indices]
    
    def to_sent(self, indices):
        """Indices (ints) -> sentence (1 string) conversion."""
        return " ".join(self.to_words(indices))
    
    def to_indices(self, words):
        """Words (strings) -> indices (ints) conversion."""
        return [self.get_index(word, add=False) for word in words]

    
def get_word_counts(word_to_counts, document_path, print_every=1000):
    """Update a word->counts dictionary from a document source.
    
    Args:
        word_to_counts: Counter object.
        document_path: path to a file where each line is a string sentence.
        print_every: report frequency.
    """
    with open(document_path, "r") as document:
        token_count = 0
        for index, line in enumerate(document):
            words = line.strip().split()
            for word in words:
                token_count += 1
                word_to_counts[word] += 1
            if index != 0 and index % print_every == 0:
                print("... processed %d lines." % index)
    print("\nDone! (%d tokens processed)\n" % token_count)
    
    
def get_line_count(document_path):
    """Return number of lines in a document."""
    with open(document_path, "r") as document:
        return len(document.readlines())
    
    
def get_data_size(source_path, target_path):
    """Return #lines in paraphrase document pair (raise exception if not match)."""
    source_size = get_line_count(source_path)
    target_size = get_line_count(target_path)
    if source_size == target_size:
        return source_size
    else:
        raise Exception("Paraphrase documents must match in line counts, but got",
                        "Source #lines = %d | Target #lines = %d" % (source_size,
                                                                     target_size))
        
        
def words_to_clean_sentences(words):
    """Strip start/end/pad tokens and join words."""
    special_tokens = ["<s>", "</s>", "PAD"]
    return " ".join(word for word in words if word not in special_tokens)


def batch_indices_to_sentences(indexer, batch_indices):
    """Convert a batch of word indices into sentences."""
    batch_sentences = [words_to_clean_sentences(indexer.to_words(indices))
                       for indices in batch_indices]
    return batch_sentences


def batch_sentences_to_indices(indexer, batch_sentences):
    """Convert a batch of sentences into index lists."""
    # batch_sentences format: [["..."], ["..."], ...].
    batch_indices = [indexer.to_indices(sentence[0].split()) for sentence in batch_sentences]
    return batch_indices


def to_tensor(inputs, tensor_type):
    """Object -> tensor."""
    return torch.Tensor(inputs).type(tensor_type).to(DEVICE)
        
        
def add_border_token(words, border_token):
    """Adding either a start or end token."""
    if border_token == "<s>":
        return [border_token] + words
    elif border_token == "</s>":
        return words + [border_token]
    else:
        raise Exception("`border_token` must be either <s> or </s>, but got %s." % border_token)


def get_sentences(document_path, indices, border_token, indexer=None):
    """Read specified sentences from a document path."""
    sentences = []
    for index in indices:
        sentence = linecache.getline(document_path, index).strip()
        words = add_border_token(sentence.split(), border_token)
        if indexer is not None:
            words = [indexer.get_index(word, add=False) for word in words]
        sentences.append(words)
    return sentences


def pad_sentences(sentences, pad):
    """Pad a batch of sentences to its longest member."""
    sentences_padded = []
    pad_length = max(len(sentence) for sentence in sentences)
    for sentence in sentences:
        length = len(sentence)
        if length < pad_length:
            sentence = sentence + [pad] * (pad_length - length)
        sentences_padded.append(sentence)
    return sentences_padded


class ParaphraseIterator(object):
    """Data iterator for paraphrase pairs."""
    
    def __init__(self, source_path, target_path, indexer):
        """Initializer.
        
        Args:
            source_path: path to a document where each line is a string sentence.
            target_path: same as `source_path`, but for target.
            indexer: Indexer object.
        """
        self.source_path = source_path
        self.target_path = target_path
        self.indexer = indexer
        self.pad = indexer.get_index("PAD", add=False)
        self.data_size = get_data_size(source_path, target_path)
        self.indices = np.arange(1, self.data_size + 1) # line indices start from 1.
        self.cursor = 0
        self.epoch = 0
        self.shuffle()
    
    def shuffle(self):
        """Randomly shuffle sentence indices."""
        random.shuffle(self.indices)
        
    def get_next_batch(self, batch_size):
        """Return a batch of sentences in ELMO embeddings."""
        if self.cursor >= self.data_size:
            self.cursor = 0
            self.epoch += 1
            self.shuffle()
        batch_indices = self.indices[self.cursor : self.cursor + batch_size]
        batch_source = pad_sentences(get_sentences(self.source_path,
                                                   batch_indices,
                                                   "</s>",
                                                   self.indexer), self.pad)
        batch_target = pad_sentences(get_sentences(self.target_path,
                                                   batch_indices,
                                                   "</s>",
                                                   self.indexer), self.pad)
        self.cursor += batch_size
        return to_tensor(batch_source, torch.LongTensor), \
               to_tensor(batch_target, torch.LongTensor)
    
    
def get_sample_predictions(indexer, logits):
    """Get predictions of a NestedVAE in sentences."""
    predictions = logits.argmax(dim=-1)
    if DEVICE.type == "cuda":
        predictions = predictions.cpu()
    predictions = batch_indices_to_sentences(indexer, predictions.detach().numpy())
    return predictions
    
    
def print_samples(indexer, logits):
    """Print sample predictions."""
    predictions = get_sample_predictions(indexer, logits)
    for prediction in predictions:
        print("PRED >> %s" % prediction)

        
def print_pairs(indexer, logits, targets):
    """Print sample predictions and gold targets."""
    predictions = logits.argmax(dim=-1)
    if DEVICE.type == "cuda":
        predictions, targets = predictions.cpu(), targets.cpu()
    predictions = batch_indices_to_sentences(indexer, predictions.detach().numpy())
    targets = batch_indices_to_sentences(indexer, targets.detach().numpy())
    for prediction, target in zip(predictions, targets):
        print("PRED >>", prediction)
        print("TRUE >>", target, "\n")
        
        
##### NYT Pretraining Utils #####

class NYTLoader(object):
    """NYT narrative loader. Randomly or by index."""
    
    def __init__(self, data_path, indexer):
        """Load NYT salads data (.p) from path."""
        self.nyt_dataframe = dill.load(open(data_path, "rb"))
        self.indexer = indexer
        self.size = len(self.nyt_dataframe)
        
    def get_random_index(self):
        """Return a random index."""
        return random.choice(range(self.size))
    
    def load_document_random(self):
        """Randomly load a narrative (see `load_narrative_by_index`)."""
        index = self.get_random_index()
        return self.load_document_by_index(index) 
    
    def load_document_by_index(self, index):
        """Load a narrative by index: string narrative ID and a list of list of word indices."""
        narrative = self.nyt_dataframe.iloc[index]
        narrative_id = narrative["id"]
        sentences = []
        for sentence in narrative["narrative"]:
            sentence = self.indexer.add_sentence(sentence, add=False)
            sentences.append(sentence)
        return narrative_id, sentences
    
    
def in_document_sample(document, number_samples=3):
    """Partition a document into one set of k sample sentences and another k-1 sentences.
    
    E.g.
    >>> a = [1,2,3,4,5]
    >>> a_sp = in_document_sample(a)
    >>> a, a_sp
    ([3, 5], [4, 1, 2])
    """
    samples = []
    for _ in range(number_samples):
        samples.append(document.pop(random.choice(range(len(document)))))
    return samples


def generate_coherence_data_entry(data_loader, number_samples=3):
    """Generate a data point for the coherence checker pretraining.
    
    Take a document, sample three sentences in it as positive cases,
    then sample `number_samples` sentences from other randomly sampled (different) documents.
    """
    _, document = data_loader.load_document_random()
    positive_sentences = in_document_sample(document, number_samples)
    negative_sentences = []
    for _ in range(number_samples):
        _, other_document = data_loader.load_document_random()
        negative_sentences.append(random.choice(other_document))
    return document, positive_sentences, negative_sentences


def shuffle_coherence_data_entry(document, positive_sentences, negative_sentences):
    """Create a batch of data: context, candidates and binary labels (coherent or not to context)."""
    context = np.array(document)
    candidates = positive_sentences + negative_sentences
    candidate_indices = np.arange(len(candidates), dtype=int)
    candidate_labels = np.concatenate((np.ones(len(positive_sentences), dtype=int),
                                       np.zeros(len(negative_sentences), dtype=int)))
    random.shuffle(candidate_indices)
    candidates = np.array(candidates)[candidate_indices]
    candidate_labels = candidate_labels[candidate_indices]
    return context, candidates, candidate_labels


class BatchGenerator(object):
    """Random data batch generator for coherence checker."""
    
    def __init__(self, data_loader, number_samples, pad):
        """Initializer.
        
        Args:
            data_loader: an NYTLoader object.
            indexer: an Indexer object.
            number_samples: number of positive/negative cases to be sampled.
        """
        self.data_loader = data_loader
        self.number_samples = number_samples
        self.pad = pad
    
    def get_random_batch(self):
        """Randomly create a batch of data and format it as a data entry.
        
        Returns:
            context: torch LongTensor, shape = <batch-size, seq-length>.
            candidates: same type as `context`.
            candidate_labels: torch LongTensor with binary labels, 
                              1 if a candidate is coherent to context, 0 else.
        """
        document, \
        positive_sentences, negative_sentences = generate_coherence_data_entry(self.data_loader,
                                                                               self.number_samples)
        context, candidates, candidate_labels = shuffle_coherence_data_entry(document,
                                                                              positive_sentences,
                                                                              negative_sentences)
        context = torch.LongTensor(pad_sentences(context, self.pad)).to(DEVICE)
        candidates = torch.LongTensor(pad_sentences(candidates, self.pad)).to(DEVICE)
        candidate_labels = torch.FloatTensor(candidate_labels).to(DEVICE)
        return context, candidates, candidate_labels