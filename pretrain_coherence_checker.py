import argparse
import time
import torch
import torch.optim as optim

from modules import *
from utils import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_batch_results(indexer, context, candidates, candidate_labels, predictions):
    """Pretty print results from coherence checker."""
    if DEVICE.type == "cuda":
        context = context.cpu()
        candidates = candidates.cpu()
        candidate_labels = candidate_labels.cpu()
        predictions = predictions.cpu()
    context = batch_indices_to_sentences(indexer, context.detach().numpy())
    candidates = batch_indices_to_sentences(indexer, candidates.detach().numpy())
    candidate_labels = candidate_labels.detach().numpy()
    predictions = predictions.detach().numpy()
    print("##### Sample Outputs #####\n")
    print("CONTEXT:\n")
    for sentence in context:
        print(">>", sentence)
    print("\nTRUE | PRED | CANDIDATE:")
    for true, prediction, sentence in zip(candidate_labels, predictions, candidates):
        print(true, "|", prediction, "|", sentence)
    print("\n")


def run_batch(model, optimizer, batch, indexer,
              step, data_group, print_results=False):
    """Run a batch of coherence checker pretraining.
    
    Args:
        model: CoherenceChecker object.
        optimizer: torch.optim.Adam.
        batch: (context, candidates, candidate_labels) triple.
               context: torch LongTensor, shape <batch-size, seq-length>.
               candidates: same type as `context`, with different batch-size & seq-length.
               candidate_labels: <batch-size,> shaped torch FloatTensor with binary cells, 
                                 same batch-size as `candidates`.
        indexer: Indexer object.
        step: global step index.
        data_group: either Train or Valid.
        print_results: print out prediction-true if True.
    """
    context, candidates, candidate_labels = batch
    logits, predictions = model(context, candidates)
    loss = F.binary_cross_entropy(logits, candidate_labels)
    if data_group == "Train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = ((predictions == candidate_labels).sum().float() / \
                candidate_labels.size(0))
    if print_results:
        print_batch_results(indexer, context, candidates, candidate_labels, predictions)
    return loss.item(), accuracy.item()


def train(info_dir, data_path, data_loader,
          embed_size, hidden_size, attention_size, 
          number_highway_layers, number_rnn_layers,
          number_samples, number_batches, learning_rate,
          save_path, load_path, glove_path,
          valid_size=200, valid_every=5000,
          train_print_every=1000, valid_print_every=100):
    """Train a Coherence checker.
    
    Args:
        info_dir: folder to read indexer and save configuration.
        data_path: path to a ".p" pickle data file.
        data_loader: an NYTLoader, or ROCStoriesLoader, etc.
        embed_size: word embedding size.
        hidden_size: RNN hidden size.
        attention_size: size of attention model.
        number_highway_layers: number of layers of highway net.
        number_rnn_layers: number of layers of stacked RNN.
        number_samples: number of positive/negative samples in data generation.
        number_batches: number of batches to train.
        learning_rate: (starting) learning rate.
        save_path: path to save model.
        laod_path: path to load model.
        glove_path: path to GloVe `.txt` file.
        valid_size: validation size.
        valid_every: validate after every `valid_every` global steps.
        train_print_every: print out loss after every `train_print_every` global steps.
        valid_print_every: print out loss after every `valid_print_every` valid steps.
    """
    print("Preparing data loader ...\n")
    indexer = dill.load(open(info_dir + "indexer.p", "rb"))
    data_loader = data_loader(data_path, indexer)
    configs = {"embed_size": embed_size,
               "hidden_size": hidden_size,
               "attention_size": attention_size,
               "number_highway_layers": number_highway_layers,
               "number_rnn_layers": number_rnn_layers}
    dill.dump(configs, open(info_dir + "configs.p", "wb"))
    data_iterator = BatchGenerator(data_loader, number_samples, 
                                   indexer.get_index("PAD", add=False))
    vocab_size = indexer.size
    if glove_path is None:
        glove_init = None
    else:
        glove_init = load_glove(glove_path, indexer, embed_size)
    model = CoherenceChecker(embed_size, vocab_size,
                             number_highway_layers, number_rnn_layers,
                             hidden_size, attention_size,
                             glove_init).to(DEVICE)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    current_epoch = 0
    best_valid_loss = np.inf
    start = time.time()
    train_losses, train_accuracies = [], []
    print("\nBegin training ...\n")
    for global_step in range(number_batches):
        model.train()
        batch = data_iterator.get_random_batch()
        print_results = True if global_step % train_print_every == 0 else False
        train_loss, train_accuracy = run_batch(model, optimizer, batch, indexer,
                                               global_step, "Train", print_results)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        if print_results:
            print("Step %d loss = %.6f | accuracy = %.4f (%.2f elapsed).\n" % (global_step, 
                                                                               np.mean(train_losses),
                                                                               np.mean(train_accuracies),
                                                                               time.time() - start))
            start = time.time()
        if global_step != 0 and global_step % valid_every == 0:
            print("\n##### Running validation #####\n")
            model.eval()
            valid_losses, valid_accuracies = [], []
            for valid_step in range(valid_size):
                batch = data_iterator.get_random_batch()
                print_results = True if valid_step % valid_print_every == 0 else False
                valid_loss, valid_accuracy = run_batch(model, optimizer, batch, indexer,
                                                       valid_step, "Valid", print_results)
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_accuracy)
            average_train_loss = np.mean(train_losses)
            average_train_accuracy = np.mean(train_accuracies)
            average_valid_loss = np.mean(valid_losses)       
            average_valid_accuracy = np.mean(valid_accuracies)
            print("\nStep %d (valid size = %d):" % (global_step, valid_size))
            print("Average TRAIN loss = %.6f | accuracy = %.4f" % (average_train_loss,
                                                                   average_train_accuracy))
            print("Average VALID loss = %.6f | accuracy = %.4f\n" % (average_valid_loss,
                                                                     average_valid_accuracy))            
            train_losses, train_accuracies = [], []
            if average_valid_loss < best_valid_loss:
                print("Saving model weights for best valid loss %.6f" % average_valid_loss)
                best_valid_loss = average_valid_loss
                torch.save(model.state_dict(), save_path)
                print("Saved as %s\n" % save_path)
            model.train()
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--embed_size", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--attention_size", type=int)
    parser.add_argument("--number_highway_layers", type=int)
    parser.add_argument("--number_rnn_layers", type=int)
    parser.add_argument("--number_samples", type=int)
    parser.add_argument("--number_batches", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--glove_path", type=str)
    parser.add_argument("--valid_size", type=int)
    parser.add_argument("--valid_every", type=int)
    parser.add_argument("--train_print_every", type=int)
    parser.add_argument("--valid_print_every", type=int)
    args = parser.parse_args()
    
    print("Device:", DEVICE)
    print("Configs:", args)
    print("\n>>>>>>>>>> START TRAINING <<<<<<<<<<\n")
    train(args.info_dir,
          args.data_path,
          NYTLoader,
          args.embed_size,
          args.hidden_size,
          args.attention_size,
          args.number_highway_layers,
          args.number_rnn_layers,
          args.number_samples,
          args.number_batches,
          args.learning_rate,
          args.save_path,
          args.load_path,
          args.glove_path,
          args.valid_size,
          args.valid_every,
          args.train_print_every,
          args.valid_print_every)  