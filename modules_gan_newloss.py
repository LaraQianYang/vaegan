import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embeddings(nn.Module):
    """Embedding lookup."""
    
    def __init__(self, embed_size, vocab_size, glove_init=None):
        """
        Args:
            embed_size: embedding size.
            vocab_size: vocab_size size.
            glove_init: numpy.ndarray, initializing embeddings.
        """
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        if glove_init is not None:
            assert glove_init.shape == (vocab_size, embed_size)
            self.embeddings.weight.data.copy_(torch.from_numpy(glove_init))
        self.embed_size = embed_size

    def forward(self, inputs):
        """
        Args:
            inputs: <batch-size, seq-length>.
        """
        # Lookup embeddings: <batch-size, seq-length> 
        #                 -> <batch-size, seq-length, embed-size>
        return self.embeddings(inputs) * math.sqrt(self.embed_size)
    
    
class HighwayLayer(nn.Module):
    """Multi-layer highway transformation.
    
    Source: Srivastava/15, Eq. (3). https://arxiv.org/pdf/1505.00387.pdf.
    """
    
    def __init__(self, input_size, number_layers, nonlinearity):
        """Initializer.
        
        Args:
            input_size: feature size of `inputs`. Typically embedding size.
            number_layers: number of highway layers.
            nonlinearity: a function for non-linear transformation (e.g. ReLU).
        """
        super(HighwayLayer, self).__init__()
        self.number_layers = number_layers
        self.nonlinear_layers = nn.ModuleList([nn.Sequential(nn.Linear(input_size, 
                                                                       input_size),
                                                             nonlinearity)
                                               for _ in range(number_layers)])
        self.linear_layers = nn.ModuleList([nn.Linear(input_size, 
                                                      input_size)
                                            for _ in range(number_layers)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(input_size,
                                                            input_size),
                                                  nn.Sigmoid())
                                    for _ in range(number_layers)])
    
    def forward(self, inputs):
        """Forward pass.
        
        Args:
            inputs: torch tensor of any shape.
        Returns:
            outputs: torch tensor of the same shape and type after highway transformation.
        """
        outputs = inputs
        for layer_index in range(self.number_layers):
            nonlinear = self.nonlinear_layers[layer_index](outputs)
            linear = self.linear_layers[layer_index](outputs)
            gate = self.gates[layer_index](outputs)
            outputs = gate * nonlinear + (1 - gate) * linear
        return outputs
    
    
def process_state(state):
    """Peformance last layer slicing and bidirectional stacking on LSTM final state (h or c)."""
    # Input shape = <number-layers*number-directions, batch-size, hidden-size>.
    _, batch_size, hidden_size = state.size()
    # Get last layer (-1: number-layers is inferred).
    #   <2, batch-size, hidden-size>.
    state = state.reshape(-1, 2, batch_size, hidden_size)[-1] 
    # Bidirectional stacking.
    #   <batch-size, 2*hidden-size>.
    state = state.permute(1, 0, 2).reshape(batch_size, -1)
    return state   


# Nested VAE Modules    
    
class NestedEncoder(nn.Module):
    """The `Encoder Side` as per Gupta/18."""
    
    def __init__(self, input_size, hidden_size, latent_size, 
                 number_layers, highway, dropout_rate):
        """Initializer.
        
        Args:
            input_size: typically embedding size.
            hidden_size: hidden size of LSTMs.
            latent_size: reparametric size.
            number_layers: number of layers of stacked LSTM.
            highway: HighwayLayer object.
            dropout_rate: dropout rate.
        """
        super(NestedEncoder, self).__init__()
        self.highway = highway
        self.encoder = nn.LSTM(input_size, hidden_size, number_layers,
                               batch_first=True, bidirectional=True, 
                               dropout=dropout_rate)
        self.decoder = nn.LSTM(input_size, hidden_size, number_layers,
                               batch_first=True, bidirectional=True, 
                               dropout=dropout_rate)
        self.linear_mu = nn.Linear(hidden_size * 4, latent_size)
        self.linear_logvar = nn.Linear(hidden_size * 4, latent_size)
        
    def forward(self, source_inputs, target_inputs):
        """Encoding-Decoding pass on source & target inputs and return reparam-ed latent states.
        
        Args:
            source_inputs: torch FloatTensor, shape = <batch-size, seq-length, input-size>.
                           input-size is typically embedding size.
            target_inputs: same type and shape as `source_inputs` with the exception of
                           a different seq-length.
        Returns:
            mu: torch FloatTensor, shape = <batch-size, latent-size>.
            logvar: same type and shape as `mu`.
        """
        # Encoding-Decoding.
        source_inputs = self.highway(source_inputs)
        _, state = self.encoder(source_inputs)
        target_inputs = self.highway(target_inputs)
        _, state = self.decoder(target_inputs, state)
        # Compute final state.
        [h_state, c_state] = state
        final_state = torch.cat([process_state(h_state),
                                 process_state(c_state)], dim=-1)
        # Compute mu and logvar.
        mu = self.linear_mu(final_state)
        logvar = self.linear_logvar(final_state)
        return mu, logvar
    
# encoder at test time
class NestedEncoder_test(nn.Module):
    
    def __init__(self, input_size, hidden_size, latent_size, 
                 number_layers, highway, dropout_rate):
        """Initializer.
        
        Args:
            input_size: typically embedding size.
            hidden_size: hidden size of LSTMs.
            latent_size: reparametric size.
            number_layers: number of layers of stacked LSTM.
            highway: HighwayLayer object.
            dropout_rate: dropout rate.
        """
        super(NestedEncoder_test, self).__init__()
        self.highway = highway
        self.encoder = nn.LSTM(input_size, hidden_size, number_layers,
                               batch_first=True, bidirectional=True, 
                               dropout=dropout_rate)
        self.linear_mu = nn.Linear(hidden_size * 4, latent_size)
        self.linear_logvar = nn.Linear(hidden_size * 4, latent_size)
        
    def forward(self, source_inputs):
        """Encoding-Decoding pass on source & target inputs and return reparam-ed latent states.
        
        Args:
            source_inputs: torch FloatTensor, shape = <batch-size, seq-length, input-size>.
                           input-size is typically embedding size.
            target_inputs: same type and shape as `source_inputs` with the exception of
                           a different seq-length.
        Returns:
            mu: torch FloatTensor, shape = <batch-size, latent-size>.
            logvar: same type and shape as `mu`.
        """
        # Encoding
        source_inputs = self.highway(source_inputs)
        _, state = self.encoder(source_inputs)
        # Compute final state.
        [h_state, c_state] = state
        final_state = torch.cat([process_state(h_state),
                                 process_state(c_state)], dim=-1)
        # Compute mu and logvar.
        mu = self.linear_mu(final_state)
        logvar = self.linear_logvar(final_state)
        return mu, logvar
    
    
class NestedDecoder(nn.Module):
    """The `Decoder Side` as per Gupta/18."""
    
    def __init__(self, input_size, hidden_size, latent_size, 
                 number_layers, highway, dropout_rate):
        """Initializer.
        
        Args:
            input_size: typically embedding size.
            hidden_size: hidden size of LSTMs.
            latent_size: reparametric size.
            number_layers: number of layers of stacked LSTM.
            highway: HighwayLayer object.
            dropout_rate: dropout rate.
        """
        super(NestedDecoder, self).__init__()
        self.highway = highway
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.encoder = nn.LSTM(input_size, hidden_size, number_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout_rate)
        self.decoder = nn.LSTM(latent_size + input_size, hidden_size, number_layers,
                               batch_first=True, bidirectional=False,
                               dropout=dropout_rate)
        self.h_to_initial_state = nn.Linear(hidden_size * 2,
                                            number_layers * hidden_size)
        self.c_to_initial_state = nn.Linear(hidden_size * 2,
                                            number_layers * hidden_size)
    
    def format_initial_state(self, state, batch_size):
        """Format an h or c state for LSTM initialization."""
        # Input shape = <batch-size, number-decoder-layers*decoder-hidden-size>.
        # Separate number-decoder-layers & decoder-hidden-size as two dimensions.
        state = state.reshape(batch_size, self.number_layers, self.hidden_size)
        # Format to standard initial shape <number-layers, batch-size, hidden-size>.
        state = state.permute(1, 0, 2)
        return state
    
    def get_initial_state(self, source_inputs):
        """Compute initial (h, c) state for decoder."""
        # Compute initial state as the output of the encoder.
        batch_size = source_inputs.size(0)
        source_inputs = self.highway(source_inputs)
        _, state = self.encoder(source_inputs)
        [h_state, c_state] = state
        # Format the h & c states to standard initial shape:
        #   <number-layers, batch-size, hidden-size>.
        #   NB: initial states must be contiguous!
        h_initial = self.format_initial_state(self.h_to_initial_state(process_state(h_state)),
                                              batch_size).contiguous()
        c_initial = self.format_initial_state(self.c_to_initial_state(process_state(c_state)),
                                              batch_size).contiguous()
        return h_initial, c_initial
    
    def forward(self, decode_inputs, latent_inputs, previous_state=None):
        """Encoding-Decoding 1-step pass on source & target inputs on decoder side.
        
        Args:
            decode_inputs: torch FloatTensor, shape = <batch-size, 1, input-size>
            latent_inputs: torch FloatTensor, shape = <batch-size, latent-size>.
            previous_state: previous (h, t) state tuple, None or with the shape
                           <number-(decoder)-layers, batch-size, (decoder)-hidden-size>.
        Returns:
            outputs: torch FloatTensor, shape = <batch-size, 1, vocab-size>.
            final_state: decoder final state (h, t) tuple, torch FloatTensor with the shape
                         <number-(decoder)-layers, batch-size, (decoder)-hidden-size>
        """
        batch_size = decode_inputs.size(0)
        # Concat target and latent.
        #   -> <batch-size, 1, latent-size>
        latent_inputs = latent_inputs.unsqueeze(1)
        #   -> <batch-size, 1, hidden-size+latent-size>
        decode_inputs = torch.cat([decode_inputs, latent_inputs], dim=-1)
        # Decode and format outputs.
        outputs, state = self.decoder(decode_inputs, previous_state)
        return outputs, state
    
    
class NestedVAE(nn.Module):
    """Nested Variational Autoencoder Seq2seq model, as per Gupta/18."""
    
    def __init__(self, input_size, hidden_size, latent_size, vocab_size,
                 number_highway_layers, number_rnn_layers, 
                 dropout_rate, enforce_ratio,
                 start_index, end_index, glove_init, alpha=0.7):
        """Initializer.
        
        Args:
            input_size: typically embedding size.
            hidden_size: hidden size of LSTMs.
            latent_size: reparametric size.
            vocab_size: vocabulary size.
            number_highway_layers: number of layers of highway net.
            number_rnn_layers: number of layers of stacked LSTM.
            dropout_rate: dropout rate.        
            enforce_ratio: ratio of teacher enforce (during training).
            start_index: index of start symbol <s>.
            end_index: index of end symbol </s>.
            glove_init: initializing GloVe embedding matrix.
            alpha:   temperature for softmax
        """
        super(NestedVAE, self).__init__()
        self.latent_size = latent_size
        self.enforce_ratio = enforce_ratio
        self.start_index = start_index
        self.end_index = end_index
        self.alpha = alpha
        self.embedder = Embeddings(input_size, vocab_size, glove_init)
        self.highway = HighwayLayer(input_size, number_highway_layers, nn.ReLU())
        self.nested_encoder = NestedEncoder(input_size, hidden_size, latent_size,
                                            number_rnn_layers, self.highway, dropout_rate)
        self.nested_encoder_test = NestedEncoder_test(input_size, hidden_size, latent_size,
                                            number_rnn_layers, self.highway, dropout_rate)
        self.nested_decoder = NestedDecoder(input_size, hidden_size, latent_size,
                                            number_rnn_layers, self.highway, dropout_rate)
        self.linear_final = nn.Linear(hidden_size, vocab_size)
        
    def get_latent_inputs(self, source_inputs, target_inputs, training, mode=1):
        """Get reparametric/latent states.
        
        Args:
            source_inputs: torch FloatTensor, shape = <batch-size, seq-length, input-size>.
                           input-size is typically embedding size.
            target_inputs: same type and shape as `source_inputs` with the exception of
                           a different seq-length.
            training: boolean. Returns latent states from the encoder side if True (training),
                      returns randomly sampled latent states from unit Gaussian otherwise.
        Returns:
            latent_inputs: torch FloatTensor, shape = <batch-size, latent-size>.
            kl_loss: tensor (backwardable) float loss if `training` is True, 0 else.
        """
        batch_size = source_inputs.size(0)
        if training:
            if mode == 1:
                mu, logvar = self.nested_encoder(source_inputs, target_inputs)
                kl_loss = (-0.5 * torch.sum(logvar - \
                                            torch.pow(mu, 2) - \
                                            torch.exp(logvar) + 1, 
                                            dim=1)).mean().squeeze()
            else:
                mu, logvar = self.nested_encoder_test(source_inputs)
                kl_loss = 0

            std = torch.exp(0.5 * logvar)
            latent_inputs = torch.randn([batch_size, self.latent_size]).to(DEVICE) * \
                            std + mu
        else:
            kl_loss = 0
            mu, logvar = self.nested_encoder_test(source_inputs)
            std = torch.exp(0.5 * logvar)
            latent_inputs = torch.randn([batch_size, self.latent_size]).to(DEVICE) * std + mu
        return latent_inputs, kl_loss
    
    def get_initial_inputs(self, batch_size):
        """Get initial inputs.
        
        Args:
            batch_size: batch size.
        Returns:
            Embedded batch of start symbols, shape = <batch-size, 1>.
        """
        return self.embedder(to_tensor([[self.start_index] 
                                        for _ in range(batch_size)],
                                       torch.LongTensor)).to(DEVICE)
    
    def decode_train(self, source_inputs, target_inputs, latent_inputs):
        """Train-time decoding: feed true next-token at a teacher enforce ratio."""
        batch_size = source_inputs.size(0)
        decode_inputs = self.get_initial_inputs(batch_size)
        state = self.nested_decoder.get_initial_state(source_inputs)
        decode_length = target_inputs.size(1)
        logits = []
        for i in range(decode_length):
            enforce = random.random() < self.enforce_ratio
            outputs, state = self.nested_decoder(decode_inputs, latent_inputs, state)
            logit = self.linear_final(outputs)
            logit = logit * self.alpha
            logits.append(logit)
            if enforce:
                decode_inputs = target_inputs[:, i, :].unsqueeze(1)
            else:
                decode_inputs = self.embedder(F.softmax(logit, dim=-1).argmax(dim=-1))
        logits = torch.cat(logits, dim=1)
        return logits
    
    def decode_eval(self, source_inputs, target_inputs, latent_inputs):
        """Evaluation-time decoding: always feed model prediction as next-token."""
        batch_size = source_inputs.size(0)
        decode_inputs = self.get_initial_inputs(batch_size)
        state = self.nested_decoder.get_initial_state(source_inputs)
        mask = torch.LongTensor([[1] for _ in range(batch_size)]).to(DEVICE)
        decode_length = target_inputs.size(1)
        logits = []
        for i in range(decode_length):
            outputs, state = self.nested_decoder(decode_inputs, latent_inputs, state)
            logit = self.linear_final(outputs)
            logits.append(logit)
            prediction = F.softmax(logit, dim=-1).argmax(dim=-1)
            decode_inputs = self.embedder(prediction * mask)
            # If hit end symbol, reassign mask cell to 0 to 
            #   turn next predictions to PAD index (hard coded as 0).
            for j, prediction in enumerate(prediction):
                if prediction[0] == self.end_index:
                    mask[j][0] = 0
        logits = torch.cat(logits, dim=1)
        return logits
    
    def forward(self, source, target, training):
        """Decoding.
        
        Args:
            source: torch LongTensor, shape = <batch-size, seq-length>.
            target: same type as `source`, but with a different seq-length.
            training: run train-time decoding if True, evaluation-time decoding else.
        Returns:
            logits: torch FloatTensor, shape = <batch-size, seq-length, vocab-size>.
            kl_loss: tensor (backwardable) float loss if `training` is True, 0 else.
        """
        source_inputs, target_inputs = self.embedder(source), self.embedder(target)

        if training:
            latent_inputs, kl_loss = self.get_latent_inputs(source_inputs, target_inputs, training, mode=1)
            logits = self.decode_train(source_inputs, target_inputs, latent_inputs)

            latent_inputs_test, _ = self.get_latent_inputs(source_inputs, target_inputs, training, mode=0)
            logits_test = self.decode_train(source_inputs, target_inputs, latent_inputs_test)

            return (logits, logits_test), kl_loss
        else:
            latent_inputs, kl_loss = self.get_latent_inputs(source_inputs, target_inputs, training, mode=0)
            logits = self.decode_eval(source_inputs, target_inputs, latent_inputs)
          
            return logits, kl_loss
        
    def predict(self, indexer, sentence, decode_length, number_samples=5,
                location=0, scale=1, pretty_print=True):
        """Generate a number of varied paraphrases.
        
        Args:
            indexer: Indexer object.
            sentence: a string sentence.
            decode_length: number of decoding steps (still stops on hitting end symbol).
            location: mean of random Gaussian for latent states.
            scale: variance of random Gaussian for latent states.
        """
        print("SOURCE >> %s\n" % sentence)
        source = torch.LongTensor([indexer.to_indices(sentence.split() + ["</s>"])]).to(DEVICE)
        source_inputs = self.embedder(source)
        target_placeholder = torch.zeros((1, decode_length)) # used to encode decoding length.
        latent_distribution = latent_inputs = torch.distributions.normal.Normal(location, scale)
        predictions = []
        for i in range(number_samples):
            latent_inputs, kl_loss = self.get_latent_inputs(source_inputs,
                                                            target_inputs=None,
                                                            training=False)
            latent_inputs = latent_distribution.sample(latent_inputs.shape).to(DEVICE)
            logits = self.decode_eval(source_inputs, target_placeholder, latent_inputs)
            predictions.append(get_sample_predictions(indexer, logits))
            if pretty_print:
                print_samples(indexer, logits)
        print("\n")
        return predictions
        

'''
General Class Wrapper around RNNs that supports variational dropout
'''
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers, dropout_rate):
        super(Discriminator, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, number_layers,
                               batch_first=True, bidirectional=True, 
                               dropout=dropout_rate)

        self.output_layer = nn.Linear(hidden_size*2, 1)
        self.critic       = nn.Linear(hidden_size*2, 1)


    def forward(self, sent, embedder, highway):
        source_inputs =  embedder(sent)
        # Encoding
        source_inputs = highway(source_inputs)
        final_state, state = self.encoder(source_inputs.detach())
        # Compute final state.
        #[h_state, c_state] = state
        #final_state = torch.cat([process_state(h_state),
        #                         process_state(c_state)], dim=-1)


        baseline = torch.ones_like(sent[:, [0]]).float() * np.log(0.5)

        disc_logits = self.output_layer(final_state).squeeze(-1)
        baseline_ = self.critic(final_state.detach()).squeeze(-1) 
        baseline = torch.cat([baseline, baseline_], dim=1)[:, :-1]
        
        return disc_logits, baseline


            
# Coherence Checker Modules

class BiLSTMEncoder(nn.Module):
    """Single-layer BiLSTM encoder."""
    
    def __init__(self, input_size, output_size, number_layers):
        """Initializer.
        
        Args:
            input_size: the last dimension of the input tensor.
                        e.g. embed_size fir <batch-size, seq-length, emb_size>.
            output_size: the hidden size of the BiLSTM.
        """
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bilstm = nn.LSTM(input_size, output_size, 
                              num_layers=number_layers,
                              batch_first=True, bidirectional=True)
    
    def forward(self, inputs):
        """Forwarding.
        
        Args:
            inputs: torch.LongTensor of shape <.., input_size>. 
                    `..` for any number of dimensions, 
                    usually <batch-size, seq-length, emb_size>.
        Returns:
            memory: <batch-size, seq-length, 2*hidden-size>.
            states: <2, seq-length, hidden-size>.
        """
        outputs_tuple = self.bilstm(inputs)
        memory, (states, _) = outputs_tuple
        return memory, states
    
    
class Bilinear(nn.Module):
    """Blinear layer."""
    
    def __init__(self, hidden_size):
        """Initializer.
        
        Args:
            hidden_size: the size of bilinear (square) weights.
        """
        super(Bilinear, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, first_inputs, second_inputs):
        """Forwarding.
        
        Args:
            first_inputs: feature matrix of shape <batch-size, hidden-size>.
            second_inputs: another feature matrix with the same shape as `first_inputs`.
        Returns:
            Bilinear (binary) interaction matrix of shape <batch-size, batch-size>.
        """
        return self.linear(first_inputs).mm(second_inputs.transpose(0, 1))
    
    
class BilinearAttention(nn.Module):
    """Luong attention. 
    
    Source: https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
    """
    
    def __init__(self, hidden_size, attention_size):
        """Initializer.
        
        Args:
            hidden_size: size of bi-linear interaction weights.
            attention_size: size of attention vectors.
        """
        super(BilinearAttention, self).__init__()
        self.score = Bilinear(hidden_size)
        self.linear = nn.Linear(hidden_size * 2, attention_size)
    
    def get_attention_weights(self, attendee, attender):
        """Compute attention weights (Eq (1) in source)."""
        unnorm_attention_weights = self.score(attendee, attender)
        attention_weights = F.softmax(unnorm_attention_weights, dim=0)
        return attention_weights
    
    def get_context_vectors(self, attendee, attention_weights):
        """Compute context vectors (Eq (2) in source)."""
        context_vectors = []
        for column_index in range(attention_weights.size(1)):
            context_vector = torch.mul(attendee, 
                                       attention_weights[:, column_index].unsqueeze(0).transpose(0, 1)).sum(dim=0)
            context_vectors.append(context_vector.unsqueeze(0))
        context_vectors = torch.cat(context_vectors, dim=0)
        return context_vectors
    
    def get_attention_vectors(self, attendee, attender):
        """Compute attention vectors (Eq (3) in source)."""
        attention_weights = self.get_attention_weights(attendee, attender)
        context_vectors = self.get_context_vectors(attendee, attention_weights)
        attention_vectors = torch.tanh(self.linear(torch.cat([attender, context_vectors], dim=1)))
        return attention_vectors
    
    
class CoherenceChecker(nn.Module):
    """Coherence checker that scores coherence between context and candidate sentences."""
    
    def __init__(self, embed_size, vocab_size, 
                 number_highway_layers, number_rnn_layers,
                 hidden_size, attention_size,
                 glove_init=None):
        """Initializer."""
        super(CoherenceChecker, self).__init__()
        self.encoder = nn.Sequential(Embeddings(embed_size, vocab_size, glove_init),
                                     HighwayLayer(embed_size, number_highway_layers, nn.ReLU()),
                                     BiLSTMEncoder(embed_size, hidden_size, number_rnn_layers))
        self.attention = BilinearAttention(hidden_size * 2 * number_rnn_layers, attention_size)
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 2 * number_rnn_layers + \
                                                  attention_size, 1),
                                        nn.Sigmoid())
    
    def forward(self, context, candidates):
        """Forward passing.
        
        Args:
            context: torch LongTensor, shape <batch-size, seq-length>.
            candidates: same type as `context`, with different batch-size & seq-length.
        Returns:
            logits: torch FloatTensor, <batch-size (of `candidates`),>.
            predictions: same shape and type as `logits`, with binarily valued cells.
                         1 for cells predicted as coherent with `context`, 0 otherwise.
        """
        # Encode sentences.
        context_batch_size, candidates_batch_size = context.size(0), candidates.size(0)
        _, context_states = self.encoder(context)
        _, candidates_states = self.encoder(candidates)
        context_encoded = context_states.transpose(0, 1).reshape(context_batch_size, -1)
        candidates_encoded = candidates_states.transpose(0, 1).reshape(candidates_batch_size, -1)
        # Apply candidates -> context attention.
        attention_vectors = self.attention.get_attention_vectors(context_encoded,
                                                                 candidates_encoded)
        final_inputs = torch.cat([candidates_encoded, attention_vectors], dim=-1)
        # Classify.
        logits = self.classifier(final_inputs).squeeze()
        # Predict.
        sigmoid_threshold = torch.FloatTensor([0.5] * logits.size(0)).to(DEVICE)
        predictions = (logits > sigmoid_threshold).float()
        return logits, predictions

    
def sentences_to_coherence_checker_inputs(indexer, batch_sentences):
    """Convert a batch of sentences to inputs to CoherenceChecker (either context or candidates)."""
    batch_indices = batch_sentences_to_indices(indexer, batch_sentences)
    inputs = torch.LongTensor(pad_sentences(batch_indices,
                                            indexer.get_index("PAD", add=False))).to(DEVICE)
    return inputs
