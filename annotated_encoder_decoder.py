#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tensorboardX import SummaryWriter
from tqdm import tqdm as tqdm

USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

writer = SummaryWriter()

# ## Model class
# 
# Our base model class `EncoderDecoder` is very similar to the one in *The Annotated Transformer*.
# 
# One difference is that our encoder also returns its final states (`encoder_final` below), which is used to initialize the decoder RNN. We also provide the sequence lengths as the RNNs require those.

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)


# To keep things easy we also keep the `Generator` class the same. 
# It simply projects the pre-output layer ($x$ in the `forward` function below) to obtain the output layer, so that the final dimension is the target vocabulary size.

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ## Encoder
# 
# Our encoder is a bi-directional LSTM. 
# 
# Because we want to process multiple sentences at the same time for speed reasons (it is more effcient on GPU), we need to support **mini-batches**. Sentences in a mini-batch may have different lengths, which means that the RNN needs to unroll further for certain sentences while it might already have finished for others:
# 
# ```
# Example: mini-batch with 3 source sentences of different lengths (7, 5, and 3).
# End-of-sequence is marked with a "3" here, and padding positions with "1".
# 
# +---------------+
# | 4 5 9 8 7 8 3 |
# +---------------+
# | 5 4 8 7 3 1 1 |
# +---------------+
# | 5 8 3 1 1 1 1 |
# +---------------+
# ```
# You can see that, when computing hidden states for this mini-batch, for sentence #2 and #3 we will need to stop updating the hidden state after we have encountered "3". We don't want to incorporate the padding values (1s).
# 
# Luckily, PyTorch has convenient helper functions called `pack_padded_sequence` and `pad_packed_sequence`.
# These functions take care of masking and padding, so that the resulting word representations are simply zeros after a sentence stops.
# 
# The code below reads in a source sentence (a sequence of word embeddings) and produces the hidden states.
# It also returns a final vector, a summary of the complete sentence, by concatenating the first and the last hidden states (they have both seen the whole sentence, each in a different direction). We will use the final vector to initialize the decoder.

class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # output is of size (batch_length, seq_length, num_directions*hidden_size)
        # final[0] is of size (num_layers*num_directions, batch_length, hidden_size)
        # final[0][0] is fwd for first layer. final[0][2] is forward for second layer.

        # we need to manually concatenate the final states for both directions
        # final is a tuple of (hidden, cell)
        #print("output")
        #print(output.size())
        #print("final size")
        #print(final[0].size())
        fwd_final_hidden = final[0][0:final[0].size(0):2]
        #print("final_fwd size")
        #print(fwd_final_hidden.size())
        bwd_final_hidden = final[0][1:final[0].size(0):2]
        final_hidden = torch.cat([fwd_final_hidden, bwd_final_hidden], dim=2)  # [num_layers, batch, 2*dim]

        fwd_final_cell = final[1][0:final[1].size(0):2]
        bwd_final_cell = final[1][1:final[1].size(0):2]
        final_cell = torch.cat([fwd_final_cell, bwd_final_cell], dim=2)  # [num_layers, batch, 2*dim]
        return output, (final_hidden, final_cell)


# ### Decoder
# 
# The decoder is a conditional LSTM. Rather than starting with an empty state like the encoder, its initial hidden state results from a projection of the encoder final vector. 
# 
# #### Training
# In `forward` you can find a for-loop that computes the decoder hidden states one time step at a time. 
# Note that, during training, we know exactly what the target words should be! (They are in `trg_embed`.) This means that we are not even checking here what the prediction is! We simply feed the correct previous target word embedding to the LSTM at each time step. This is called teacher forcing.
# 
# The `forward` function returns all decoder hidden states and pre-output vectors. Elsewhere these are used to compute the loss, after which the parameters are updated.
# 
# #### Prediction
# For prediction time, for forward function is only used for a single time step. After predicting a word from the returned pre-output vector, we can call it again, supplying it the word embedding of the previously predicted word and the last state.


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
                 
        self.rnn = nn.LSTM(emb_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge_hidden = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None
        self.bridge_cell = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        #we only want the hidden, not the cell state of the lstm CZW, hence the hidden[0]
        query = hidden[0][-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        #print("encoder final shape")
        #print(encoder_final[0].size())
        if encoder_final is None:
            return None  # start with zeros

        return (torch.tanh(self.bridge_hidden(encoder_final[0])),
                torch.tanh(self.bridge_cell(encoder_final[1])))


# ### Attention                                                                                                                                                                               
# 
# At every time step, the decoder has access to *all* source word representations $\mathbf{h}_1, \dots, \mathbf{h}_M$. 
# An attention mechanism allows the model to focus on the currently most relevant part of the source sentence.
# The state of the decoder is represented by GRU hidden state $\mathbf{s}_i$.
# So if we want to know which source word representation(s) $\mathbf{h}_j$ are most relevant, we will need to define a function that takes those two things as input.
# 
# Here we use the MLP-based, additive attention that was used in Bahdanau et al.
# 
# 
# We apply an MLP with tanh-activation to both the current decoder state $\bf s_i$ (the *query*) and each encoder state $\bf h_j$ (the *key*), and then project this to a single value (i.e. a scalar) to get the *attention energy* $e_{ij}$. 
# 
# Once all energies are computed, they are normalized by a softmax so that they sum to one: 
# 
# $$ \alpha_{ij} = \text{softmax}(\mathbf{e}_i)[j] $$
# 
# $$\sum_j \alpha_{ij} = 1.0$$ 
# 
# The context vector for time step $i$ is then a weighted sum of the encoder hidden states (the *values*):
# $$\mathbf{c}_i = \sum_j \alpha_{ij} \mathbf{h}_j$$

class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        #print(query.size())
        #print(proj_key.size())
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


# ## Embeddings and Softmax                                                                                                                                                                                                                                    
# We use learned embeddings to convert the input tokens and output tokens to vectors of dimension `emb_size`.
# 
# We will simply use PyTorch's [nn.Embedding](https://pytorch.org/docs/stable/nn.html?highlight=embedding#torch.nn.Embedding) class.

# ## Full Model
# 
# Here we define a function from hyperparameters to a full model. 


def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab))

    return model.cuda() if USE_CUDA else model


# # Training
# 
# This section describes the training regime for our models.

# We stop for a quick interlude to introduce some of the tools 
# needed to train a standard encoder decoder model. First we define a batch object that holds the src and target sentences for training, as well as their lengths and masks. 

# ## Batches and Masking


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()

# ## Training Loop
# The code below trains the model for 1 epoch (=1 pass through the training data).


def run_epoch(data_iter, model, loss_compute, print_every=50, num_batches=0):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in tqdm(enumerate(data_iter, 1)):
        
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

        if num_batches > 0 and i > num_batches:
            break

    return math.exp(total_loss / float(total_tokens))

class OracleCriterion:
    
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, x, y):
        #x is a list of (vocab_size) vectors in batches. ((batch*seq_length) x vocab_size)
        #y is a list of indicies into a vocab_size vector (batch*seq_length)
        mask = y.ne(self.pad_index) # this is a mask that is the shape of a 1D list 
        #print("czw, x.size()", x.size())
        #print("czw mask.size()", mask.size())
        #print("czw x[torch.arange(x.size(0)), y]", x[torch.arange(x.size(0)),y].size())
        probs = x[torch.arange(x.size(0)), y] #index select
        masked_probs = torch.masked_select(probs, mask)
        loss = -1*torch.sum(masked_probs)
        #print("czw loss", loss)
        return loss
        

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)

        #print("czw x", x.contiguous().view(-1, x.size(-1)).size())
        #print("czw y", y.contiguous().view(-1).size())
        #x.view(-1,x.size(-1)) creates a block that is (n_seq*seq_leq, vocab_size)
        #y.view creates a 1D list of indices
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        #use the below for NLLLoss()
        #loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
        #                      y.contiguous().view(-1))
        #print("czw x", x.view(-1, x.size(-1)))
        #print("czw y", y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm

class BeamNode():
    def __init__(self, prev_input, prev_h, logProb, words=[], attention_scores=[]):
        self.words = words
        self.prev_input = prev_input
        self.prev_h = prev_h
        self.logProb = logProb
        self.attention_scores = attention_scores

def beam_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None, beam_size=5):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)

    output = []
    hidden = None

    #for i in range(max_len):
    i = 0
    beam_nodes = []
    beam_nodes.append(BeamNode(sos_index, hidden, 0))
    ended = False
    while i<max_len and not ended:
        new_nodes = []
        for node in beam_nodes:
            prev_word = node.prev_input
            prev_y = torch.ones(1, 1).fill_(prev_word).type_as(src)
            trg_mask = torch.ones_like(prev_y)
            hidden = node.prev_h
            with torch.no_grad():
                out, hidden, pre_output = model.decode(
                  encoder_hidden, encoder_final, src_mask,
                  prev_y, trg_mask, hidden)

                # we predict from the pre-output layer, which is
                # a combination of Decoder state, prev emb, and context
                prob = model.generator(pre_output[:, -1])

            probs, words = torch.topk(prob, beam_size, dim=1)
            #print(probs, words)
            probs = probs.squeeze().cpu().numpy()
            words = words.squeeze().cpu().numpy()
            #print([lookup_words(x, TRG.vocab) for x in words])
#            print(lookup_words(words, TRG.vocab))
            #print(probs)
            #print(words)
            for j in range(len(probs)):
                #print(j)
                probj = probs[j]
                next_word = words[j]
                #print(probi)
                #print(wordi)
                new_words = node.words.copy() + [next_word]
                new_prob = node.logProb + probj
                new_node = BeamNode(next_word, hidden, new_prob, words=new_words, attention_scores=node.attention_scores.copy())
                new_node.attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
                new_nodes.append(new_node)
        i+=1
        #print("first", len(beam_nodes))
        beam_nodes = sorted(new_nodes, key=lambda node: -node.logProb)[:beam_size] 
        #print(lookup_words([n.prev_input for n in beam_nodes], TRG.vocab))
        #print([n.logProb for n in beam_nodes])
        #print([n.logProb for n in beam_nodes])
        #print(len(beam_nodes))
        ended = any([True if node.prev_input==eos_index else False for node in beam_nodes])
        #print(ended)
    output = []
    attns = []
    if ended:
        end_node_i = [1 if node.prev_input==eos_index else 0 for node in beam_nodes].index(1)
        end_node = beam_nodes[end_node_i]
        output = np.array(end_node.words[:-1])
    else:
        end_node = beam_nodes[0]
        output = np.array(end_node.words)
    #print(end_node.attention_scores) 
    #print(np.array(end_node.attention_scores).shape) 
    #print([x.shape for x in end_node.attention_scores])
    #print(output)
    return output, np.concatenate(np.array(end_node.attention_scores), axis=1)
  

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]


def print_examples(example_iter, model, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = beam_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()
        
        count += 1
        if count == n:
            break

# ## Data Loading
# 
# We will load the dataset using torchtext and spacy for tokenization.
# 
# This cell might take a while to run the first time, as it will download and tokenize the IWSLT data.
# 
# For speed we only include short sentences, and we include a word in the vocabulary only if it occurs at least 5 times. In this case we also lowercase the data.

# For data loading.
from torchtext import data, datasets

import spacy

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"    
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
LOWER = True

# we include lengths to provide to the RNNs
SRC = data.Field(batch_first=True, lower=LOWER, include_lengths=True,
                 unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)
TRG = data.Field(batch_first=True, lower=LOWER, include_lengths=True,
                 unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

#MAX_LEN = 100  # NOTE: we filter out a lot of sentences for speed
# ### Let's look at the data
# 
# It never hurts to look at your data and some statistics.


def print_data_info(my_data, src_field, trg_field):
    """ This prints some useful stuff about our data sets. """
    train_data = my_data["train"]
    valid_data = my_data["val"]
    test_data = my_data["test"]

    print("Data set sizes (number of sentence pairs):")
    print('train', len(train_data))
    print('valid', len(valid_data))
    print('test', len(test_data), "\n")

    print("First training example:")
    print("src:", " ".join(vars(train_data[0])['src']))
    print("trg:", " ".join(vars(train_data[0])['trg']), "\n")

    print("Most common words (src):")
    print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
    print("Most common words (trg):")
    print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words (src):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
    print("First 10 words (trg):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

    print("Number of NL words (types):", len(src_field.vocab))
    print("Number of AMR words (types):", len(trg_field.vocab), "\n")
    
    

def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.src, batch.trg, pad_idx)


# ## Training the System
# 
# Now we train the model. 
# 
# On a Titan X GPU, this runs at ~18,000 tokens per second with a batch size of 64.

def train(model, num_epochs=10, lr=0.0003, print_every=100, num_batches=0):
    """Train a model on IWSLT"""
    
    if USE_CUDA:
        model.cuda()

    # optionally add label smoothing; see the Annotated Transformer
    #criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    criterion = OracleCriterion(PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    dev_perplexities = []
    min_perplexity = 1000

    for epoch in range(num_epochs):
      
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), 
                                     model,
                                     SimpleLossCompute(model.generator, criterion, optim),
                                     print_every=print_every,
                                     num_batches=num_batches)
        
        model.eval()
        with torch.no_grad():
            print_examples((rebatch(PAD_INDEX, x) for x in valid_iter), 
                           model, n=3, src_vocab=SRC.vocab, trg_vocab=TRG.vocab)        

            dev_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter), 
                                       model, 
                                       SimpleLossCompute(model.generator, criterion, None))
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)
            if dev_perplexity < min_perplexity:
                torch.save(model, "best_model.pt")
                min_perplexity = dev_perplexity
        
    model = torch.load("best_model.pt")
    return dev_perplexities
        


import sacrebleu

def eval_val(file_name, model, valid_iter, targ_field, datasets):
    references = [" ".join(example.trg) for example in datasets["val"]]

    hypotheses = []
    alphas = []  # save the last attention scores
    for batch in tqdm(valid_iter):
      batch = rebatch(PAD_INDEX, batch)
      pred, attention = beam_decode(
        model, batch.src, batch.src_mask, batch.src_lengths,
        sos_index=targ_field.vocab.stoi[SOS_TOKEN],
        eos_index=targ_field.vocab.stoi[EOS_TOKEN])
      hypotheses.append(pred)
      alphas.append(attention)

    hypotheses = [lookup_words(x, TRG.vocab) for x in hypotheses]
    hypotheses = [" ".join(x) for x in hypotheses]

    with open(file_name, "w") as file:
        for line in hypotheses:
            file.write(line+"\n")

    bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
    print(bleu)

for num_batches in [100]:
    my_data = {}

    for split in ["train", "val", "test"]:
        my_data[split] = datasets.TranslationDataset(path="data/new_"+split,
                        exts=('.nl', '.amr'), fields=(SRC, TRG))
                        #filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                        #    len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
    SRC.build_vocab(my_data["train"].src, min_freq=MIN_FREQ)
    TRG.build_vocab(my_data["train"].trg, min_freq=MIN_FREQ)

    PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]

    print_data_info(my_data, SRC, TRG)
    train_iter = data.BucketIterator(my_data["train"], batch_size=100, train=True, 
                                     sort_within_batch=True, 
                                     sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                     device=DEVICE)

    valid_iter = data.Iterator(my_data["val"], batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

    model = make_model(len(SRC.vocab), len(TRG.vocab),
                       emb_size=500, hidden_size=500,
                       num_layers=2, dropout=0.5)
    dev_perplexities = train(model, num_epochs=60, print_every=500, num_batches=num_batches)
    torch.save(model, str(num_batches) + ".pt")
    file_name = str(num_batches) + "_batches.pred"
    eval_val(file_name, model, valid_iter, TRG, my_data)

# ## Attention Visualization
# 
# We can also visualize the attention scores of the decoder.

def plot_heatmap(src, trg, scores):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis')

    ax.set_xticklabels(trg, minor=False, rotation='vertical')
    ax.set_yticklabels(src, minor=False)

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    #plt.show()
    plt.savefig("attn_map.png")



# This plots a chosen sentence, for which we saved the attention scores above.
#idx = 5
#src = my_data["val"][idx].src + ["</s>"]
#trg = my_data["val"][idx].trg + ["</s>"]
#pred = hypotheses[idx].split() + ["</s>"]
#pred_att = alphas[idx][0].T[:, :len(pred)]
#print("src", src)
#print("ref", trg)
#print("pred", pred)
#plot_heatmap(src, pred, pred_att)
