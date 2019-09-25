#!/storage/czw/anaconda3/envs/anno/bin/python3
# coding: utf-8
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tensorboardX import SummaryWriter
from tqdm import tqdm as tqdm

from torchtext import data, datasets

import spacy

USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False 
DEVICE=torch.device('cuda') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#writer = SummaryWriter()
class Autoencoder(nn.Module):
    def __init__(self, EncoderDecoder1, EncoderDecoder2, amr_eos_token):
        super(Autoencoder, self).__init__()
        self.enc_dec_1 = EncoderDecoder1
        self.enc_dec_2 = EncoderDecoder2
    
    def encode(self, nl_src, nl_src_mask, nl_src_lengths, amr_src, amr_src_mask):
        """
            nl_src is [batch, seq_len] (ends with end token?)
            nl_src_mask is [batch, 1, seq_len]
            nl_src_lengths is [batch]
            amr_src is [batch, seq_len] (begins with start token?)
            amr_src_mask is [batch, seq_len]
        """
        encoder_hidden, encoder_final = self.enc_dec_1.encode(nl_src, nl_src_mask, nl_src_lengths)
        print("czw encode encoder_hidden", encoder_hidden)
        out, hidden, pre_output = self.enc_dec_1.decode(encoder_hidden, encoder_final, nl_src_mask, amr_src, amr_src_mask)
        #out is [batch, seq_len, hidden_size]
        #pre_output is concat of rnn out, context, and prev_embed
        #which has been fed through a linear layer
        pred = self.enc_dec_1.generator(pre_output)
        #out is [batch, seq_len, vocab_size]
        pred = torch.max(pred, dim=2)[1]
        #out is [batch, seq_len]
        #print(pred.size())
        #TODO get rid of magic number
        inter_src, inter_mask, inter_lengths = get_intermediary_batch(pred, 3)
        print("czw auto inter_src", inter_src.size())
        print("czw auto inter_src_mask", inter_mask.size())
        print("czw auto inter_src_lengths", inter_lengths.size())
        encoder_hidden, encoder_final = self.enc_dec_2.encode(inter_src, inter_mask, inter_lengths)
        print("czw auto src (as target)" , nl_src.size())
        return encoder_hidden, encoder_final, inter_mask
    
    def decode(self, encoder_hidden, encoder_final, inter_mask, nl_src, nl_src_mask,
               decoder_hidden=None):
        out, hidden, pre_output = self.enc_dec_2.decode(encoder_hidden, encoder_final, inter_mask, nl_src, nl_src_mask, decoder_hidden=decoder_hidden)
        return out, hidden, pre_output


    def forward(self, nl_src, amr_src, nl_src_mask, amr_src_mask, nl_src_lengths, amr_src_lengths):
        """Take in and process masked src and target sequences.
            nl_src is [batch, seq_len] (ends with end token?)
            amr_src is [batch, seq_len] (begins with start token?)
            nl_src_mask is [batch, 1, seq_len]
            amr_src_mask is [batch, seq_len]
            nl_src_lengths is [batch]
            amr_src_lengths is [batch]
        """
        encoder_hidden, encoder_final, inter_mask = self.encode(nl_src, nl_src_mask, nl_src_lengths, amr_src, amr_src_mask)
        return self.decode(encoder_hidden, encoder_final, inter_mask, nl_src, nl_src_mask)

def get_intermediary_batch(src, amr_eos_token):
    """
        create batch of amrs as intermediary representation
        src is [batch, seq_len]
    """
    #TODO decide whether to pad the src
    #is it enough to keep track of the lengths? It looks like that is 
    #the only thing that matters for pad_pack and pack_pad
    np_src = src.detach().cpu().numpy()
    mask = []
    lengths = []
    max_len = np_src.shape[1]
    for i in range(np_src.shape[0]):
        row = np_src[i]
        eos_indices = np.where(row==amr_eos_token)[0]
        length = max_len
    #    print(row)
    #    print(eos_indices)
        if len(eos_indices) > 0:
            length = eos_indices[0] + 1
        lengths.append(length)
        mask.append([[0]*length + [1]*(max_len - length)])
        #print (row)
    #print(lengths)
    tensor_mask = torch.Tensor(mask).cuda() if USE_CUDA else torch.Tensor(mask)
    tensor_lengths = torch.Tensor(lengths).cuda() if USE_CUDA else torch.Tensor(lengths)
    return src, tensor_mask, tensor_lengths 

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
        print("czw encoderdecoder encode src", src)
        print("czw encoderdecoder encode src lengths", src_lengths)
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
        """
            In most cases, x is (batch_len x seq_len x hidden_size)
        """
        return F.log_softmax(self.proj(x), dim=-1)


# ## Encoder
# 
# Our encoder is a bi-directional LSTM. 
# 
# Because we want to process multiple sentences at the same time for speed reasons (it is more effcient on GPU), we need to support **mini-batches**. Sentences in a mini-batch may have different lengths, which means that the RNN needs to unroll further for certain sentences while it might already have finished for others:
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
        mask is [batch, 1, seq_len]
        """
        print("czw encoder x", x.size())
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # output is of size (batch_length, seq_length, num_directions*hidden_size)

        # we need to manually concatenate the final states for both directions
        # final is a tuple of (hidden, cell)
        # final[0] is the final hidden state. It has size (num_layers*num_directions, batch_length, hidden_size)
        # final[0][0] is the fwd direction for first layer. final[0][2] is forward for second layer and so on.
        fwd_final_hidden = final[0][0:final[0].size(0):2]# [num_layers, batch_len, dim]
        bwd_final_hidden = final[0][1:final[0].size(0):2]
        final_hidden = torch.cat([fwd_final_hidden, bwd_final_hidden], dim=2)  # [num_layers, batch, num_directions*dim]

        fwd_final_cell = final[1][0:final[1].size(0):2]
        bwd_final_cell = final[1][1:final[1].size(0):2]
        final_cell = torch.cat([fwd_final_cell, bwd_final_cell], dim=2)  # [num_layers, batch, num_directions*dim]
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
    
    def __init__(self, emb_size, hidden_size, attention=None, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        #the LSTM takes                 
        self.rnn = nn.LSTM(emb_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge_hidden = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None
        self.bridge_cell = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)

        #input is prev embedding (emb_size), output (hidden_size), and context (num_directions*hidden_size)
        #output is a vector of size (hidden_size). This vector is fed through a softmax layer to get a 
        #distribution over vocab
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)
           prev_embed (batch_size x seq_len x emb_size) is the target previous word
           encoder_hidden is atuple of elements with size (batch_size x seq_len x num_directions*hidden_size) is the output of the encoder
           hidden (batch_size x 1 x hidden_size) is the forward hidden state of the previous time step
        """

        # compute context vector using attention mechanism
        #we only want the hidden, not the cell state of the lstm CZW, hence the hidden[0]
        query = hidden[0][-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        # context is batch x 1 x num_directions*hidden_size
        # the lstm takes the previous target embedding and the attention context as input
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        # at this stage, pre_output batch x seq_len x hidden_size
        # output is batch x seq_len x hidden_size 
        # hidden is a tuple of hidden_cell, hidden_state
        # pre_output is actually used to compute the prediction
        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time.
           trg_embed is (batch_len x trg_max_seq_len x hidden_layer_size)
           encoder_hidden is (batch_len x src_max_seq_len x num_directions*hidden_layer_size)
           encoder_final is a tuple of final hidden and final cell state. 
             each state is (num_layers x batch x num_directions*hidden_layer_size)
           src_mask is (batch_len x 1 x src_max_seq_len)
        """
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_embed.size(1)

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
        # teacher forcing
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
def make_autoencoder(nl_vocab, amr_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    nl_embed = nn.Embedding(nl_vocab, emb_size)
    amr_embed = nn.Embedding(amr_vocab, emb_size)

    EncoderDecoder1 = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nl_embed, #src_embed 
        amr_embed, #tgt_embed 
        Generator(hidden_size, amr_vocab))

    EncoderDecoder2 = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        amr_embed, #src_embed
        nl_embed, #tgt_embed
        Generator(hidden_size, nl_vocab))

    #TODO magic number
    model = Autoencoder(EncoderDecoder1, EncoderDecoder2, 3)
    return model.cuda() if USE_CUDA else model


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
    #TODO make this hold three elements: source NL, intermediary AMR, tgt NL
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
def run_epoch(data_iter, model, loss_compute, print_every=50, num_batches=0, phase="train", epoch_num=0):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in tqdm(enumerate(data_iter, 1)):
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        print("czw batch src", type(batch.src), batch.src.size())
        print("czw batch trg", type(batch.trg), batch.trg.size())
        print("czw batch src mask", type(batch.src_mask), batch.src_mask.size())
        print("czw batch trg mask", type(batch.trg_mask), batch.trg_mask.size())
        print("czw batch src_lengths", type(batch.src_lengths), batch.src_lengths.size())
        print("czw out", out.size())
        print("czw pre_output", pre_output.size())

        #pred = model.generator(pre_output)
        #out is [batch, seq_len, vocab_size]
        #pred = torch.max(pred, dim=2)[1]
        #out is [batch, seq_len]
        #print("czw pred", pred.size())
        #amr_src, amr_lengths, amr_mask = get_intermediary_batch(pred, 3)
        #print("czw amr_src", amr_src.size())
        #print("czw amr_lengths", amr_lengths.size())
        #print("czw amr_mask", amr_mask.size())
 
        print("begin loss compute")
        loss = loss_compute(pre_output, batch.src, batch.nseqs)
        print("end loss compute")
        #print(f'epoch loss {loss}')
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

    return math.exp(total_loss / float(total_tokens)), total_loss
    #return total_loss / float(total_tokens)

class myNLL:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, x, y):
        #x is a list of (vocab_size) vectors in batches. ((batch*seq_length) x vocab_size)
        #y is a list of indicies into a vocab_size vector (batch*seq_length)
        mask = y.ne(self.pad_index) # this is a mask that is the shape of a 1D list 
        probs = x[torch.arange(x.size(0)), y] #index select
        masked_probs = torch.masked_select(probs, mask)
        loss = -torch.sum(masked_probs)#sum for log prob
        #print("czw loss", loss)
        return loss

class PermissiveCriterion:
    
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, x, y):
        #x is batch x seq_len x vocab_size
        #y is batch x seq_len
        mask = y.ne(self.pad_index)
        batch_len = y.size(0)
        flat_x = x.view(-1, x.size(-1))
        flat_y = y.view(-1)
        probs = flat_x[torch.arange(flat_x.size(0)), flat_y]
        probs = probs.view(batch_len, -1) #probs is now the same shape as y
        losses = torch.zeros(batch_len)
        for i in range(probs.size(0)):
            if torch.masked_select(probs[i], mask[i]).size(0) < 5:
                losses[i] = 1 - torch.exp(torch.max(torch.masked_select(probs[i], mask[i])))
            else:
                top5, _ = torch.topk(torch.masked_select(probs[i], mask[i]), 5)
                losses[i] = 5 - torch.sum(torch.exp(top5))
        return torch.sum(losses)
            
       
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
        probs = torch.exp(x[torch.arange(x.size(0)), y]) #index select
        #print("czw probs", probs)
        #print("czw probs size", probs.size())
        masked_probs = torch.masked_select(probs, mask)
        loss = 1-torch.prod(masked_probs)#sum for log prob
        #print("czw loss", loss)
        return loss

class TokenNoiseLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, error_per, trg_vocab_size, pad_index,opt=None):
        self.generator = generator
        self.criterion = criterion
        self.error_per = error_per
        self.opt = opt
        self.trg_vocab_size = trg_vocab_size
        self.pad_index = pad_index

    def __call__(self, x, y, norm):
        x = self.generator(x)
        np_y = y.contiguous().view(-1).detach().cpu().numpy()
        new_y = [
                    np.random.randint(1+self.pad_index, self.trg_vocab_size)
                    if random.random() < self.error_per and x != self.pad_index
                    else x
                    for x in np_y
                ]
        new_y = torch.Tensor(new_y).type_as(y)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              new_y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm
 
class NoisyLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, error_per, trg_vocab_size, pad_index,opt=None):
        self.generator = generator
        self.criterion = criterion
        self.error_per = error_per
        self.opt = opt
        self.trg_vocab_size = trg_vocab_size
        self.pad_index = pad_index

    def __call__(self, x, y, norm):
        #y is (n_seq, seq_len)
        #print(x.size(), y.size())
        x = self.generator(x)
        new_y = torch.ones_like(y)
        #count1 = 0
        #count2 = 0
        for i in range(y.size(0)):#iterate over the seqs in a batch
            if random.random() < self.error_per:
                rand_size = np.random.randint(y.size(1)-1)
                random_targ = np.zeros(y.size(1)).astype(int)
                random_targ.fill(self.pad_index)
                #random_targ.fill(18)
                random_targ[:rand_size] = np.random.randint(self.pad_index+1, self.trg_vocab_size, size=rand_size)
                #rand_words = lookup_words(random_targ, AMR_SRC.vocab)
                #rand_words = " ".join(rand_words)
                #print(rand_words)
 
         #       count1 += 1
                new_y[i] = torch.Tensor(random_targ).type_as(y)
                #print(new_y[i])
            else:
         #       count2 += 1
                new_y[i] = y[i]
        #print(count1, count2)
        #x.view(-1,x.size(-1)) creates a block that is (n_seq*seq_leq, vocab_size)
        #y.view creates a 1D list of indices
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              new_y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm
 
class PermissiveLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        #x is n_seq x seq_len x vocab_size
        #y is n_seq * seq_len
        x = self.generator(x)

        loss = self.criterion(x.contiguous(),
                              y.contiguous())
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        print("czw x", x.size())
        print("czw y", y.size())
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

def beam_decode(model, src, src_mask, src_lengths, trg, trg_mask, max_len=100, sos_index=1, eos_index=None, beam_size=5):

    #src is nl
    #trg is amr

    with torch.no_grad():
        encoder_hidden, encoder_final, inter_mask = model.encode(src, src_mask, src_lengths, trg, trg_mask)

    output = []
    hidden = None

    i = 0
    beam_nodes = []
    beam_nodes.append(BeamNode(sos_index, hidden, 0))
    ended = False #Flag raised when EOS token found
    while i<max_len and not ended:
        new_nodes = []
        for node in beam_nodes:
            prev_word = node.prev_input
            #print('czw prev word', prev_word)
            prev_y = torch.ones(1, 1).fill_(prev_word).type_as(src)
            trg_mask = torch.ones_like(prev_y)
            hidden = node.prev_h
            with torch.no_grad():
                out, hidden, pre_output = model.decode(
                  encoder_hidden, encoder_final, inter_mask,
                  prev_y, trg_mask, hidden)

                # we predict from the pre-output layer, which is
                # a combination of Decoder state, prev emb, and context
                prob = model.enc_dec_2.generator(pre_output[:, -1])
                #print("czw pre_output", pre_output[:,-1])
            probs, words = torch.topk(prob, beam_size, dim=1)
            probs = probs.squeeze().cpu().numpy()
            words = words.squeeze().cpu().numpy()
            print("czw words", words)
            print("czw probs", probs)

            for j in range(len(probs)):
                probj = probs[j]
                next_word = words[j]
                new_words = node.words.copy() + [next_word]
                new_prob = node.logProb + probj
                new_node = BeamNode(next_word, hidden, new_prob, words=new_words, attention_scores=node.attention_scores.copy())
                new_node.attention_scores.append(model.enc_dec_2.decoder.attention.alphas.cpu().numpy())
                new_nodes.append(new_node)
        i+=1
        beam_nodes = sorted(new_nodes, key=lambda node: -node.logProb)[:beam_size] 
        ended = any([True if node.prev_input==eos_index else False for node in beam_nodes])

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
        src_sos_index = trg_vocab.stoi[SOS_TOKEN]
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
          batch.trg, batch.trg_mask,
          max_len=max_len, sos_index=src_sos_index, eos_index=src_eos_index)
        print("Example #%d" % (i+1))
        print("NL : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("AMR : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=src_vocab)))
        print()
        
        count += 1
        if count == n:
            break

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"    
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
LOWER = True

# we include lengths to provide to the RNNs
NL_SRC = data.Field(batch_first=True, lower=LOWER, include_lengths=True,
                 unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)
AMR_SRC = data.Field(batch_first=True, lower=LOWER, include_lengths=True,
                 unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

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

def train(model, num_epochs=10, lr=0.0003, print_every=100, num_batches=0, error_per=0):
    """Train a model on IWSLT"""
    
    if USE_CUDA:
        model.cuda()

    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    #criterion = OracleCriterion(PAD_INDEX)
    #criterion = PermissiveCriterion(PAD_INDEX)
    #criterion = myNLL(PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    dev_perplexities = []
    min_perplexity = float('inf')

    for epoch in range(num_epochs):
      
        print("Epoch", epoch)
        model.train()
        train_perplexity, train_loss = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), 
                                     model,
                                     SimpleLossCompute(model.enc_dec_2.generator, criterion, optim),
                                     #TokenNoiseLossCompute(model.generator, criterion, error_per, len(AMR_SRC.vocab), PAD_INDEX, optim),
                                     #NoisyLossCompute(model.generator, criterion, error_per, len(AMR_SRC.vocab), PAD_INDEX, optim),
                                     #PermissiveLossCompute(model.generator, criterion, optim),
                                     print_every=print_every,
                                     num_batches=num_batches,
                                     phase="train")
        
        print("train loss", train_loss)
        #writer.add_scalar("train loss", train_loss, epoch)
        model.eval()
        with torch.no_grad():
            #print("val examples")
            #print_examples((rebatch(PAD_INDEX, x) for x in valid_iter), 
            #               model, n=3, src_vocab=NL_SRC.vocab, trg_vocab=AMR_SRC.vocab)        

            #print("train examples")
            #print_examples((rebatch(PAD_INDEX, x) for x in train_iter), 
            #               model, n=3, src_vocab=NL_SRC.vocab, trg_vocab=AMR_SRC.vocab)        

            dev_perplexity, val_loss = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter), 
                                       model, 
                                       SimpleLossCompute(model.generator, criterion, None), 
                                       phase="val")
            #writer.add_scalar("val loss", val_loss, epoch)
            print("Validation perplexity: %f" % dev_perplexity)
            print("val loss", val_loss)
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
    for batch in valid_iter:
      batch = rebatch(PAD_INDEX, batch)
      pred, attention = beam_decode(
        model, batch.src, batch.src_mask, batch.src_lengths,
        batch.trg, batch.trg_mask,
        sos_index=targ_field.vocab.stoi[SOS_TOKEN],
        eos_index=targ_field.vocab.stoi[EOS_TOKEN])
      hypotheses.append(pred)
      alphas.append(attention)

    hypotheses = [lookup_words(x, NL_SRC.vocab) for x in hypotheses]
    hypotheses = [" ".join(x) for x in hypotheses]

    with open(file_name, "w") as file:
        for line in hypotheses:
            file.write(line+"\n")

    bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
    print(bleu)

for error in range(1,2):
    my_data = {}
    num_batches=10
    error_per = error/10. 

    for split in ["train", "val", "test"]:
        my_data[split] = datasets.TranslationDataset(path="data/new_"+split,
                        exts=('.nl', '.amr'), fields=(NL_SRC, AMR_SRC))
    MIN_FREQ = 5
    NL_SRC.build_vocab(my_data["train"].src, min_freq=MIN_FREQ)
    AMR_SRC.build_vocab(my_data["train"].trg, min_freq=MIN_FREQ)

    PAD_INDEX = AMR_SRC.vocab.stoi[PAD_TOKEN]

    print_data_info(my_data, NL_SRC, AMR_SRC)
    train_iter = data.BucketIterator(my_data["train"], batch_size=100, train=True, 
                                     sort_within_batch=True, 
                                     sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                     device=DEVICE)

    valid_iter = data.Iterator(my_data["val"], batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

    model = make_autoencoder(len(NL_SRC.vocab), len(AMR_SRC.vocab),
                       emb_size=500, hidden_size=500,
                       num_layers=2, dropout=0.5)
    dev_perplexities = train(model, num_epochs=15, print_every=500, num_batches=num_batches, error_per=error_per)
    #torch.save(model, "reverse.pt")
    #file_name = "reverse.pred"
    print(file_name)
    eval_val(file_name, model, valid_iter, AMR_SRC, my_data)

#writer.close()
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
