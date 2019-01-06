import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data
import argparse
import time
import csv
import torch.nn as nn
from torch import optim
import math
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import ParameterSampler, ParameterGrid
from collections import namedtuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import torch.nn.functional as F
import Levenshtein as L
from torch.utils.data.dataloader import _use_shared_memory
import time
import sys
import collections


os.environ['WSJ_PATH']='../'

class WSJ():
    """ reuse code from hw1
        
        Ensure WSJ_PATH is path to directory containing 
        all data files (.npy) provided on Kaggle.
        
        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)
            
    """
  
    def __init__(self):
        self.dev_set = None
        self.dev_label = None
        self.train_set = None
        self.train_label = None
        self.test_set = None
        
    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set, self.dev_label = load_raw(os.environ['WSJ_PATH'], 'dev')
        return self.dev_set, self.dev_label

    @property
    def train(self):
        if self.train_set is None:
            self.train_set, self.train_label = load_raw(os.environ['WSJ_PATH'], 'train')
        return self.train_set, self.train_label
  
    @property
    def test1(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'AIRBUS_test_features1.npy'), encoding='bytes'), None)
        return self.test_set
    @property
    def test2(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'AIRBUS_test_features2.npy'), encoding='bytes'), None)
        return self.test_set
    @property
    def test3(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'AIRBUS_test_features3.npy'), encoding='bytes'), None)
        return self.test_set


def load_raw(path, name):
    return (
        np.load(os.path.join(path, 'AIRBUS_{}_features.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, 'AIRBUS_{}_transcripts.npy'.format(name)), encoding='bytes')
    )

def to_tensor(numpy_array, datatype):
    # Numpy array -> Tensor
    if datatype == 'int':
        return torch.from_numpy(numpy_array).int()
    elif datatype == 'long':
        return torch.from_numpy(numpy_array).long()
    else:
        return torch.from_numpy(numpy_array).float()


def to_variable(tensor, cpu=False):
    # Tensor -> Variable (on GPU if possible)
    #print(type(tensor))
    if torch.cuda.is_available() and not cpu:
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def str_to_label(strs, STR_DICT, inorout):
    label = []
    for x in strs:
        sentense = []
        if inorout == 'in':
            x = '&' + x
        else:
            x = x + '*'
        for char in x:
            sentense.append(STR_DICT[char])
        label.append(np.array(sentense))  

    label = np.array(label)    
    return label


def multipleof8len(x):
    length = len(x)
    return 8 - (length % 8) + length


class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, STR_DICT, test=False):
        sequences, strs = dataset
        self.sequences = [torch.from_numpy(x) for x in sequences]
        self.upper_seq_lengths = list(map(len, self.sequences))
        self.seq_lengths = list(map(len, self.sequences))
        
        if test:
            self.inputs = [torch.zeros((1,)) for x in sequences]
            self.label_lengths = [1 for x in sequences]
            self.outputs = [torch.zeros((1,)) for x in sequences]
        else:    
            ins = str_to_label(strs, STR_DICT, 'in')
            outs = str_to_label(strs, STR_DICT, 'out')
            self.inputs = [torch.from_numpy(x) for x in ins]
            self.label_lengths = list(map(len, self.inputs))
            self.outputs = [torch.from_numpy(x) for x in outs]

    def __getitem__(self, index):
        return self.sequences[index], self.seq_lengths[index], self.upper_seq_lengths[index], self.inputs[index], self.outputs[index], self.label_lengths[index]

    def __len__(self):
        return len(self.sequences)


def my_collate_fn(batch):
    '''
    sort sequences and seq_lengths in the batch according to sequence length
    sort inputs, outputs and label_lengths in the batch according to character level length
    give character level length sorting index to resort the sequence after unpacked in decoder

    B for batchsize, S for sequence length (frames), D for dimension (of frequency), L for character level length

    sorted according to sequence length:
    seq_sorted [B, S, D]
    length_sorted [B]

    sorted according to character level length (label_length):
    inputs_sorted [B, L]
    outputs_concat [sum of valid label length]
    label_length_sorted [B]
    lbl_perm_idx [B]
    '''
    batch_size = len(batch)

    max_len = 0
    max_output_len = 0
    sum_output_len = 0
    for sequence, seq_length, upper_seq_length, input, output, label_length in batch:
        max_len = max(max_len, upper_seq_length)
        max_output_len = max(max_output_len, label_length)
        sum_output_len += label_length

    sequences = None
    seq_lengths = None
    inputs = None
    outputs = None
    outputs_concat = None
    label_lengths = None
    unshuffle_idx = None
    if _use_shared_memory:
        sequences = torch.FloatStorage._new_shared(batch[0][0], batch_size*max_len*40).new(batch_size, max_len, 40).zero_().float()
        seq_lengths = torch.FloatStorage._new_shared(batch[0][0], batch_size).new(batch_size,).zero_().int()
        inputs = torch.FloatStorage._new_shared(batch[0][0], batch_size*max_output_len).new(batch_size, max_output_len).zero_().long()
        outputs = torch.FloatStorage._new_shared(batch[0][0], batch_size*max_output_len).new(batch_size, max_output_len).zero_().long()
        outputs_concat = torch.FloatStorage._new_shared(batch[0][0], sum_output_len).new(sum_output_len, ).zero_().long()
        label_lengths = torch.FloatStorage._new_shared(batch[0][0], batch_size).new(batch_size,).zero_().int()
        unshuffle_idx = torch.FloatStorage._new_shared(batch[0][0], batch_size).new(batch_size,).zero_().long()
    else:
        sequences = batch[0][0].new(batch_size, max_len, 40).zero_().float()
        seq_lengths = batch[0][0].new(batch_size,).zero_().int()
        inputs = batch[0][0].new(batch_size, max_output_len).zero_().long()
        outputs = batch[0][0].new(batch_size, max_output_len).zero_().long()
        outputs_concat = batch[0][0].new(sum_output_len, ).zero_().long()
        label_lengths = batch[0][0].new(batch_size,).zero_().int()
        unshuffle_idx = batch[0][0].new(batch_size,).zero_().long()

    i = 0
    for sequence, seq_length, upper_seq_length, input, output, label_length in batch:
        sequences[i, :seq_length, :] = sequence

        inputs[i, :label_length] = input
        outputs[i, :label_length] = output

        seq_lengths[i] = upper_seq_length # upper_seq_length
        label_lengths[i] = label_length
        i += 1
        
    return sequences, seq_lengths, inputs, outputs, label_lengths


def wsj_initializer(m):
    """
    reuse my code from part1
    Simple initializer
    """
    if hasattr(m, 'weight'):
        torch.nn.init.uniform(m.weight.data, -0.1, 0.1)
    if hasattr(m, 'weight_ih_l0'):
        torch.nn.init.uniform(m.weight_ih_l0.data, -0.1, 0.1)
    if hasattr(m, 'weight_hh_l0'):
        torch.nn.init.uniform(m.weight_hh_l0.data, -0.1, 0.1)


def generate_mask(p, inp=None):
    '''
    p is the drop rate, aka the probability of being 0
    modified from https://github.com/salesforce/awd-lstm-lm/blob/master/
    using .new() to initialize from the same device, much more efficient
    '''
    return (Variable(
        inp.data.new(1, inp.size(1)).bernoulli_(1. - p)) / (1. - p)).expand_as(inp)


class weightdrop_LSTM(nn.LSTM):
    '''modified from code on piazza'''

    def __init__(self, input_size, hidden_size, dropout_weight, dropout_bet):
        super(weightdrop_LSTM, self).__init__(input_size=input_size,
                                              hidden_size=hidden_size,
                                              bidirectional=True)
        self.old_weight_hh_l0 = self.weight_hh_l0
        self.weight_hh_l0 = None
        del self._parameters['weight_hh_l0']
        self.dropout_layer = nn.Dropout(dropout_weight)
        self.dropout_bet = dropout_bet

    def flatten_parameters(self):
        # overwrite, prevent pytorch from putting all weight into a large chunk
        self._data_ptrs = []

    def forward(self, inp, hx=None):
        self.weight_hh_l0 = self.dropout_layer(self.old_weight_hh_l0)
        raw_output = None

        if self.training and self.dropout_bet != 0:
            input, batch_size = inp
            between_layer_mask = generate_mask(p=self.dropout_bet, inp=input)
            dropedinput = input * between_layer_mask
            inp = PackedSequence(dropedinput, batch_size)

        return super(weightdrop_LSTM, self).forward(inp, hx=hx)


def concate_sequence(sequence, length):
    '''pyramid BiLSTM, merge consecutive time step together'''
    shape = sequence.size() # (S, B, D)
        
    # efficient using indexing, don't need iteration
    input_range = sequence.data.new(shape[0]//2).zero_().long()
    torch.arange(1, int(shape[0]), 2, out=input_range)
    input_concate = torch.cat((sequence[input_range-1], sequence[input_range]), 2)
    
    length = np.array(length) // 2

    return input_concate, length


def seq_to_nonseq(sequences, seq_lengths):
    sequence_size = sequences.size() # (S, B, D)
    nonseq = torch.cat([sequences[:length, i, :] for i, length in enumerate(seq_lengths)], 0)
    # output size of (combined_length, D)
    return nonseq


def nonseq_to_seq(nonseq, seq_lengths):
    dim = nonseq.size(1)
    length = int(torch.max(seq_lengths))
    batch_size = seq_lengths.shape[0]

    sequences = nonseq.data.new(batch_size, length, dim).zero_()
    sequences = to_variable(sequences) # no need to set requires_grad=True
    
    lbl_ind = 0
    for i, seq_length in enumerate(seq_lengths):
        sequences[i, :seq_length, :] = nonseq[lbl_ind:lbl_ind+seq_length]
        lbl_ind += seq_length
        
    # output sequence (B, S, D)
    return sequences


class CrossEntropyLossMask(nn.CrossEntropyLoss):
    '''
    calculate the gradient of valid label only
    input (B, S, C), target (total length of valid labels, )
    '''
    def __init__(self, *args, **kwargs):
        super(CrossEntropyLossMask, self).__init__(*args, reduce=False, **kwargs)
        #super(CrossEntropyLossMask, self).__init__(weight=None, size_average=True, ignore_index=-100, reduce=True) 

    def forward(self, logits, target, label_lengths):
        logits_size = logits.size() # (B, S, D)

        # generate this length in both training and evaluation
        # length should be the same for target and logits
        maxlen = target.size(1) 

        mask = output_mask(maxlen, label_lengths)
        mask = torch.transpose(mask, 0, 1)
        mask = to_variable(mask).float() # (B, S)

        logits = logits * mask.unsqueeze(2)
        losses = super(CrossEntropyLossMask, self).forward(logits.view(-1, logits_size[2]), target.view(-1))

        # two steps into one, but we might need the losses of each sentence
        loss = torch.sum(mask.view(-1) * losses) / logits_size[0]

        #masked_loss = mask.view(-1) * losses
        #reshape_losses = masked_loss.view(logits_size[0], logits_size[1]).sum(1)

        # take the mean over mini-batch
        #loss = reshape_losses.mean()

        return loss



class AdvancedLSTM(nn.LSTM):
    # Class for learning initial hidden states when using LSTMs
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(-1, n, -1).contiguous(),
            self.c0.expand(-1, n, -1).contiguous()
        )

    def forward(self, input, hx=None):
        if hx is None:
            n = input.batch_sizes[0]
            hx = self.initial_state(n)
        return super(AdvancedLSTM, self).forward(input, hx=hx)


class AdvancedLSTMCell(nn.LSTMCell):
    # Extend LSTMCell to learn initial state
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


class pLSTM(AdvancedLSTM):
    # Pyramidal LSTM
    def __init__(self, *args, **kwargs):
        super(pLSTM, self).__init__(*args, **kwargs)
        self.shuffle = SequenceShuffle()

    def forward(self, input, hx=None):
        return super(pLSTM, self).forward(self.shuffle(input), hx=hx)


class SequenceShuffle(nn.Module):
    # Performs pooling for pBLSTM
    def forward(self, seq):
        assert isinstance(seq, PackedSequence)
        h, seq_lengths = pad_packed_sequence(seq)
        
        # pyramid BiLSTM, merge consecutive time step together
        # input size should be (S, B, D)
        h, seq_lengths = concate_sequence(h, seq_lengths)
        h = pack_padded_sequence(h, seq_lengths)
        return h


class MLP(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(MLP, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, args.linear_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(args.linear_dim, output_dim))
        #self.layers.append(nn.ReLU())

    def forward(self, h):
        for l in self.layers:
            h = l(h)
        return h


class CNN(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels=input_dim, out_channels=output_dim, padding=1, kernel_size=3, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(output_dim))
        self.layers.append(nn.Dropout(0.1))
        self.layers.append(nn.Conv1d(in_channels=output_dim, out_channels=output_dim, padding=1, kernel_size=3, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(output_dim))
        self.layers.append(nn.Dropout(0.1))

    def forward(self, h):
        for l in self.layers:
            h = l(h)
        return h


class EncoderModel(nn.Module):
    # Encodes utterances to produce keys and values
    def __init__(self, args, input_dim):
        super(EncoderModel, self).__init__()
        self.cnn = CNN(args, input_dim, args.encoder_dim)
        self.rnns = nn.ModuleList()
        #self.rnns.append(AdvancedLSTM(input_dim, args.encoder_dim, bidirectional=True))
        self.rnns.append(AdvancedLSTM(args.encoder_dim, args.encoder_dim, bidirectional=True))
        self.rnns.append(AdvancedLSTM(args.encoder_dim * 2, args.encoder_dim, bidirectional=True))
        #self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        #self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        self.rnns.append(pLSTM(args.encoder_dim * 4, args.encoder_dim, bidirectional=True))
        self.key_projection = MLP(args, args.encoder_dim * 2, args.key_dim)
        #self.value_projection = MLP(args, args.encoder_dim * 2, args.value_dim)

    def forward(self, utterances, utterance_lengths):
        h = utterances # (B, S, D)

        # CNN 
        h = torch.transpose(h, 1, 2) # input (B, D, S)
        h = self.cnn(h)
        h = torch.transpose(h, 1, 2) # (B, S, D)

        # listen
        h = torch.transpose(h, 0, 1) # (S, B, D)

        sorted_lengths, order = torch.sort(utterance_lengths, 0, descending=True)
        _, backorder = torch.sort(order, 0)
        h = h[:, order, :]
        h = pack_padded_sequence(h, sorted_lengths.data.cpu().numpy())

        for rnn in self.rnns:
            h, _ = rnn(h)

        # Unpack and unsort the sequences
        h, output_lengths = pad_packed_sequence(h)
        h = h[:, backorder, :]
        output_lengths = output_lengths[backorder]

        values = torch.transpose(h, 0, 1) # (B, S, D)

        # input of seq_to_nonseq should be (S, B, D)
        non_seq_hid = seq_to_nonseq(h, output_lengths) # (sum of length of valid labels, D)
        h1 = self.key_projection(non_seq_hid) 
        keys = nonseq_to_seq(h1, output_lengths) # (B, u, D)

        #h2 = self.value_projection(non_seq_hid)
        #values = nonseq_to_seq(h2, output_lengths) # (B, u, D)
        
        return keys, values, output_lengths


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_argmax(logits, dim):
    # Draw from a multinomial distribution efficiently
    return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]


class DecoderModel(nn.Module):
    # Speller/Decoder
    def __init__(self, args, vocab_size, init_bias):
        super(DecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, args.decoder_dim)
        self.input_rnns = nn.ModuleList()
        self.value_dim = args.encoder_dim * 2
        #value_dim = args.value_dim
        self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim + self.value_dim, args.decoder_dim))
        self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim, args.decoder_dim))
        rnn_out_dim = args.decoder_dim * 2
        self.input_rnns.append(AdvancedLSTMCell(args.decoder_dim, rnn_out_dim))
        self.query_projection = MLP(args, rnn_out_dim, args.key_dim)
        #self.query_enhance_input = Query_with_enhance_input(args, args.decoder_dim+self.value_dim, args.enhance_dim, rnn_out_dim, args.key_dim)
        # weight-tying with decoder_dim
        self.char_projection = nn.Sequential(
            nn.Linear(rnn_out_dim+self.value_dim, args.decoder_dim),
            nn.LeakyReLU(), 
            nn.Linear(args.decoder_dim, vocab_size)
        )
        self.net_out_prob = args.net_out_prob
        # TODO: see if it really helps
        self.char_projection[-1].weight = self.embedding.weight  # weight tying
        if init_bias is not None:
            self.char_projection[-1].bias.data = init_bias
        
    def forward(self, input_y, keys, values, max_label_length, feature_lengths, eval=False):

        batch = keys.size(0) # B
        max_feature_length = torch.max(feature_lengths)
        if max_label_length is not None:
            if int(max_feature_length) < int(max_label_length):
                print("features length: {} smaller that label length: {}".format(int(max_feature_length), int(max_label_length)))
        length = max_feature_length # TODO: this might not be enough
        if max_label_length is not None:
            # label length
            length = max_label_length
        
        # Initial states
        input_states = [rnn.initial_state(batch) for rnn in self.input_rnns]

        # Initial context
        h0 = input_states[-1][0] # (B, D)
        query = self.query_projection(h0)

        # compute mask
        mask = output_mask(max_feature_length, feature_lengths)
        mask = to_variable(torch.transpose(mask, 0, 1).unsqueeze(2)).float()
        context = compute_context(keys, query, values, mask)
        #context = to_variable(keys.data.new(batch, 1, self.value_dim).zero_())
        
        pred_char = to_variable(context.data.new(batch, 1).zero_().long())

        logits = [] 
        generateds = []
        greedys = []
        for i in range(length):
            # input states: pre_context_states (si-1), input_y (yi-1), context (ci-1)
            # compute phi
            input_char = pred_char
            if input_y is not None:
                input_char = self.getInputChar(pred_char, input_y[:,i:i+1])

            h = self.embedding(input_char) # (B, 1, D2)
            # context (B, 1, D)
            rnn_input = torch.cat((h, context), 2)  
            rnn_input = torch.squeeze(rnn_input) # (B, D)
            ht = rnn_input

            # pre_context_out si (B, D)
            new_input_states = []
            for rnn, state in zip(self.input_rnns, input_states):
                ht, newstate = rnn(ht, state)
                new_input_states.append((ht, newstate))
            input_states = new_input_states

            #pre_context_out = input_states[-1][0]
            #pre_context_out = torch.unsqueeze(pre_context_out, 1)
            #ht = torch.unsqueeze(ht, 1)

            # phi in mlp output on si
            #rnn_input = torch.unsqueeze(rnn_input, 1)
            #query = self.query_enhance_input(pre_context_out, rnn_input)
            query = self.query_projection(ht).unsqueeze(2)

            context = compute_context(keys, query, values, mask)

            # spell
            # (B, 1, D1 + D2) -> (B, 1, output_dim)
            #state_context = torch.cat((pre_context_out, context), 2)
            ht = ht.unsqueeze(1)
            state_context = torch.cat((ht, context), 2)
            logit = self.char_projection(state_context)
            logits.append(logit)

            #pred_char = torch.max(logit, dim=2)[1]
            greedy = torch.max(logit, dim=2)[1]
            greedys.append(greedy)

            # TODO: random 
            random_sample = gumbel_argmax(logit, 2)  # (N,1)
            generateds.append(random_sample)

            # TODO
            #pred_char = greedy
            if eval:
                pred_char = greedy
            else:
                pred_char = random_sample

        logits = torch.cat(logits, dim=1)
        greedys = torch.cat(greedys, dim=1)
        generateds = torch.cat(generateds, dim=1)
        return logits, generateds, greedys
    
    
    def getInputChar(self, pred_char, input_y):
        '''
        select input from prediction of last time step or from groundtruth
        the probability of choosing groundtruth is given by self.net_out_prob
        input (B, 1)
        '''   
        # efficient, less variable
        p = torch.ones((pred_char.size()), out=pred_char.data.new(pred_char.size())).float() * (1 - self.net_out_prob)
        cond1 = to_variable(torch.bernoulli(p, out=p.new(p.size())).long())
        cond2 = 1 - cond1

        input_char = cond1 * input_y + cond2 * pred_char

        return input_char

    def calculate_prob(self, states, t, pred_char, keys, values, mask):
        # return output logits, newstates
        (input_states, query, context) = states

        input_char = pred_char.long()

        h = self.embedding(input_char) # (B, 1, D2)

        # context (B, 1, D)
        rnn_input = torch.cat((h, context), 2)  
        rnn_input = torch.squeeze(rnn_input, 1) # (B, D)
        ht = rnn_input

        new_input_states = []
        for rnn, state in zip(self.input_rnns, input_states):
            ht, newstate = rnn(ht, state)
            new_input_states.append((ht, newstate))

        new_query = self.query_projection(ht).unsqueeze(2)

        new_context = compute_context(keys, new_query, values, mask)

        ht = ht.unsqueeze(1)
        state_context = torch.cat((ht, new_context), 2)
        logit = self.char_projection(state_context)

        newstates = (new_input_states, new_query, new_context)

        return logit, newstates


    def get_initial_states(self, keys, values, utterance_length):
        # B = 1
        # Initial states
        input_states = [rnn.initial_state(1) for rnn in self.input_rnns]

        # Initial context
        h0 = input_states[-1][0] # (S, B, D)
        query = self.query_projection(h0)
        
        # compute mask
        mask = output_mask(int(utterance_length), utterance_length)
        mask = to_variable(torch.transpose(mask, 0, 1).unsqueeze(2)).float()
        context = compute_context(keys, query, values, mask)

        return (input_states, query, context), mask
    

def output_mask(maxlen, lengths):
    """
    Create a mask on-the-fly
    :param maxlen: length of mask
    :param lengths: length of each sequence
    :return: mask shaped (maxlen, len(lengths))
    """
    lens = lengths.unsqueeze(0)
    ran = torch.arange(0, maxlen, 1, out=lengths.new()).unsqueeze(1)
    mask = ran < lens
    return mask
    
# psi (B, u, D3)  phi (B, D3, 1) -> (B, u, 1)
def compute_context(keys, query, values, mask):
    if query.dim() < 3:
        query = query.unsqueeze(2)

    attention_over_sequence = torch.bmm(keys, query)
    attention_size = attention_over_sequence.size()

    # masked softmax, calculate attention over only the non-padding regions
    # mask before and after softmax
    masked_attention_over_sequence = attention_over_sequence * mask
    pre_alpha = F.softmax(masked_attention_over_sequence, 1)
    masked_pre_alpha = pre_alpha * mask

    masked_pre_alpha = masked_pre_alpha.transpose(1, 2) # (B, 1, u)

    # will give nan
    #mask_sum = torch.sum(masked_pre_alpha, 2, keepdim=True)
    #alpha = masked_pre_alpha / mask_sum

    # normalize
    alpha = F.normalize(masked_pre_alpha, p=1, dim=2)

    # (B, 1, u) (B, u, D) -> (B, 1, D)
    context = torch.bmm(alpha, values)
    return context

    '''
    if query.dim() < 3:
        query = query.unsqueeze(2)
    attention_over_sequence = torch.bmm(keys, query)
    attention_size = attention_over_sequence.size()
    alpha = F.softmax(attention_over_sequence, 1).view(attention_size[0], 1, attention_size[1]) # (B, 1, u)

    # (B, 1, u) (B, u, D) -> (B, 1, D)\
    context = torch.bmm(alpha, values)
    return context
    '''


class Query_with_enhance_input(nn.Module):
    def __init__(self, args, input_dim, enhance_dim, decoder_dim, key_dim):
        super(Query_with_enhance_input, self).__init__()
        self.enhance_layer = nn.Linear(input_dim, enhance_dim)
        self.enhance_layer_relu = nn.LeakyReLU()
        
        self.linear_layers = nn.ModuleList([
            nn.Linear(decoder_dim, args.linear_dim),
            nn.LeakyReLU()
        ])
        self.merge_layer = nn.Linear(args.linear_dim + enhance_dim, key_dim)
        #self.merge_layer_relu = nn.ReLU()
            
    def forward(self, pre_context_out, rnn_input):
        enhance = self.enhance_layer_relu(self.enhance_layer(rnn_input))

        h = pre_context_out
        for l in self.linear_layers:
            h = l(h)

        h = torch.cat((h, enhance), 2)
        h = self.merge_layer(h)
        
        h_size = h.size() # (B, 1, D3)
        h = h.view(h_size[0], h_size[2], 1)
        return h


class Seq2SeqModel(nn.Module):
    # Tie encoder and decoder together
    def __init__(self, args, input_dim, vocab_size, init_bias):
        super(Seq2SeqModel, self).__init__()
        self.encoder = EncoderModel(args, input_dim)
        self.decoder = DecoderModel(args, vocab_size, init_bias)

    def forward(self, utterances, utterance_lengths, chars, max_label_length, eval=False):
        keys, values, seq_lengths = self.encoder(utterances, utterance_lengths)
        logits, generateds, greedys = self.decoder(chars, keys, values, max_label_length, seq_lengths, eval=eval)
        return logits, generateds, greedys


def params_str(args, trloss, vlloss, epoch, edit_distance):
    params = vars(args)
    params_str_ = sorted(params.items(), key=lambda x: x[0])
    return params['output_dir'] + '/' + \
            "trloss_%.2f_vlloss_%.2f_epoch_%d_L_%.2f_" % (trloss, vlloss, epoch, edit_distance) + \
            "_".join(["{}_{}".format(k, v) for k, v in params_str_ if k not in ['output_dir', 'epochs']])


def save_args(args):
   # Save argparse arguments to a file for reference
    with open(os.path.join(args.output_dir, 'args.txt'), 'a') as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


# convert dictionary to namespace
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_parameters(param_grid, n_iter=100):
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter))
    #param_list = list(ParameterGrid(param_grid))
    for item in param_list:
        yield item

'''
def convert_to_string(tokens, unshuffle_idx, vocab):
    chars = []
    unshuffled_tokens = unshuffle(tokens, unshuffle_idx)
    strings = []
    #tokens = tokens.data.cpu().numpy()
    for token in unshuffled_tokens:
        print(token)
        for x in token:
            if x == 1:
                break
            chars.append(vocab[x])
        strings.append(''.join(chars))
        chars.clear()
    return strings
'''
def convert_to_string(tokens, vocab):
    chars = []
    tokens = tokens.data.cpu().numpy()
    strings = []
    for token in tokens:
        for x in token:
            if x == 1:
                break
            chars.append(vocab[x])
        strings.append(''.join(chars))
        chars.clear()
    return strings


def unshuffle(x, unshuffle_idx):
    x = x.data.cpu().numpy()
    res = np.zeros(x.shape).astype(int)
    for i, idx in enumerate(unshuffle_idx):
        res[idx,:] = x[i]
    return res


class model_data_optim():
    def __init__(self, dataloader, valid_dataloader, test_data_loader, args, init_bias, STRINGS, input_dim, vocab_size):
        self.net = Seq2SeqModel(args, input_dim, vocab_size, init_bias)
        print(self.net)
        self.data_loader = dataloader
        self.valid_data_loader = valid_dataloader
        self.test_data_loader = test_data_loader
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        self.criterion = CrossEntropyLossMask()
        self.best_validation_loss = float('inf')
        self.model_param_str = args.output_dir
        self.best_model_param_str = self.model_param_str
        self.args = args
        self.STRINGS = STRINGS
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_args(args)
        print(args)
        if torch.cuda.is_available():
            self.net.cuda()
            self.criterion.cuda()


    def train_model(self, num_epochs=None):
        start_time = time.time()
        num_epochs = self.args.epochs if not num_epochs else num_epochs
        for epoch in range(num_epochs):
            self.net.train()
            
            self.adjust_lr(epoch)
            losses = []
            
            for sequence, seq_length, input, output, label_length in self.data_loader:
                
                if torch.cuda.is_available():
                    sequence = sequence.cuda()
                    input = input.cuda()
                    output = output.cuda()
                    label_length = label_length.cuda()

                self.optimizer.zero_grad()  # Reset the gradients

                max_label_length = torch.max(label_length)
                logits, generateds, greedys = self.net(to_variable(sequence), seq_length, to_variable(input), max_label_length) # weird

                loss = self.criterion(logits, to_variable(output), label_length) # weird 2
                loss.backward(retain_graph=True) 
                losses.append(loss.data.cpu().numpy())
                print("epoch: {}, loss: {}".format(epoch, loss.data.cpu().item()))
                torch.nn.utils.clip_grad_norm(self.net.parameters(), 0.25)  # gradient clip
                self.optimizer.step()
            loss_print = np.asscalar(np.mean(losses))
            print("epoch: {}, loss: {}".format(epoch, loss_print))

            # evaluate at the end of each epoch
            evaluation_loss, edit_distance = self.eval_model()
            self.model_param_str = params_str(self.args, loss_print, evaluation_loss, epoch, edit_distance)
            with open(os.path.join(self.args.output_dir, 'args.txt'), 'a') as f:
                f.write("save as:\n{}".format(self.model_param_str + '.pkl'))
            print("save as:\n{}".format(self.model_param_str + '.pkl'))
            torch.save(self.net.state_dict(), self.model_param_str + '.pkl')
            if evaluation_loss < self.best_validation_loss:
                self.best_validation_loss = evaluation_loss
                self.best_model_param_str = self.model_param_str
            
            old_time = start_time
            start_time = time.time()
            print("--------epoch time: {}--------".format(start_time - old_time))
            with open(os.path.join(self.args.output_dir, 'args.txt'), 'a') as f:
                f.write("epoch: {}, training loss: {:.4f}, validation loss: {:.4f}\n---------epoch time: {}---------\n".format(epoch, loss_print, evaluation_loss, start_time - old_time))

            self.write_predictions()
        
        print('the best model param is {}'.format(self.best_model_param_str))
        return self.best_model_param_str, self.best_validation_loss


    def adjust_lr(self, epoch):
        lr = self.args.init_lr * (0.1 ** (epoch // 7))
        lr = max(1e-5, lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if epoch % 3 == 2:
            self.args.net_out_prob += 0.1
            self.args.net_out_prob = min(0.5, self.args.net_out_prob)

        print("----------adjusting learning rate: {}, net_out_prob: {}----------".format(lr, self.args.net_out_prob))


    def eval_model(self):
        self.net.eval()
        losses = []
        mean_edit_distance1 = 0.
        mean_edit_distance2 = 0.
        total_cnt = 0
        for sequence, seq_length, input, output, label_length in self.valid_data_loader:
            total_cnt += 1

            if torch.cuda.is_available():
                sequence = sequence.cuda()
                input = input.cuda()
                output = output.cuda()
                label_length = label_length.cuda()

            max_label_length = torch.max(label_length)
            # TODO: to_variable(input)
            logits, generateds, greedys = self.net(to_variable(sequence), seq_length, None, max_label_length, eval=True) # weird

            # print out 
            greedy_str = convert_to_string(greedys, self.STRINGS)
            print("greedy", greedy_str)

            generated_str = convert_to_string(generateds, self.STRINGS)
            print("generated", generated_str)

            label_str = convert_to_string(output, self.STRINGS)
            print("outputs", label_str)

            ls1 = 0.
            ls2 = 0.
            for pred1, pred2, true in zip(generated_str, greedy_str, label_str):
                ls1 += L.distance(pred1, true)
                ls2 += L.distance(pred2, true)
                
            ls1 /= len(label_str)
            ls2 /= len(label_str)
            mean_edit_distance1 += ls1
            mean_edit_distance2 += ls2

            # prediction and output could both be longer than each other
            loss = self.criterion(logits, to_variable(output), label_length) # weird 2
            losses.append(loss.data.cpu().numpy())

        loss_print = np.asscalar(np.mean(losses))
        mean_edit_distance1 /= total_cnt
        mean_edit_distance2 /= total_cnt
        print("edit distance1: {}, edit distance2: {}, validation loss: {}".format(mean_edit_distance1, mean_edit_distance2, loss_print))
        return loss_print, mean_edit_distance2  


    def load_model(self, model_dir):
        self.net.load_state_dict(torch.load(model_dir + '.pkl'))
        self.model_param_str = model_dir


    def write_predictions(self):
        self.net.eval()

        with open(self.model_param_str + '.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Id', 'Predicted'])
            id = 0
            for sequence, seq_length, _, _, _ in self.test_data_loader:

                if torch.cuda.is_available():
                    sequence = sequence.cuda()
                logits, generateds, greedys = self.net(to_variable(sequence), seq_length, None, None, eval=True)

                output_strs = convert_to_string(generateds, self.STRINGS)
                #print("generateds", output_strs)

                output_str = convert_to_string(greedys, self.STRINGS)
                #print("greedy", output_str)

                for output_str in output_strs:
                    w.writerow([id, output_str])
                    id += 1


def make_new_beam():
    fn = lambda : -float("inf")
    return collections.defaultdict(fn)


class BeamSearchVtlp(nn.Module):
    # Tie encoder and decoder together
    def __init__(self, args, input_dim, vocab_size, STRINGS, beam_size=3, eos_index=1):
        super(BeamSearchVtlp, self).__init__()
        self.encoder = EncoderModel(args, input_dim)
        self.decoder = DecoderModel(args, vocab_size, None)
        self.STRINGS = STRINGS
        self.beam_size = beam_size
        self.eos_index = eos_index
        self.vocab_size = vocab_size
        if torch.cuda.is_available():
            self.cuda()


    def forward(self, utterances, utterance_lengths):
        utterance1, utterance2, utterance3 = utterances
        utterance_length1, utterance_length2, utterance_length3 = utterance_lengths
        key1, value1, seq_lengths = self.encoder(utterance1, utterance_length1)
        key2, value2, seq_lengths = self.encoder(utterance2, utterance_length2)
        key3, value3, seq_lengths = self.encoder(utterance3, utterance_length3)
        keys = (key1, key2, key3)
        values = (value1, value2, value3)
        labels, score, beams = self.beamsearch_through_sequence(seq_lengths, 
                                                                keys, values)
        return labels, score, beams


    def load_model(self, model_dir):
        self.load_state_dict(torch.load(model_dir + '.pkl'))


    def beamsearch_through_sequence(self, utterance_length, keys, values):
        (key1, key2, key3) = keys
        (value1, value2, value3) = values
        states1, mask = self.decoder.get_initial_states(key1, value1, utterance_length)
        states2, mask = self.decoder.get_initial_states(key2, value2, utterance_length)
        states3, mask = self.decoder.get_initial_states(key3, value3, utterance_length)
        beam = [("", (states1, states2, states3, 0.0, 0))]
        complete_sequences = []
        
        T = int(utterance_length)

        for t in range(T):
            beam, complete_sequences = self.beamsearch_one_step(beam, 
                                                            complete_sequences, t, 
                                                            keys, values, mask)

        # finish up none ending sequences
        beam = [(x, z / T) for x, (_, _, _, z, _) in beam]
        beam = beam + complete_sequences

        beam = sorted(beam,
                    key=lambda x : x[1],
                    reverse=True)
        best = beam[0]
        print(best)
        return best[0], best[1], beam    


    def beamsearch_one_step(self, beam, complete_sequences, t, keys, values, mask):
        next_beam = make_new_beam()
        key1, key2, key3 = keys
        value1, value2, value3 = values

        for prefix, (states1, states2, states3, p_b, prev_output) in beam: # Loop over beam
            pred_char = torch.IntTensor([[prev_output]]) # use previous output as input
            if torch.cuda.is_available():
                pred_char = pred_char.cuda()

            output1, newstates1 = self.decoder.calculate_prob(states1, t, pred_char, 
                                                            key1, value1, mask)
            output2, newstates2 = self.decoder.calculate_prob(states2, t, pred_char, 
                                                            key2, value2, mask)
            output3, newstates3 = self.decoder.calculate_prob(states3, t, pred_char, 
                                                            key3, value3, mask)
            output = (output1 + output2 + output3) / 3
            logprobs = F.log_softmax(output, 2).data.cpu().numpy()

            for s_index in range(self.vocab_size): # loop over vocabulary

                p = logprobs[0, 0, s_index]
                n_p_b = p_b + p # included the prob of the last

                if s_index != self.eos_index:
                    n_prefix = prefix + self.STRINGS[s_index]
                    # *NB* this would be a good place to include an LM score.
                    next_beam[n_prefix] = (newstates1, newstates2, newstates3, n_p_b, s_index)
                else:
                    # normalize by length
                    complete_sequences.append((prefix, n_p_b / (t+1))) 

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : x[1][-2],
                reverse=True)
        beam = beam[:self.beam_size]

        return beam, complete_sequences


class BeamSearcher(nn.Module):
    # Tie encoder and decoder together
    def __init__(self, args, input_dim, vocab_size, STRINGS, beam_size=3, eos_index=1):
        super(BeamSearcher, self).__init__()
        self.encoder = EncoderModel(args, input_dim)
        self.decoder = DecoderModel(args, vocab_size, None)
        self.STRINGS = STRINGS
        self.beam_size = beam_size
        self.eos_index = eos_index
        self.vocab_size = vocab_size
        if torch.cuda.is_available():
            self.cuda()


    def forward(self, utterance, utterance_length):
        keys, values, seq_lengths = self.encoder(utterance, utterance_length)
        labels, score, beams = self.beamsearch_through_sequence(seq_lengths, 
                                                                keys, values)
        return labels, score, beams


    def load_model(self, model_dir):
        self.load_state_dict(torch.load(model_dir + '.pkl'))


    def beamsearch_through_sequence(self, utterance_length, keys, values):
        states, mask = self.decoder.get_initial_states(keys, values, utterance_length)
        beam = [("", (states, 0.0, 0))]
        complete_sequences = []
        
        T = int(utterance_length)

        for t in range(T):
            #print("----------decoding time step : {}--------".format(t))
            beam, complete_sequences = self.beamsearch_one_step(beam, 
                                                            complete_sequences, t, 
                                                            keys, values, mask)

        # finish up none ending sequences
        beam = [(x, z / T) for x, (y, z, _) in beam]
        beam = beam + complete_sequences

        beam = sorted(beam,
                    key=lambda x : x[1],
                    reverse=True)
        best = beam[0]
        return best[0], best[1], beam    


    def beamsearch_one_step(self, beam, complete_sequences, t, keys, values, mask):
        next_beam = make_new_beam()

        for prefix, (states, p_b, prev_output) in beam: # Loop over beam
            pred_char = torch.IntTensor([[prev_output]]) # use previous output as input
            if torch.cuda.is_available():
                pred_char = pred_char.cuda()

            output, newstates = self.decoder.calculate_prob(states, t, pred_char, 
                                                            keys, values, mask)
            logprobs = F.log_softmax(output, 2).data.cpu().numpy()

            decoded_char_idx = np.argmax(F.softmax(output, 2).data.cpu().numpy(), 2)[0][0]
            #print("greedy choice: {}, logprob: {}".format(self.STRINGS[decoded_char_idx], logprobs[0,0,decoded_char_idx]))

            for s_index in range(self.vocab_size): # loop over vocabulary
                #print("this guy: {}, logprob: {}".format(self.STRINGS[s_index], logprobs[0,0,s_index]))
                
                p = logprobs[0, 0, s_index]
                n_p_b = p_b + p # included the prob of the last

                if s_index != self.eos_index:
                    n_prefix = prefix + self.STRINGS[s_index]
                    # *NB* this would be a good place to include an LM score.
                    next_beam[n_prefix] = (newstates, n_p_b, s_index)
                else:
                    # normalize by length
                    complete_sequences.append((prefix, n_p_b / (t+1))) 

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : x[1][1],
                reverse=True)
        beam = beam[:self.beam_size]

        return beam, complete_sequences


def grid_search():
    param_grid = {'encoder_dim': [128, 256, 512], 'batch_size': [32], 'epochs':[5], 
            'cuda': [True], 'net_out_prob': [0.1], 'weight_decay': [1e-5], 
            'key_dim':[128, 256, 512], 'value_dim':[128, 256, 512], 
            'enhance_dim':[256, 512, 1024], 'decoder_dim':[512, 768, 1024], 
            'init_lr':[1e-3], 'output_dir':['./input-enhance-output/']}
    cuda = param_grid['cuda'][0]
    batch_size = param_grid['batch_size'][0]


    wsj = WSJ()
    train_data, train_label = wsj.train 
    valid_data, valid_label = wsj.dev 

    STRINGS = [x for x in "&*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz #'-/@_"]
    INPUT_DIM = 40
    vocab_size = len(STRINGS)
    STR_DICT = {}
    for i, x in enumerate(STRINGS):
        STR_DICT[x] = i

    kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(
        myDataset(wsj.train, STR_DICT), shuffle=True,
        batch_size=batch_size, collate_fn=my_collate_fn, **kwargs)
    valid_loader = DataLoader(
        myDataset(wsj.dev, STR_DICT), shuffle=True,
        batch_size=batch_size, collate_fn=my_collate_fn, **kwargs)

    # define and train
    init_bias = None

    # train model
    best_validation_loss = float('inf')
    best_params = None
    for i, params in enumerate(get_parameters(param_grid)):
        print("model: {}".format(i))
        args = Bunch(params)

        mdo = model_data_optim(train_loader, valid_loader, None, args, init_bias, STRINGS, INPUT_DIM, vocab_size)
        model_param_str, vlloss = mdo.train_model()
        if vlloss < best_validation_loss:
            best_validation_loss = vlloss
            best_params = params
            print('best params: {}'.format(best_params))
        
    
    print('the best model param is {}, the best validation loss is {}'.format(best_params, best_validation_loss))


def beamsearch(args, model_dir):

    wsj = WSJ()

    STRINGS = [x for x in "&*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz #'-/@_"]
    INPUT_DIM = 40
    vocab_size = len(STRINGS)
    STR_DICT = {}
    for i, x in enumerate(STRINGS):
        STR_DICT[x] = i

    kwargs = {'num_workers': 32, 'pin_memory': True} if args.cuda else {}

    test_loader1 = DataLoader(
        myDataset(wsj.test1, STR_DICT, True), shuffle=False,
        batch_size=1, collate_fn=my_collate_fn, **kwargs)
    test_loader2 = DataLoader(
        myDataset(wsj.test2, STR_DICT, True), shuffle=False,
        batch_size=1, collate_fn=my_collate_fn, **kwargs)
    test_loader3 = DataLoader(
        myDataset(wsj.test3, STR_DICT, True), shuffle=False,
        batch_size=1, collate_fn=my_collate_fn, **kwargs)

    init_bias = None

    beamsearcher = BeamSearchVtlp(args, INPUT_DIM, vocab_size, STRINGS, beam_size=3, eos_index=1)
    beamsearcher.load_model(model_dir)
    beamsearcher.eval()
    losses = []
    with open(model_dir + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Predicted'])
        id = 0
        for loader1, loader2, loader3 in zip(test_loader1, test_loader2, test_loader3):
            (sequence1, seq_length1, _, _, _) = loader1
            (sequence2, seq_length2, _, _, _) = loader2
            (sequence3, seq_length3, _, _, _) = loader3

            if torch.cuda.is_available():
                sequence1 = sequence1.cuda()
                sequence2 = sequence2.cuda()
                sequence3 = sequence3.cuda()

            sequences = (to_variable(sequence1), to_variable(sequence2), to_variable(sequence3))
            seq_lengths = (seq_length1, seq_length2, seq_length3)
            labels, score, beams = beamsearcher(sequences, seq_lengths) # weird
            w.writerow([id, labels])
            id += 1



def main(args, model_dir):

    wsj = WSJ()
    train_data, train_label = wsj.train 

    STRINGS = [x for x in "&*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz #'-/@_"]
    INPUT_DIM = 40
    vocab_size = len(STRINGS)
    STR_DICT = {}
    for i, x in enumerate(STRINGS):
        STR_DICT[x] = i

    kwargs = {'num_workers': 32, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(
        myDataset(wsj.train, STR_DICT), shuffle=True,
        batch_size=args.batch_size, collate_fn=my_collate_fn, **kwargs)
    valid_loader = DataLoader(
        myDataset(wsj.dev, STR_DICT), shuffle=True,
        batch_size=args.batch_size, collate_fn=my_collate_fn, **kwargs)
    #test_loader1 = DataLoader(
    #    myDataset(wsj.test1, STR_DICT, True), shuffle=False,
    #    batch_size=1, collate_fn=my_collate_fn, **kwargs)
    test_loader = DataLoader(
        myDataset(wsj.test2, STR_DICT, True), shuffle=False,
        batch_size=args.batch_size, collate_fn=my_collate_fn, **kwargs)
    #test_loader3 = DataLoader(
    #    myDataset(wsj.test3, STR_DICT, True), shuffle=False,
    #    batch_size=1, collate_fn=my_collate_fn, **kwargs)

    # define and train
    # hack initialization for last layer bias with log unigram probability
    if model_dir is not None:
        init_bias = None
    else:
        total_cnt = 0
        init_bias = torch.zeros((vocab_size,))
        for data in train_label:
            for word in data:
                wordind = STR_DICT[word]
                init_bias[wordind] = init_bias[wordind] + 1
                total_cnt += 1
        init_bias = init_bias / total_cnt

        smoothing = 0.1
        init_bias = (init_bias * (1. - smoothing)) + (smoothing / vocab_size)

        init_bias = torch.log(init_bias)


    mdo = model_data_optim(train_loader, valid_loader, test_loader, args, init_bias, STRINGS, INPUT_DIM, vocab_size)
    
    model_param_str = None
    if not model_dir:
        model_param_str, _ = mdo.train_model()
    else:
        # load model
        mdo.load_model(model_dir)
        if args.epochs != 0:
            mdo.eval_model()
            model_param_str, _ = mdo.train_model()

    if model_param_str: # load best model
        mdo.load_model(model_param_str)

    mdo.eval_model()
    mdo.write_predictions()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--encoder_dim', dest='encoder_dim', type=int, default=256)
    parser.add_argument('--key_dim', dest='key_dim', type=int, default=1024)
    parser.add_argument('--value_dim', dest='value_dim', type=int, default=1024)
    parser.add_argument('--enhance_dim', dest='enhance_dim', type=int, default=256)
    parser.add_argument('--decoder_dim', dest='decoder_dim', type=int, default=1024)
    parser.add_argument('--linear_dim', dest='linear_dim', type=int, default=1024)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-6)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=1e-3)
    parser.add_argument('--net_out_prob', dest='net_out_prob', type=float, default=0.1)
    parser.add_argument('--cuda', dest='cuda',
                              action='store_true',
                              help="Whether to use cuda in worker for dataloader.")
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='./vtlp')
    args = parser.parse_args(sys.argv[1:])

    #grid_search()
    #main(args, model_dir=None)
    beamsearch(args, model_dir='./vtlp/trloss_1.50_vlloss_837.96_epoch_7_L_17.18_batch_size_16_cuda_False_decoder_dim_1024_encoder_dim_256_enhance_dim_256_init_lr_0.0003_key_dim_1024_linear_dim_1024_net_out_prob_0.30000000000000004_value_dim_1024_weight_decay_1e-06')



