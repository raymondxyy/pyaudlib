"""Main script for training DNNs for speech recognition."""

import numpy as np
import os
import torch
import torch.utils.data
import argparse
import time
import csv
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import sys

from audlib.asr.util import levenshtein
from audlib.nn.rnn import MLP, ExtendedLSTMCell, ExtendedLSTM,\
    PyramidalLSTM
from audlib.nn.transform import Compose, ToDevice

from transforms import sort_by_feat_length_collate_fn
from dataset import FEATDIM, CHARMAP, WSJ_TRAIN, WSJ_VALID, WSJ_TEST,\
    VOCAB_HIST


class CNN(nn.Module):
    # TODO: Remove hard-coded parameters and generalize parameters.
    # TODO: Add docstring.
    def __init__(self, args, input_dim, output_dim):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(
            in_channels=input_dim, out_channels=output_dim, padding=1,
            kernel_size=3, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(output_dim))
        self.layers.append(nn.Dropout(0.1))
        self.layers.append(nn.Conv1d(
            in_channels=output_dim, out_channels=output_dim, padding=1,
            kernel_size=3, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(output_dim))
        self.layers.append(nn.Dropout(0.1))

    def forward(self, h):
        for l in self.layers:
            h = l(h)
        return h


def multipleof8len(x):
    length = len(x)
    return 8 - (length % 8) + length


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


def seq_to_nonseq(sequences, seq_lengths):
    sequence_size = sequences.size()  # (S, B, D)
    nonseq = torch.cat([sequences[:length, i, :]
                        for i, length in enumerate(seq_lengths)], 0)
    # output size of (combined_length, D)
    return nonseq


def nonseq_to_seq(nonseq, seq_lengths):
    dim = nonseq.size(1)
    length = int(torch.max(seq_lengths))
    batch_size = seq_lengths.shape[0]

    sequences = nonseq.data.new(batch_size, length, dim).zero_()
    #sequences = to_variable(sequences)  # no need to set requires_grad=True

    lbl_ind = 0
    for i, seq_length in enumerate(seq_lengths):
        sequences[i, :seq_length, :] = nonseq[lbl_ind:lbl_ind+seq_length]
        lbl_ind += seq_length

    # output sequence (B, S, D)
    return sequences


class EncoderModel(nn.Module):
    # Encodes utterances to produce keys and values
    def __init__(self, args, input_dim):
        super(EncoderModel, self).__init__()
        self.cnn = CNN(args, input_dim, args.encoder_dim)
        self.rnns = nn.ModuleList()
        self.rnns.append(ExtendedLSTM(args.encoder_dim,
                                      args.encoder_dim, bidirectional=True))
        self.rnns.append(ExtendedLSTM(args.encoder_dim * 2,
                                      args.encoder_dim, bidirectional=True))
        self.rnns.append(PyramidalLSTM(args.encoder_dim * 4,
                                       args.encoder_dim, bidirectional=True))
        self.key_projection = MLP(
            args.encoder_dim * 2, args.key_dim, hiddims=[args.linear_dim])
        #self.value_projection = MLP(args, args.encoder_dim * 2, args.value_dim)

    def forward(self, utterances, utterance_lengths):
        h = utterances  # (B, S, D)

        # CNN
        h = torch.transpose(h, 1, 2)  # input (B, D, S)
        h = self.cnn(h)
        h = torch.transpose(h, 1, 2)  # (B, S, D)

        # listen
        h = torch.transpose(h, 0, 1)  # (S, B, D)

        sorted_lengths, order = torch.sort(
            utterance_lengths, 0, descending=True)
        _, backorder = torch.sort(order, 0)
        h = h[:, order, :]
        h = pack_padded_sequence(h, sorted_lengths.data.cpu().numpy())

        for rnn in self.rnns:
            h, _ = rnn(h)

        # Unpack and unsort the sequences
        h, output_lengths = pad_packed_sequence(h)
        h = h[:, backorder, :]
        output_lengths = output_lengths[backorder]

        values = torch.transpose(h, 0, 1)  # (B, S, D)

        # input of seq_to_nonseq should be (S, B, D)
        # (sum of length of valid labels, D)
        non_seq_hid = seq_to_nonseq(h, output_lengths)
        h1 = self.key_projection(non_seq_hid)
        keys = nonseq_to_seq(h1, output_lengths)  # (B, u, D)

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
        self.input_rnns.append(ExtendedLSTMCell(
            args.decoder_dim + self.value_dim, args.decoder_dim))
        self.input_rnns.append(ExtendedLSTMCell(
            args.decoder_dim, args.decoder_dim))
        rnn_out_dim = args.decoder_dim * 2
        self.input_rnns.append(ExtendedLSTMCell(args.decoder_dim, rnn_out_dim))
        self.query_projection = MLP(
            rnn_out_dim, args.key_dim, hiddims=[args.linear_dim])
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

        batch = keys.size(0)  # B
        max_feature_length = torch.max(feature_lengths)
        if max_label_length is not None:
            if int(max_feature_length) < int(max_label_length):
                print("features length: {} smaller that label length: {}".format(
                    int(max_feature_length), int(max_label_length)))
        length = max_feature_length  # TODO: this might not be enough
        if max_label_length is not None:
            # label length
            length = max_label_length

        # Initial states
        input_states = [rnn.initial_state(batch) for rnn in self.input_rnns]

        # Initial context
        h0 = input_states[-1][0]  # (B, D)
        query = self.query_projection(h0)

        # compute mask
        mask = output_mask(max_feature_length, feature_lengths)
        mask = torch.transpose(mask, 0, 1).unsqueeze(2).float()
        #mask = to_variable(torch.transpose(mask, 0, 1).unsqueeze(2)).float()
        context = compute_context(keys, query, values, mask)
        #context = to_variable(keys.data.new(batch, 1, self.value_dim).zero_())

        #pred_char = to_variable(context.data.new(batch, 1).zero_().long())
        pred_char = context.data.new(batch, 1).zero_().long()

        logits = []
        generateds = []
        greedys = []
        for i in range(length):
            # input states: pre_context_states (si-1), input_y (yi-1), context (ci-1)
            # compute phi
            input_char = pred_char
            if input_y is not None:
                input_char = self.getInputChar(pred_char, input_y[:, i:i+1])

            h = self.embedding(input_char)  # (B, 1, D2)
            # context (B, 1, D)
            rnn_input = torch.cat((h, context), 2)
            rnn_input = torch.squeeze(rnn_input)  # (B, D)
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

    def getInputChar(self, pred, input):
        """
        Sample input from prediction of last time step and from ground-truth,
        the probability of choosing ground-truth is given by self.net_out_prob.

        Parameters
        ----------

        pred: int tensor
            character output prediction of shape (B, 1).
        input: int tensor
            character ground-truth input of shape (B, 1).
        """
        p = torch.ones((pred.size()), out=pred.data.new(
            pred.size())).float() * self.net_out_prob
        cond1 = torch.bernoulli(p).long()
        cond2 = 1 - cond1

        sampled_input = cond1 * input + cond2 * pred

        return sampled_input

    def calculate_prob(self, states, t, pred_char, keys, values, mask):
        # return output logits, newstates
        (input_states, query, context) = states

        input_char = pred_char.long()

        h = self.embedding(input_char)  # (B, 1, D2)

        # context (B, 1, D)
        rnn_input = torch.cat((h, context), 2)
        rnn_input = torch.squeeze(rnn_input, 1)  # (B, D)
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
        h0 = input_states[-1][0]  # (S, B, D)
        query = self.query_projection(h0)

        # compute mask
        mask = output_mask(int(utterance_length), utterance_length)
        mask = torch.transpose(mask, 0, 1).unsqueeze(2).float()
        #mask = to_variable(torch.transpose(mask, 0, 1).unsqueeze(2)).float()
        context = compute_context(keys, query, values, mask)

        return (input_states, query, context), mask


def output_mask(maxlen, lengths):
    # TODO: Need a clearer docstring.
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


class CrossEntropyLossMask(nn.CrossEntropyLoss):
    '''
    calculate the gradient of valid label only
    input (B, S, C), target (total length of valid labels, )
    '''

    def __init__(self, *args, **kwargs):
        super(CrossEntropyLossMask, self).__init__(
            *args, reduce=False, **kwargs)
        #super(CrossEntropyLossMask, self).__init__(weight=None, size_average=True, ignore_index=-100, reduce=True)

    def forward(self, logits, target, label_lengths):
        logits_size = logits.size()  # (B, S, D)

        # generate this length in both training and evaluation
        # length should be the same for target and logits
        maxlen = target.size(1)

        mask = output_mask(maxlen, label_lengths)
        mask = torch.transpose(mask, 0, 1).float()
        #mask = to_variable(mask).float()  # (B, S)

        logits = logits * mask.unsqueeze(2)
        losses = super(CrossEntropyLossMask, self).forward(
            logits.view(-1, logits_size[2]), target.view(-1))

        # two steps into one, but we might need the losses of each sentence
        loss = torch.sum(mask.view(-1) * losses) / logits_size[0]

        #masked_loss = mask.view(-1) * losses
        #reshape_losses = masked_loss.view(logits_size[0], logits_size[1]).sum(1)

        # take the mean over mini-batch
        #loss = reshape_losses.mean()

        return loss


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

    masked_pre_alpha = masked_pre_alpha.transpose(1, 2)  # (B, 1, u)

    # will give nan
    #mask_sum = torch.sum(masked_pre_alpha, 2, keepdim=True)
    #alpha = masked_pre_alpha / mask_sum

    # normalize
    alpha = F.normalize(masked_pre_alpha, p=1, dim=2)

    # (B, 1, u) (B, u, D) -> (B, 1, D)
    context = torch.bmm(alpha, values)
    return context


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

        h_size = h.size()  # (B, 1, D3)
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
        logits, generateds, greedys = self.decoder(
            chars, keys, values, max_label_length, seq_lengths, eval=eval)
        return logits, generateds, greedys


def params_str(args, trloss, vlloss, epoch, edit_distance):
    params = vars(args)
    params_str_ = sorted(params.items(), key=lambda x: x[0])
    return params['output_dir'] + '/' + \
        "trloss_%.2f_vlloss_%.2f_epoch_%d_L_%.2f_" % (trloss, vlloss, epoch, edit_distance) + \
        "_".join(["{}_{}".format(k, v)
                  for k, v in params_str_ if k not in ['output_dir', 'epochs']])


def save_args(args):
   # Save argparse arguments to a file for reference
    with open(os.path.join(args.output_dir, 'args.txt'), 'a') as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


def convert_to_string(tokens, transmap):
    # TODO: Consider put this into the optimizer class
    chars = []
    tokens = tokens.data.cpu().numpy()
    strings = []
    for token in tokens:
        for x in token:
            if x == 1:
                break
            chars.append(transmap.labeldict[x])
        strings.append(''.join(chars))
        chars.clear()
    return strings


def unshuffle(x, unshuffle_idx):
    x = x.data.cpu().numpy()
    res = np.zeros(x.shape).astype(int)
    for i, idx in enumerate(unshuffle_idx):
        res[idx, :] = x[i]
    return res


class model_data_optim():
    # TODO: Add docstring.
    # TODO: Consider refactorizing this class in something like nn/optim.py.
    def __init__(self, dataloader, valid_dataloader, test_data_loader, args,
                 init_bias, transmap, input_dim, vocab_size):
        self.net = Seq2SeqModel(args, input_dim, vocab_size, init_bias)
        print(self.net)
        self.data_loader = dataloader
        self.valid_data_loader = valid_dataloader
        self.test_data_loader = test_data_loader
        self.optimizer = optim.Adam(self.net.parameters(
        ), lr=args.init_lr, weight_decay=args.weight_decay)
        self.criterion = CrossEntropyLossMask()
        self.best_validation_loss = float('inf')
        self.model_param_str = args.output_dir
        self.best_model_param_str = self.model_param_str
        self.args = args
        self.tmap = transmap
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        save_args(args)
        print(args)
        #if torch.cuda.is_available():
        #    self.net.cuda()
        #    self.criterion.cuda()

    def train(self, nepochs=None):
        start_time = time.time()
        nepochs = self.args.epochs if not nepochs else nepochs
        for epoch in range(nepochs):
            self.net.train()

            self.adjust_lr(epoch)
            losses = []

            for sequence, seq_length, input, output, label_length in self.data_loader:

                self.optimizer.zero_grad()  # Reset the gradients

                max_label_length = torch.max(label_length)
                logits, generateds, greedys = self.net(
                    sequence, seq_length, input, max_label_length)
                loss = self.criterion(logits, output, label_length)  # weird 2
                loss.backward(retain_graph=True)
                losses.append(loss.data.cpu().numpy())
                print("epoch: {}, loss: {}".format(
                    epoch, loss.data.cpu().item()))
                torch.nn.utils.clip_grad_norm(
                    self.net.parameters(), 0.25)  # gradient clip
                self.optimizer.step()
            loss_print = np.asscalar(np.mean(losses))
            print("epoch: {}, loss: {}".format(epoch, loss_print))

            # evaluate at the end of each epoch
            evaluation_loss, edit_distance = self.eval_model()
            self.model_param_str = params_str(
                self.args, loss_print, evaluation_loss, epoch, edit_distance)
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
                f.write("epoch: {}, training loss: {:.4f}, validation loss: {:.4f}\n---------epoch time: {}---------\n".format(
                    epoch, loss_print, evaluation_loss, start_time - old_time))

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

        print("----------adjusting learning rate: {}, net_out_prob: {}----------".format(lr,
                                                                                         self.args.net_out_prob))

    def eval_model(self):
        self.net.eval()
        losses = []
        mean_edit_distance1 = 0.
        mean_edit_distance2 = 0.
        total_cnt = 0
        for sequence, seq_length, input, output, label_length in self.valid_data_loader:
            total_cnt += 1

            max_label_length = torch.max(label_length)
            # TODO: to_variable(input)
            #logits, generateds, greedys = self.net(to_variable(
            #    sequence), seq_length, None, max_label_length, eval=True)  # weird
            logits, generateds, greedys = self.net(
                sequence, seq_length, None, max_label_length, eval=True)
            # print out
            greedy_str = convert_to_string(greedys, self.tmap)
            print("greedy", greedy_str)

            generated_str = convert_to_string(generateds, self.tmap)
            print("generated", generated_str)

            label_str = convert_to_string(output, self.tmap)
            print("outputs", label_str)

            ls1 = 0.
            ls2 = 0.
            for pred1, pred2, true in zip(generated_str, greedy_str, label_str):
                ls1 += levenshtein(pred1, true)[0]
                ls2 += levenshtein(pred2, true)[0]

            ls1 /= len(label_str)
            ls2 /= len(label_str)
            mean_edit_distance1 += ls1
            mean_edit_distance2 += ls2

            # prediction and output could both be longer than each other
            #loss = self.criterion(logits, to_variable(
            #    output), label_length)  # weird 2
            loss = self.criterion(logits, output, label_length)  # weird 2
            losses.append(loss.data.cpu().numpy())

        loss_print = np.asscalar(np.mean(losses))
        mean_edit_distance1 /= total_cnt
        mean_edit_distance2 /= total_cnt
        print("edit distance1: {}, edit distance2: {}, validation loss: {}".format(
            mean_edit_distance1, mean_edit_distance2, loss_print))
        return loss_print, mean_edit_distance2

    def load_model(self, modelpath):
        self.net.load_state_dict(torch.load(modelpath + '.pkl'))
        self.model_param_str = modelpath

    def write_predictions(self):
        self.net.eval()

        with open(self.model_param_str + '.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Id', 'Predicted'])
            id = 0
            for sequence, seq_length, _, _, _ in self.test_data_loader:

                #logits, generateds, greedys = self.net(
                #    to_variable(sequence), seq_length, None, None, eval=True)
                logits, generateds, greedys = self.net(
                    sequence, seq_length, None, None, eval=True)

                output_strs = convert_to_string(generateds, self.tmap)
                #print("generateds", output_strs)

                output_str = convert_to_string(greedys, self.tmap)
                #print("greedy", output_str)

                for output_str in output_strs:
                    w.writerow([id, output_str])
                    id += 1

def main(args, modelpath=None):
    """Training a attention model for speech recognition on WSJ."""
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vocab_size = len(CHARMAP)

    collate_fn = Compose([sort_by_feat_length_collate_fn, ToDevice(device)])
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(WSJ_TRAIN, shuffle=True,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn, **kwargs)
    valid_loader = DataLoader(WSJ_VALID, shuffle=True,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn, **kwargs)
    test_loader = DataLoader(WSJ_TEST, shuffle=False,
                             batch_size=args.batch_size,
                             collate_fn=collate_fn, **kwargs)

    # define and train
    # hack initialization for last layer bias with log unigram probability
    if modelpath is not None:
        init_bias = None
    else:
        init_bias = VOCAB_HIST / sum(VOCAB_HIST)

        smoothing = 0.1
        init_bias = (init_bias * (1. - smoothing)) + (smoothing / vocab_size)
        init_bias = torch.log(torch.from_numpy(init_bias))

    mdo = model_data_optim(train_loader, valid_loader, test_loader,
                           args, init_bias, CHARMAP, FEATDIM, vocab_size)

    model_param_str = None
    if not modelpath:
        model_param_str, _ = mdo.train()
    else:
        # load model
        mdo.load_model(modelpath)
        if args.epochs != 0:
            mdo.eval_model()
            model_param_str, _ = mdo.train()

    if model_param_str:  # load best model
        mdo.load_model(model_param_str)

    mdo.eval_model()
    mdo.write_predictions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=16)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--encoder_dim', dest='encoder_dim',
                        type=int, default=256)
    parser.add_argument('--key_dim', dest='key_dim', type=int, default=1024)
    parser.add_argument('--value_dim', dest='value_dim',
                        type=int, default=1024)
    parser.add_argument('--enhance_dim', dest='enhance_dim',
                        type=int, default=256)
    parser.add_argument('--decoder_dim', dest='decoder_dim',
                        type=int, default=1024)
    parser.add_argument('--linear_dim', dest='linear_dim',
                        type=int, default=1024)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        type=float, default=1e-6)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=1e-3)
    parser.add_argument('--net_out_prob', dest='net_out_prob',
                        type=float, default=0.1)
    parser.add_argument('--cuda', dest='cuda',
                        action='store_true',
                        help="Whether to use cuda in worker for dataloader.")
    parser.add_argument('--output_dir', dest='output_dir',
                        type=str, default='./vtlp')
    args = parser.parse_args(sys.argv[1:])

    #grid_search()
    main(args, modelpath=None)
    #beamsearch(args, modelpath='./vtlp/trloss_1.50_vlloss_837.96_epoch_7_L_17.18_batch_size_16_cuda_False_decoder_dim_1024_encoder_dim_256_enhance_dim_256_init_lr_0.0003_key_dim_1024_linear_dim_1024_net_out_prob_0.30000000000000004_value_dim_1024_weight_decay_1e-06')
