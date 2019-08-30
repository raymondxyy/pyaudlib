"""Neural network models for automatic speech recognition."""

import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_sequence

from audlib.nn.rnn import MLP, ExtendedLSTMCell, ExtendedLSTM,\
    PyramidalLSTM, UnpackedSequence


class Seq2SeqModel(nn.Module):
    # Tie encoder and decoder together
    def __init__(self, args, input_dim, vocab_size, init_bias):
        super(Seq2SeqModel, self).__init__()
        self.encoder = SeqEncoder(args, input_dim)
        self.decoder = DecoderModel(args, vocab_size, init_bias)

    def forward(self, utterances, utterance_lengths, chars, max_label_length, eval=False):
        keys, values, seq_lengths = self.encoder(utterances, utterance_lengths)
        logits, generateds, greedys = self.decoder(
            chars, keys, values, max_label_length, seq_lengths, eval=eval)
        return logits, generateds, greedys


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


class SeqEncoder(nn.Module):
    def __init__(self, indim, encdim, hiddims=[]):
        super(SeqEncoder, self).__init__()
        self.rnns = nn.ModuleList([
            ExtendedLSTM(indim, indim, bidirectional=True),
            ExtendedLSTM(2*indim, indim, bidirectional=True),
            PyramidalLSTM(4*indim, indim, bidirectional=True)
        ])
        self.encoder = MLP(2*indim, encdim, hiddims=hiddims)

    def forward(self, x):
        """Forward-pass a PackedSequence through the sequence encoder."""
        assert isinstance(x, PackedSequence), "Input must be PackedSequence."
        for rnn in self.rnns:
            x, _ = rnn(x)
        return PackedSequence(data=self.encoder(x.data),
                              batch_sizes=x.batch_sizes)


if __name__ == "__main__":
    import numpy as np
    import torch
    # Test Encoder
    indim, encdim = 10, 5
    net = SeqEncoder(indim, encdim)
    print(net)
    xlst = [np.zeros((int((8-ii)*7), indim))+ii for ii in range(8)]
    x = pack_sequence(
        sorted([torch.from_numpy(x).to(dtype=torch.float32) for x in xlst],
               reverse=True,
               key=len,
               )
    )
    y = net(x)
    print(y.data.shape)
    loss = y.data.mean()
    loss.backward()
