"""Utility Functions and Neural Networks."""
import numpy as np
import os
import torch
import torch.nn.functional as F


def detach(states):
    """Truncate backpropagation (usually used in RNN)."""
    return [state.detach() for state in states]


def hasnan(m):
    """Check if torch.tensor m have NaNs in it."""
    return np.any(np.isnan(m.cpu().data.numpy()))


def printnn(model):
    """Print out neural network."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("{}[{}]\n{}".format('-' * 30, name, param.data.numpy()))


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
    # sequences = to_variable(sequences)  # no need to set requires_grad=True

    lbl_ind = 0
    for i, seq_length in enumerate(seq_lengths):
        sequences[i, :seq_length, :] = nonseq[
                                       lbl_ind:lbl_ind + seq_length]
        lbl_ind += seq_length

    # output sequence (B, S, D)
    return sequences


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

    # normalize
    alpha = F.normalize(masked_pre_alpha, p=1, dim=2)

    # (B, 1, u) (B, u, D) -> (B, 1, D)
    context = torch.bmm(alpha, values)
    return context


def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on https://github.com/ericjang/gumbel-softmax/blob/
    3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_argmax(logits, dim):
    # Draw from a multinomial distribution efficiently
    return torch.max(logits + sample_gumbel(logits.size(),
                                            out=logits.data.new()), dim)[1]


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


def params_str(args, trloss, vlloss, epoch, edit_distance):
    params = vars(args)
    params_str_ = sorted(params.items(), key=lambda x: x[0])
    return params['output_dir'] + '/' + \
           "trloss_%.2f_vlloss_%.2f_epoch_%d_L_%.2f_" % (
           trloss, vlloss, epoch, edit_distance) + \
           "_".join(["{}_{}".format(k, v)
                     for k, v in params_str_ if
                     k not in ['output_dir', 'epochs']])
