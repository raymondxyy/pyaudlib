"""Loss function for Neural Networks."""
import torch
import torch.nn as nn
from audlib.nn.util import output_mask


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