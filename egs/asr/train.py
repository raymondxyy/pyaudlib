"""Main script for training DNNs for speech recognition."""

import numpy as np
import torch
import torch.utils.data
from torch import from_numpy
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence

from audlib.nn.util import UnpackedSequence

from dataset import FEATDIM, CHARMAP, WSJ_TRAIN, WSJ_VALID, WSJ_TEST
from models import ListenAttendSpell


def main(args):
    """Training a attention model for speech recognition on WSJ."""
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load data, and define collating function
    def _collate(seqs):
        """Pack a batch of sequences for training."""
        feats, inseqs, targets = zip(*seqs)
        seqlens = [len(ff) for ff in feats]
        _, feats, inseqs = zip(
            *sorted(zip(seqlens, feats, inseqs),
                    key=lambda x: x[0], reverse=True)
        )
        feats = pack_sequence([from_numpy(f).float() for f in feats])
        if inseqs:
            inseqs = [from_numpy(f).long() for f in inseqs]
        # NOTE: Target has different order as in feats
        targets = pack_sequence(
            sorted([from_numpy(f).long() for f in targets],
                   key=len, reverse=True)
        )
        return feats, inseqs, targets

    kwargs = {'num_workers': args.num_workers} if args.gpu else {}
    trainld = DataLoader(WSJ_TRAIN, shuffle=True,
                         batch_size=args.batch_size,
                         collate_fn=_collate, **kwargs)
    validld = DataLoader(WSJ_VALID, shuffle=True,
                         batch_size=args.batch_size,
                         collate_fn=_collate, **kwargs)
    testld = DataLoader(WSJ_TEST, shuffle=False,
                        batch_size=args.batch_size,
                        collate_fn=_collate, **kwargs)

    # Construct NN model
    model = ListenAttendSpell(
        FEATDIM, args.encdim, len(CHARMAP.vocabdict)).to(device)
    print(f"+++ ListenAttendSpell has [] parameters +++")
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    def adjust_lr(nepoch):
        lr = max(1e-5, args.learning_rate * np.exp(-nepoch/5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"----------NEW LEARNING RATE: [{lr}] ----------")

    # Train the model
    for epoch in range(args.num_epochs):
        err_epo = 0
        adjust_lr(epoch)
        model.train()
        for ii, (feats, inseqs, targets) in enumerate(trainld):
            # Set torch variables; assume these are PackedSequence
            feats = feats.to(device)
            inseqs = [ss.to(device) for ss in inseqs]
            targets = targets.to(device)

            # Forward pass
            probtru = 1  #.5 * np.exp(-ii / (len(trainld)//5))
            outseqs = model(feats, inseqs, probtru=probtru)

            # Caculate error
            xerr = 0
            for oo, tt in zip(outseqs, UnpackedSequence(targets)):
                # Ignore fake target nodes
                print("[PRED]: ", end='')
                ooidx = oo[:len(tt)].argmax(dim=1)
                print(''.join(CHARMAP.labeldict[int(ii)] for ii in ooidx))
                print("[TARG]: ", end='')
                print(''.join(CHARMAP.labeldict[int(ii)] for ii in tt))
                xerr += criterion(oo[:len(tt)], tt)
            xerr /= len(inseqs)
            cost = xerr

            # Calculate gradients and update network weights
            optimizer.zero_grad()
            cost.backward()

            optimizer.step()

            # Print log for current step
            err_epo += cost.data
            errdict = {"cost": cost, "XEntLoss": xerr}
            errstr = ", ".join(
                "{}: {:.3f}".format(k, v) for k, v in errdict.items()
            )
            if (ii + 1) % args.log_step == 0:
                print(
                    "Epoch [{:3d}/{:3d}], Step [{:3d}/{:3d}], {}".format(
                        epoch+1, args.num_epochs, ii+1, len(trainld), errstr
                    )
                )

        print("START VALIDATION:")
        valid_epo = 0
        model.eval()
        for ii, (feats, inseqs, targets) in enumerate(validld):
            feats = feats.to(device)
            inseqs = [ss.to(device) for ss in inseqs]
            targets = targets.to(device)

            # Forward pass
            outseqs = model(feats, inseqs, train=False)

            # Caculate error
            xerr = 0
            for oo, tt in zip(outseqs, UnpackedSequence(targets)):
                print("[PRED]: ", end='')
                ooidx = oo[:len(tt)].argmax(dim=1)
                print(''.join(CHARMAP.labeldict[int(ii)] for ii in ooidx))
                print("[TARG]: ", end='')
                print(''.join(CHARMAP.labeldict[int(ii)] for ii in tt))
                xerr += criterion(oo[:len(tt)], tt)
            valid_epo += (xerr.data / len(inseqs))

            errdict = {"Prediction Error": valid_epo}
            errstr = ", ".join(
                "{}: {:.3f}".format(k, v) for k, v in errdict.items()
            )
            if (ii + 1) % args.log_step == 0:
                print(
                    "Epoch [{:3d}/{:3d}], Step [{:3d}/{:3d}], {}".format(
                        epoch+1, args.num_epochs, ii+1, len(validld), errstr
                    )
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', action='store_true')
    parser.add_argument("--no-gpu", dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
    parser.add_argument("--learning-rate", dest='learning_rate',
                        type=float, default=1e-3)
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=16)
    parser.add_argument('--num-epochs', dest='num_epochs',
                        type=int, default=20)
    parser.add_argument('--encoder-dim', dest='encdim',
                        type=int, default=256)
    parser.add_argument('--prob-truth', dest='probtru',
                        type=float, default=1)
    parser.add_argument('--weight-decay', dest='weight_decay',
                        type=float, default=1e-6)
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of threads."
    )
    parser.add_argument("--log-step", dest='log_step',
                        type=int, default=1)
    args = parser.parse_args()

    main(args)
