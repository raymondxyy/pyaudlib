"""Neural network models for automatic speech recognition."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F

from audlib.nn.rnn import MLP, ExtendedLSTM, PyramidalLSTM, UnpackedSequence


class ListenAttendSpell(nn.Module):
    """A loose implementation of Listen-Attend-Spell by Chan et al."""
    def __init__(self, featdim, encdim, vocab_size,
                 decimate=2, bidirectional=True, nhidden=2):
        super(ListenAttendSpell, self).__init__()
        self.encdim = encdim
        self.feat_extractor = SeqEncoder(featdim, decimate, bidirectional)
        self.feat_encoder = MLP(
            (2 if bidirectional else 1)*featdim, encdim,
            hiddims=[(2 if bidirectional else 1)*featdim]*nhidden,
            batchnorm=[True]*nhidden
        )
        self.char_encoder = nn.Sequential(
            nn.Embedding(vocab_size, encdim),
            MLP(encdim, encdim, hiddims=[encdim]*nhidden,
                batchnorm=[True]*nhidden)
        ) 
        # Symbol-context extractor and decoder
        self.sc_extractor = nn.ModuleList([
            ExtendedLSTM(2*encdim, encdim),
            ExtendedLSTM(encdim, encdim),
            ExtendedLSTM(encdim, encdim)
        ])
        self.sc_decoder = nn.Sequential(
            nn.Linear(encdim, encdim),
            nn.LeakyReLU(),
            nn.Linear(encdim, vocab_size)
        )

    @staticmethod
    def attend(hs, ss):
        """Apply attention.

        Parameters
        ----------
        hs: torch.tensor
            Encoded acoustic features with dimension time x encdim.
        ss: torch.tensor
            Encoded vocabulary features with dimension encdim.
        """
        attention = F.softmax(hs.mv(ss), dim=0)
        return hs.t().mv(attention)

    @staticmethod
    def trupred(truth, pred, prob):
        """
        Sample input from prediction of last time step and from ground-truth,
        the probability of choosing ground-truth is given by self.net_out_prob.

        Parameters
        ----------

        """
        ptru = torch.bernoulli(torch.ones_like(pred).float() * prob).long()

        return ptru * truth + (1-ptru) * pred

    def forward(self, feats, chars, train=True, maxlen=None, probtru=1):
        """Forward one character at a time.
        
        In general, there are three modes:
        1. Batch train with groundtruth: forward(feats, groundtruth)
        2. Train with prediction & groundtruth: forward(feats, gt, probtru=.5)
        3. Prediction: forward(feats, bos, train=False)
        """
        assert isinstance(feats, PackedSequence)
        # Encode acoustic feature first
        feats = self.feat_extractor(feats)
        hseqs = UnpackedSequence(
            PackedSequence(data=self.feat_encoder(feats.data),
                           batch_sizes=feats.batch_sizes)
        )

        if train and (probtru == 1):  # batch prediction using groundtruth
            sseqs = [self.char_encoder(c) for c in chars]
            return self.batchforward(hseqs, sseqs)

        if not maxlen:
            maxlen = max(len(c) for c in chars)
        ss = [self.char_encoder(torch.tensor([c[0] for c in chars]).to(
            feats.data.device))]
        cc = [torch.stack([self.attend(h, c) for h, c in zip(hseqs, ss[0])])]
        zis = [rnn.initial_state(len(chars)) for rnn in self.sc_extractor]
        sc = torch.cat((ss[0], cc[0]), dim=1).unsqueeze(0)
        for ii, (rnn, zi) in enumerate(zip(self.sc_extractor, zis)):
            sc, zis[ii] = rnn(sc, zi)
        logits = [self.sc_decoder(sc[0])]
        for tt in range(1, maxlen):
            pred = torch.argmax(logits[tt-1], dim=1)
            if train:
                # For timestamps without groundtruth, simply use fake label
                # NOTE: training script is responsible for dealing with this
                tru = torch.tensor(
                    [c[tt] if len(c) > tt else 0 for c in chars]).to(
                        feats.data.device).long()
                ss.append(self.char_encoder(self.trupred(tru, pred, probtru)))
            else:
                ss.append(self.char_encoder(pred))
            cc.append(
                torch.stack([self.attend(h, c) for h, c in zip(hseqs, ss[tt])])
            )
            sc = torch.cat((ss[tt], cc[tt]), dim=1).unsqueeze(0)
            for ii, (rnn, zi) in enumerate(zip(self.sc_extractor, zis)):
                sc, zis[ii] = rnn(sc, zi)
            logits.append(self.sc_decoder(sc[0]))

        logits = torch.stack(logits)
        return [logits[:, b, :] for b in range(len(chars))]


    def batchforward(self, hseqs, sseqs):
        """Forward all groundtruth characters at once."""
        contexts = []
        for hs, ss in zip(hseqs, sseqs):
            attention = F.softmax(torch.mm(ss, hs.t()), dim=1)
            contexts.append(torch.mm(attention, hs))
        sc = pack_sequence(
            sorted([torch.cat((s, c), dim=1) for s, c in zip(sseqs, contexts)],
                   reverse=True,
                   key=len)
        )

        for rnn in self.sc_extractor:
            sc, _ = rnn(sc)

        logits = UnpackedSequence(
            PackedSequence(data=self.sc_decoder(sc.data),
                           batch_sizes=sc.batch_sizes)
        )

        return logits


class SeqEncoder(nn.Module):
    def __init__(self, indim, decimate=2, bidirectional=True):
        super(SeqEncoder, self).__init__()
        kwargs = {'decimate': decimate, 'bidirectional': bidirectional}
        di = 2 if bidirectional else 1
        self.plstms = nn.ModuleList([
            PyramidalLSTM(decimate*indim, indim, **kwargs),
            PyramidalLSTM(di*decimate*indim, indim, **kwargs),
            PyramidalLSTM(di*decimate*indim, indim, **kwargs)
        ])

    def forward(self, x):
        """Forward-pass a PackedSequence through the sequence encoder."""
        assert isinstance(x, PackedSequence), "Input must be PackedSequence."
        for rnn in self.plstms:
            x, _ = rnn(x)

        return x


if __name__ == "__main__":
    import random
    import numpy as np
    import torch

    # Test Encoder
    indim, encdim = 10, 5
    batchsize = 8
    net = SeqEncoder(indim)
    print(net)
    xlst = [np.zeros((int((batchsize-ii)*7), indim))+ii for ii in range(batchsize)]
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

    # Test seq2seq
    vocabsize = 10
    model = ListenAttendSpell(indim, encdim, vocabsize)
    print(model)
    feats = x
    chars = [torch.from_numpy(np.array([random.randrange(vocabsize) for _ in range(2*(b+1))])).long() for b in range(batchsize)]
    y = model(feats, chars)
    print([len(p) for p in y])

    # Test evaluation
    z = model(feats, [torch.ones(1).long() for b in range(batchsize)], train=False, maxlen=10)
    z = model(feats, [torch.ones(10).long() for b in range(batchsize)], probtru=.5)
    loss = sum(torch.mean(seg) for seg in z)
    loss.backward()
