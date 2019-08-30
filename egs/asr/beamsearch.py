def make_new_beam():
    def fn(): return -float("inf")
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
        #if torch.cuda.is_available():
        #    self.cuda()

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

    def load_model(self, modelpath):
        self.load_state_dict(torch.load(modelpath + '.pkl'))

    def beamsearch_through_sequence(self, utterance_length, keys, values):
        (key1, key2, key3) = keys
        (value1, value2, value3) = values
        states1, mask = self.decoder.get_initial_states(
            key1, value1, utterance_length)
        states2, mask = self.decoder.get_initial_states(
            key2, value2, utterance_length)
        states3, mask = self.decoder.get_initial_states(
            key3, value3, utterance_length)
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
                      key=lambda x: x[1],
                      reverse=True)
        best = beam[0]
        print(best)
        return best[0], best[1], beam

    def beamsearch_one_step(self, beam, complete_sequences, t, keys, values, mask):
        next_beam = make_new_beam()
        key1, key2, key3 = keys
        value1, value2, value3 = values

        for prefix, (states1, states2, states3, p_b, prev_output) in beam:  # Loop over beam
            # use previous output as input
            pred_char = torch.IntTensor([[prev_output]])

            output1, newstates1 = self.decoder.calculate_prob(states1, t, pred_char,
                                                              key1, value1, mask)
            output2, newstates2 = self.decoder.calculate_prob(states2, t, pred_char,
                                                              key2, value2, mask)
            output3, newstates3 = self.decoder.calculate_prob(states3, t, pred_char,
                                                              key3, value3, mask)
            output = (output1 + output2 + output3) / 3
            logprobs = F.log_softmax(output, 2).data.cpu().numpy()

            for s_index in range(self.vocab_size):  # loop over vocabulary

                p = logprobs[0, 0, s_index]
                n_p_b = p_b + p  # included the prob of the last

                if s_index != self.eos_index:
                    n_prefix = prefix + self.STRINGS[s_index]
                    # *NB* this would be a good place to include an LM score.
                    next_beam[n_prefix] = (
                        newstates1, newstates2, newstates3, n_p_b, s_index)
                else:
                    # normalize by length
                    complete_sequences.append((prefix, n_p_b / (t+1)))

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: x[1][-2],
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
        #if torch.cuda.is_available():
        #    self.cuda()

    def forward(self, utterance, utterance_length):
        keys, values, seq_lengths = self.encoder(utterance, utterance_length)
        labels, score, beams = self.beamsearch_through_sequence(seq_lengths,
                                                                keys, values)
        return labels, score, beams

    def load_model(self, modelpath):
        self.load_state_dict(torch.load(modelpath + '.pkl'))

    def beamsearch_through_sequence(self, utterance_length, keys, values):
        states, mask = self.decoder.get_initial_states(
            keys, values, utterance_length)
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
                      key=lambda x: x[1],
                      reverse=True)
        best = beam[0]
        return best[0], best[1], beam

    def beamsearch_one_step(self, beam, complete_sequences, t, keys, values, mask):
        next_beam = make_new_beam()

        for prefix, (states, p_b, prev_output) in beam:  # Loop over beam
            # use previous output as input
            pred_char = torch.IntTensor([[prev_output]])

            output, newstates = self.decoder.calculate_prob(states, t, pred_char,
                                                            keys, values, mask)
            logprobs = F.log_softmax(output, 2).data.cpu().numpy()

            decoded_char_idx = np.argmax(
                F.softmax(output, 2).data.cpu().numpy(), 2)[0][0]
            #print("greedy choice: {}, logprob: {}".format(self.STRINGS[decoded_char_idx], logprobs[0,0,decoded_char_idx]))

            for s_index in range(self.vocab_size):  # loop over vocabulary
                #print("this guy: {}, logprob: {}".format(self.STRINGS[s_index], logprobs[0,0,s_index]))

                p = logprobs[0, 0, s_index]
                n_p_b = p_b + p  # included the prob of the last

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
                      key=lambda x: x[1][1],
                      reverse=True)
        beam = beam[:self.beam_size]

        return beam, complete_sequences


def beamsearch(args, modelpath):
    # TODO: Add docstring.
    # BUG: Update code with new interface as in main.
    wsj = WSJ()

    STRINGS = [
        x for x in "&*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz #'-/@_"]
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

    beamsearcher = BeamSearchVtlp(
        args, INPUT_DIM, vocab_size, STRINGS, beam_size=3, eos_index=1)
    beamsearcher.load_model(modelpath)
    beamsearcher.eval()
    losses = []
    with open(modelpath + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Predicted'])
        id = 0
        for loader1, loader2, loader3 in zip(test_loader1, test_loader2, test_loader3):
            (sequence1, seq_length1, _, _, _) = loader1
            (sequence2, seq_length2, _, _, _) = loader2
            (sequence3, seq_length3, _, _, _) = loader3

            #sequences = (to_variable(sequence1), to_variable(
            #    sequence2), to_variable(sequence3))
            sequences = (sequence1, sequence2, sequence3)
            seq_lengths = (seq_length1, seq_length2, seq_length3)
            labels, score, beams = beamsearcher(
                sequences, seq_lengths)  # weird
            w.writerow([id, labels])
            id += 1
