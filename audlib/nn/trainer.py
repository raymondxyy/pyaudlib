import os
import torch
import time
from torch import optim
import numpy as np

from audlib.nn.asr import Seq2SeqModel
from audlib.nn.loss import CrossEntropyLossMask
from audlib.nn.util import save_args, convert_to_string, params_str
from audlib.asr.util import levenshtein


class ASRTrainer:
    # TODO: Add docstring.
    # TODO: Consider refactorizing this class in something like nn/optim.py.
    # TODO: include dataset in trainer.
    def __init__(self, dataloader, valid_dataloader, test_data_loader, args,
                 init_bias, transmap, input_dim, vocab_size):
        self.net = Seq2SeqModel(args, input_dim, vocab_size, init_bias)
        print(self.net)
        self.data_loader = dataloader
        self.valid_data_loader = valid_dataloader
        self.test_data_loader = test_data_loader
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.init_lr,
                                    weight_decay=args.weight_decay)
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

    def train_model(self, num_epochs=None):
        start_time = time.time()
        num_epochs = self.args.epochs if not num_epochs else num_epochs
        for epoch in range(num_epochs):
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
            with open(os.path.join(self.args.output_dir, 'args.txt'),
                      'a') as f:
                f.write("save as:\n{}".format(self.model_param_str + '.pkl'))
            print("save as:\n{}".format(self.model_param_str + '.pkl'))
            torch.save(self.net.state_dict(), self.model_param_str + '.pkl')
            if evaluation_loss < self.best_validation_loss:
                self.best_validation_loss = evaluation_loss
                self.best_model_param_str = self.model_param_str

            old_time = start_time
            start_time = time.time()
            print(
                "--------epoch time: {}--------".format(start_time - old_time))
            with open(os.path.join(self.args.output_dir, 'args.txt'),
                      'a') as f:
                f.write(
                    "epoch: {}, training loss: {:.4f}, validation loss: {:.4f}\n---------epoch time: {}---------\n".format(
                        epoch, loss_print, evaluation_loss,
                        start_time - old_time))

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

        print(
            "----------adjusting learning rate: {}, net_out_prob: {}----------".format(
                lr,
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
            # logits, generateds, greedys = self.net(to_variable(
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
            for pred1, pred2, true in zip(generated_str, greedy_str,
                                          label_str):
                ls1 += levenshtein(pred1, true)[0]
                ls2 += levenshtein(pred2, true)[0]

            ls1 /= len(label_str)
            ls2 /= len(label_str)
            mean_edit_distance1 += ls1
            mean_edit_distance2 += ls2

            # prediction and output could both be longer than each other
            loss = self.criterion(logits, output, label_length)  # weird 2
            losses.append(loss.data.cpu().numpy())

        loss_print = np.asscalar(np.mean(losses))
        mean_edit_distance1 /= total_cnt
        mean_edit_distance2 /= total_cnt
        print(
            "edit distance1: {}, edit distance2: {}, validation loss: {}".format(
                mean_edit_distance1, mean_edit_distance2, loss_print))
        return loss_print, mean_edit_distance2

    def load_model(self, model_dir):
        self.net.load_state_dict(torch.load(model_dir + '.pkl'))
        self.model_param_str = model_dir

    def write_predictions(self):
        self.net.eval()
        idx = 0
        for sequence, seq_length, _, _, _ in self.test_data_loader:

            logits, generateds, greedys = self.net(
                sequence, seq_length, None, None, eval=True)

            output_strs = convert_to_string(generateds, self.tmap)
            # print("generateds", output_strs)

            output_str = convert_to_string(greedys, self.tmap)
            # print("greedy", output_str)

            for output_str in output_strs:
                print(idx, output_str)
                idx += 1
