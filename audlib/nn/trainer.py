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
    def __init__(self, train_dataloader, valid_dataloader, test_dataloader,
                 args, init_bias, transmap, input_dim, vocab_size):
        self.net = Seq2SeqModel(args, input_dim, vocab_size, init_bias)
        print(self.net)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
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

    def train_for_one_iteration(self, epoch, losses):
        for feats, feat_lengths, inputs, outputs, label_lengths in self.train_dataloader:
            self.optimizer.zero_grad()  # Reset the gradients

            max_label_length = torch.max(label_lengths)
            logits, generateds, greedys = self.net(
                feats, feat_lengths, inputs, max_label_length)
            loss = self.criterion(logits, outputs, label_lengths)  # weird 2
            loss.backward(retain_graph=True)
            losses.append(loss.data.cpu().numpy())
            print("epoch: {}, loss: {}".format(
                epoch, loss.data.cpu().item()))
            torch.nn.utils.clip_grad_norm(
                self.net.parameters(), 0.25)  # gradient clip
            self.optimizer.step()

    def log_result(self, epoch, losses, start_time):
        # print training loss
        loss_print = np.asscalar(np.mean(losses))
        print("epoch: {}, loss: {}".format(epoch, loss_print))

        # evaluate model
        evaluation_loss, edit_distance = self.eval_model()

        # save parameter
        self.model_param_str = params_str(
            self.args, loss_print, evaluation_loss, epoch, edit_distance)
        with open(os.path.join(self.args.output_dir, 'args.txt'), 'a') as f:
            f.write("save as:\n{}".format(self.model_param_str + '.pkl'))
        print("save as:\n{}".format(self.model_param_str + '.pkl'))
        torch.save(self.net.state_dict(), self.model_param_str + '.pkl')
        if evaluation_loss < self.best_validation_loss:
            self.best_validation_loss = evaluation_loss
            self.best_model_param_str = self.model_param_str

        # print evaluation loss
        print("epoch: {}, evaluation loss: {}".format(epoch, evaluation_loss))

        # log loss
        last_time = time.time() - start_time
        print("--------epoch time: {}--------".format(last_time))
        with open(os.path.join(self.args.output_dir, 'args.txt'), 'a') as f:
            f.write(
                "epoch: {}, training loss: {:.4f}, validation loss: {:.4f}\n---------epoch time: {}---------\n".format(
                    epoch, loss_print, evaluation_loss, last_time))

        self.write_predictions()

    def train_model(self, num_epochs=None):
        num_epochs = self.args.epochs if not num_epochs else num_epochs
        for epoch in range(num_epochs):
            start_time = time.time()
            self.net.train()

            self.adjust_lr(epoch)
            losses = []
            self.train_for_one_iteration(epoch, losses)

            # evaluate at the end of each epoch
            self.log_result(epoch, losses, start_time)

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

    def prepare_evaluation_metrics(self):
        self.losses = []
        self.mean_edit_distance1 = 0.
        self.mean_edit_distance2 = 0.

    def update_evaluation_metrics(self, logits, label_lengths, outputs,
                                  greedy_res, generated_res):
        # loss
        loss = self.criterion(logits, outputs, label_lengths)
        self.losses.append(loss.data.cpu().numpy())

        greedy_str = convert_to_string(greedy_res, self.tmap)
        print("greedy", greedy_res)
        generated_str = convert_to_string(generated_res, self.tmap)
        print("generated", generated_str)
        label_str = convert_to_string(outputs, self.tmap)
        print("outputs", label_str)

        # L distance
        ls1 = 0.
        ls2 = 0.
        for pred1, pred2, true in zip(generated_str, greedy_str, label_str):
            ls1 += levenshtein(pred1, true)[0]
            ls2 += levenshtein(pred2, true)[0]

        ls1 /= len(label_str)
        ls2 /= len(label_str)

        self.mean_edit_distance1 += ls1
        self.mean_edit_distance2 += ls2

    def output_evaluation_metrics(self, total_cnt):
        loss_print = np.asscalar(np.mean(self.losses))
        self.mean_edit_distance1 /= total_cnt
        self.mean_edit_distance2 /= total_cnt
        print(
            "edit dist1: {}, edit dist2: {}, validation loss: {}".format(
                self.mean_edit_distance1, self.mean_edit_distance2,
                loss_print))

    def eval_model(self):
        self.net.eval()

        self.prepare_evaluation_metrics()
        total_cnt = 0
        for feats, feat_lengths, inputs, outputs, label_lengths in self.valid_dataloader:
            total_cnt += 1

            max_label_length = torch.max(label_lengths)
            logits, generated_res, greedy_res = self.net(
                feats, feat_lengths, None, max_label_length, eval=True)

            self.update_evaluation_metrics(greedy_res, generated_res, outputs)

        self.output_evaluation_metrics(total_cnt)

    def load_model(self, model_dir):
        self.net.load_state_dict(torch.load(model_dir + '.pkl'))
        self.model_param_str = model_dir

    def write_predictions(self):
        self.net.eval()
        idx = 0
        for feats, feat_lengths, _, _, _ in self.test_dataloader:

            logits, generateds, greedys = self.net(
                feats, feat_lengths, None, None, eval=True)

            output_strs = convert_to_string(generateds, self.tmap)
            # print("generateds", output_strs)

            output_str = convert_to_string(greedys, self.tmap)
            # print("greedy", output_str)

            for output_str in output_strs:
                print(idx, output_str)
                idx += 1
