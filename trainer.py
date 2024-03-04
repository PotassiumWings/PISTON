import logging
from datetime import datetime

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD

from configs.arguments import TrainingArguments
from dataset.processor import AbstractDataset
from models import loss


class Trainer:
    def __init__(self, config: TrainingArguments, model: nn.Module, dataset: AbstractDataset, time: str):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time = time
        self.save_path = './saved_dict/' + self.time + '.ckpt'
        self.best_val_loss = float('inf')
        self.start_time = datetime.now().timestamp()
        self.optimizer = None
        self.should_step = False
        self.current_num = self.accumulated_num = 0

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def train(self, time, start_epoch=0):
        train_iter, val_iter = self.dataset.train_iter, self.dataset.val_iter
        self.init_optimizer()

        self.model.train()

        xs, maes, rmses, mapes = [], [], [], []

        last_update_val = 0  # last batch to update val loss
        self.current_num = self.accumulated_num = 0
        for epoch in range(start_epoch, self.config.num_epoches):
            train_iter.shuffle()
            ys, preds = [], []
            jump_flag = False
            for i, (x, y) in enumerate(train_iter):
                ys.append(y)
                pred = self.model(x)
                preds.append(pred)

                loss = self.model.calculate_loss(y, pred)
                loss.backward()

                self.check_step(batch_size=x.size(0))
                self.calc_eta(iter=train_iter, start_epoch=start_epoch)

                logging.info(f"Training, {i}/{len(train_iter)}, {epoch}/{self.config.num_epoches}, "
                             f"loss: {round(loss.item(), 4)}, "
                             f"step: {self.should_step} "
                             f"eta: {self.eta}")

                if self.current_num % self.config.show_period == 0:
                    # evaluate
                    ys, preds = torch.cat(ys, dim=0), torch.cat(preds, dim=0)
                    self.model.eval()

                    train_loss = self.model.calculate_loss(ys, preds)

                    mets = self.eval(self.dataset.val_iter)
                    val_rmse, val_mape, val_mae = mets[0]

                    logging.info(
                        f"Ep {epoch}/{self.config.num_epoches}, iter {self.current_num / self.config.batch_size},"
                        f" train loss {round(train_loss.item(), 4)},"
                        f" val mae {round(val_mae, 4)} mape {round(val_mape, 4)} rmse {round(val_rmse, 4)}")

                    if val_mae < self.best_val_loss:
                        self.best_val_loss = val_mae
                        torch.save(self.model.state_dict(), self.save_path)
                        logging.info("Good, saving model.")
                        last_update_val = self.current_num

                    ys, residuals, preds = [], [], []
                    self.model.train()

                    if self.current_num - last_update_val > self.config.early_stop_batch:
                        jump_flag = True
                        logging.info("Long time since last update, early stopping.")
                        break
            if jump_flag:
                break

    def visualize(self, dir_name):
        self.model.visualize(dir_name)

    def eval(self, data_iter, debug=False, fast_eval=-1, tsne=-1):
        def metrics(pred, true):
            return loss.rmse_torch(pred, true, self.config.mae_mask).item(), \
                loss.mape_torch(pred, true, self.config.mae_mask).item(), \
                loss.mae_torch(pred, true, self.config.mae_mask).item()

        self.model.eval()
        logging.info("Evaluating...")
        trues, preds, xs = [], [], []
        with torch.no_grad():
            # data_iter.shuffle()
            for i, (x, y) in enumerate(data_iter):
                xs.append(x.cpu())
                trues.append(y.cpu())  # NCVL'
                pred = self.model(x)
                preds.append(pred.cpu())  # NCVL'

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        xs = torch.cat(xs, dim=0)

        ret = [metrics(preds, trues)]
        return ret

    def test(self):
        metrics = self.eval(self.dataset.test_iter)
        test_rmse, test_mape, test_mae = metrics[0]
        logging.info(f"Test: rmse {test_rmse}, mae {test_mae}, mape {test_mape}")
        for i in range(len(metrics) - 1):
            test_rmse, test_mape, test_mae = metrics[i + 1]
            logging.info(f"Test feature {i}: rmse {test_rmse}, mae {test_mae}, mape {test_mape}")
        logging.info(f"Calculation done.")
        logging.info(f"log name: {self.time}")

    def init_optimizer(self):
        if self.config.optimizer == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "SGD":
            self.optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise AttributeError(f"Optimizer {self.config.optimizer} was not implemented.")

    def check_step(self, batch_size):  # check if optimizer can do step()
        self.accumulated_num += batch_size
        self.current_num += batch_size
        self.should_step = self.accumulated_num >= self.config.accumulate_period
        if self.should_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_num = 0

    def calc_eta(self, iter, start_epoch):
        eta_s = (datetime.now().timestamp() - self.start_time) / self.current_num * \
                (iter.N * (self.config.num_epoches - start_epoch) - self.current_num)
        if eta_s < 60:
            eta = str(round(eta_s, 2)) + "s"
        elif eta_s < 3600:
            eta = str(round(eta_s / 60, 2)) + "min"
        else:
            eta = str(round(eta_s / 3600, 2)) + "h"
        self.eta = eta


def dwa(L_old, L_new, T=2):
    L_old = torch.Tensor(L_old)
    L_new = torch.Tensor(L_new)
    N = len(L_old)
    r = L_old / L_new
    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()
