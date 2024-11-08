import logging
import math
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam, AdamW, SGD

from configs.arguments import TrainingArguments
from dataset.processor import AbstractDataset
from models import loss
from utils.file_utils import ensure_dir


class Trainer:
    def __init__(self, config: TrainingArguments, model: nn.Module, dataset: AbstractDataset, time: str):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time = time
        self.save_path = './saved_dict/' + self.time + '.ckpt'
        self.best_val_loss = float('inf')
        self.clip = config.clip
        self.start_time = datetime.now().timestamp()
        self.temperature = self.config.start_temperature  # for gumbel sampling

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def train(self, time, start_epoch=0):
        train_iter, val_iter = self.dataset.train_iter, self.dataset.val_iter

        loss_last = None
        loss_weights = np.array([self.config.loss_1, self.config.loss_2, self.config.loss_3,
                                 self.config.loss_4, self.config.loss_5])

        lamb_data = torch.Tensor([[self.config.lamb]]).to(self.device)
        lamb = nn.Parameter(data=lamb_data, requires_grad=False)

        # 试过把 lambda 作为参数，不彳亍，收敛到 -inf/+inf 去了
        # self.model.register_parameter("lambda", lamb)

        if self.config.optimizer == "Adam":
            optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "SGD":
            optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise AttributeError(f"Optimizer {self.config.optimizer} was not implemented.")

        self.model.train()

        xs, maes, rmses, mapes = [], [], [], []
        l1, l2, l3, l4 = [], [], [], []

        last_update_val = 0  # last batch to update val loss
        current_num = 0
        accumulated_num = 0
        for epoch in range(start_epoch, self.config.num_epoches):
            train_iter.shuffle()
            ys, residuals, preds = [], [], []
            jump_flag = False
            for i, (x, y) in enumerate(train_iter):
                # torch.cuda.empty_cache()
                # x: N Ci V Li
                # y: N Co V Lo=1
                ys.append(y)
                residual, pred, _, _, forward_loss = self.model(x, self.temperature)
                residuals.append(residual)
                preds.append(pred)

                loss, losses = self.model.calculate_loss(residual, pred, y, loss_weights, lamb, self.temperature,
                                                         epoch >= self.config.ignore_graph_epoch, forward_loss)

                # update loss weights 试过 dwa 效果不行，就没再用
                if self.config.use_dwa and loss_last is not None:
                    loss_weights = dwa(loss_last, losses)
                loss_last = losses

                loss.backward()

                # 解决爆显存问题，多批 loss 一回传，实现统一 batch_size
                accumulated_num += x.size(0)
                current_num += x.size(0)
                should_step = accumulated_num >= self.config.accumulate_period
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_num = 0

                # ETA
                eta_s = (datetime.now().timestamp() - self.start_time) / current_num * \
                        (train_iter.N * (self.config.num_epoches - start_epoch) - current_num)
                if eta_s < 60:
                    eta = str(round(eta_s, 2)) + "s"
                elif eta_s < 3600:
                    eta = str(round(eta_s / 60, 2)) + "min"
                else:
                    eta = str(round(eta_s / 3600, 2)) + "h"

                logging.info(f"Training, {i}/{len(train_iter)}, {epoch}/{self.config.num_epoches}, "
                             f"loss: {round(loss.item(), 4)}, "
                             f"lamb: {round(lamb[0][0].item(), 3)} "
                             f"step: {should_step} "
                             f"T: {round(self.temperature, 4)} "
                             f"eta: {eta}")

                if current_num % self.config.show_period == 0:
                    # evaluate
                    ys, residuals, preds = torch.cat(ys, dim=0), torch.cat(residuals, dim=0), torch.cat(preds, dim=0)
                    self.model.eval()

                    train_loss, _ = self.model.calculate_loss(residuals, preds, ys, loss_weights, lamb,
                                                              self.temperature,
                                                              epoch >= self.config.ignore_graph_epoch, 0)
                    l1.append(_[0])
                    l2.append(_[1])
                    l3.append(_[2])
                    l4.append(_[3])

                    # all, inflow, outflow 的三组评估指标。fast_eval 是失败的尝试，没意义
                    mets = self.eval(self.dataset.val_iter, fast_eval=self.config.fast_eval)
                    # met = self.eval(self.dataset.test_iter)
                    val_rmse, val_mape, val_mae = mets[0]

                    # for loss drawing
                    xs.append(current_num)
                    rmses.append(val_rmse)
                    maes.append(val_mae)
                    mapes.append(val_mape)

                    logging.info(f"Ep {epoch}/{self.config.num_epoches}, iter {current_num / self.config.batch_size},"
                                 f" train loss {round(train_loss.item(), 4)},"
                                 f" val mae {round(val_mae, 4)} mape {round(val_mape, 4)} rmse {round(val_rmse, 4)}")

                    v = 80  # random
                    weights = [math.exp(self.model.weights[i][0][0][v][0]) for i in range(self.config.num_subgraphs)]
                    logging.info(f"w: {[round(i / sum(weights), 2) for i in weights]}")

                    visualize_dir = f"logs/{time}/{current_num // self.config.show_period}"

                    if val_mae < self.best_val_loss:
                        self.best_val_loss = val_mae
                        torch.save(self.model.state_dict(), self.save_path)
                        logging.info("Good, saving model.")
                        last_update_val = current_num
                        visualize_dir += "*"

                        if self.config.visualize:
                            ensure_dir(visualize_dir)
                            self.visualize(visualize_dir)

                    ys, residuals, preds = [], [], []
                    self.model.train()

                    self.temperature *= self.config.decay_temperature

                    if current_num - last_update_val > self.config.early_stop_batch:
                        jump_flag = True
                        logging.info("Long time since last update, early stopping.")
                        break
            if jump_flag:
                break

        if not self.config.visualize:
            return

        self.model.print()
        from matplotlib import pyplot as plt
        plt.figure(figsize=(20, 10))
        plt.subplot(241)
        plt.plot(xs, maes, label="MAE")
        plt.legend()
        plt.subplot(242)
        plt.plot(xs, mapes, label="MAPE")
        plt.legend()
        plt.subplot(243)
        plt.plot(xs, rmses, label="RMSE")
        plt.legend()
        plt.subplot(244)
        plt.plot(xs, maes, label="MAE")
        plt.plot(xs, rmses, label="RMSE")
        plt.plot(xs, mapes, label="MAPE")
        plt.legend()
        plt.subplot(245)
        plt.plot(xs, l1, label="L1")
        plt.legend()
        plt.subplot(246)
        plt.plot(xs, l2, label="L2")
        plt.legend()
        plt.subplot(247)
        plt.plot(xs, l3, label="L3")
        plt.legend()
        plt.subplot(248)
        plt.plot(xs, l4, label="L4")
        plt.legend()
        plt.savefig(f"figs/{time}.png")

    def visualize(self, dir_name):
        self.model.visualize(dir_name)

    def eval(self, data_iter, debug=False, fast_eval=-1, tsne=-1):
        def metrics(pred, true):
            return loss.rmse_torch(pred, true, self.config.mae_mask).item(), \
                loss.mape_torch(pred, true, self.config.mae_mask).item(), \
                loss.mae_torch(pred, true, self.config.mae_mask).item()

        self.model.eval()
        logging.info("Evaluating...")
        trues, preds, xs, sub_ps, sub_rs = [], [], [], [], []
        with torch.no_grad():
            # data_iter.shuffle()
            for i, (x, y) in enumerate(data_iter):
                xs.append(x.cpu())
                trues.append(y.cpu())  # NCVL'

                _, pred, sub_preds, sub_residuals, _ = self.model(x, 1e-6, debug, tsne=tsne == i, time=self.time)
                preds.append(pred.cpu())  # NCVL'
                if debug:
                    sub_ps.append(sub_preds.cpu())  # SNCVL
                    sub_rs.append(sub_residuals.cpu())  # SNCVL
                if fast_eval != -1 and i > fast_eval:
                    break
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        xs = torch.cat(xs, dim=0)
        if debug:
            sub_ps = torch.cat(sub_ps, dim=1)  # SNCVL
            sub_rs = torch.cat(sub_rs, dim=1)
            subs_ = sub_ps.permute(0, 1, 3, 4, 2)  # SNVLC
            subs_r = sub_rs.permute(0, 1, 3, 4, 2)
        # NCVL -> NVLC
        xs_ = xs.permute(0, 2, 3, 1)
        preds_ = preds.permute(0, 2, 3, 1)
        trues_ = trues.permute(0, 2, 3, 1)
        # return metrics(preds_, trues_)
        if debug:
            v = self.config.vertex
            K = subs_.shape[0]
            data = []  # [trues_[:, v, 0, 0]]
            for i in range(K):
                subdata = subs_[i, :, v, 0, 0]
                res = []
                for d in subdata:
                    if d >= 0:
                        res.append(d)
                data.append(res)
            import seaborn as sns
            plt.figure(figsize=(5, 1.2))
            ax = sns.violinplot(data=data, edgecolor='black',  # palette=sns.color_palette("pastel", n_colors=K + 1),
                                scale='width', width=0.9)
            # plt.ylim([0, 120])
            plt.xlim([-0.8, 5.8])
            ax.tick_params(axis='y', direction='in', left=True, right=False)
            ax.tick_params(axis='x', which='both', bottom=False, top=False)
            labels = []  # ["GT"]
            labels.extend([f"{i}" for i in range(1, K + 1)])
            plt.xticks(range(K), labels)
            plt.savefig(f"figs/{self.time}_violin.pdf", format='pdf', bbox_inches="tight")

            gr = -1
            gr_mae = -1
            for i in range(self.config.num_nodes):
                _, _, mae = metrics(preds_[:, i, 0, 0], trues_[:, i, 0, 0])
                if mae > gr_mae:
                    gr_mae = mae
                    gr = i
            logging.info(f"Max MAE: {gr}, {gr_mae}")

            subs_ = subs_[:, :, v, 0, 0]  # SN
            trues_ = trues_[:, v, 0, 0]  # N
            preds_ = preds_[:, v, 0, 0]
            ns, B = subs_.shape
            xs, ys, ts, ss = [], [[] for i in range(ns)], [], []
            for i in range(B):
                xs.append(i)
                ts.append(trues_[i])
                ss.append(preds_[i])
                for j in range(ns):
                    ys[j].append(subs_[j][i])
            logging.info(f"Size: {len(xs)}")
            np.savez_compressed(f"{self.time}.npz", xs=xs, ys=ys, ts=ts, ss=ss)
            plt.figure(figsize=(len(xs) // 4, 3))
            plt.plot(xs, ts, label="true")
            plt.plot(xs, ss, label="pred")
            for j in range(ns):
                plt.plot(xs, ys[j], label=f"{j}")
            plt.legend()
            plt.savefig(f"figs/{self.time}_pred.png")
            exit(0)

        ret = [metrics(preds, trues)]
        for i in range(preds_.size(3)):
            ret.append(metrics(preds_[..., i], trues_[..., i]))
        return ret

    def test(self):
        metrics = self.eval(self.dataset.test_iter, self.config.debug, tsne=self.config.tsne)
        test_rmse, test_mape, test_mae = metrics[0]
        logging.info(f"Test: rmse {test_rmse}, mae {test_mae}, mape {test_mape}")
        for i in range(len(metrics) - 1):
            test_rmse, test_mape, test_mae = metrics[i + 1]
            logging.info(f"Test feature {i}: rmse {test_rmse}, mae {test_mae}, mape {test_mape}")
        logging.info(f"Calculation done.")
        logging.info(f"log name: {self.time}")


def dwa(L_old, L_new, T=2):
    L_old = torch.Tensor(L_old)
    L_new = torch.Tensor(L_new)
    N = len(L_old)
    r = L_old / L_new
    w = N * torch.softmax(r / T, dim=0)
    return w.numpy()
