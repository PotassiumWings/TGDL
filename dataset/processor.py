import logging
import os
import random

import numpy as np
import torch
from fastdtw import fastdtw

from configs.arguments import TrainingArguments
from utils.normalize import get_scaler


class AbstractDataset(object):
    def __init__(self, config: TrainingArguments):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_iter, self.val_iter, self.test_iter = None, None, None
        self.supports = None
        self.scaler = None


class MyDataset(AbstractDataset):
    def __init__(self, config: TrainingArguments):
        super().__init__(config)
        self.data_dir = os.path.join("data", self.config.dataset_name)
        self.batch_size = config.batch_size

        self._load_data()
        self._load_adj()

    def _load_adj(self):
        adj_mx_filename = os.path.join(self.data_dir, "adj_mx.npz")
        if not os.path.exists(adj_mx_filename):
            num_node = self.config.num_nodes
            data = self.train_data  # NLVC
            data_mean = data.mean(axis=3)
            data_mean = data_mean.mean(axis=0)
            # if data_mean.shape[0] > 500:
            #     data_mean = data_mean[:500, :]
            data_mean = data_mean.squeeze().T
            logging.info(f"Data mean shape: ({data_mean.shape})")
            dtw_distance = np.zeros((num_node, num_node))
            for i in range(num_node):
                for j in range(i, num_node):
                    dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
                    # logging.info(f"{i} {j}: {dtw_distance[i][j]}")
            for i in range(num_node):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]

            mean = np.mean(dtw_distance)
            std = np.std(dtw_distance)
            dist_matrix = (dtw_distance - mean) / std
            sigma = self.config.sigma
            dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
            dtw_matrix = np.zeros_like(dist_matrix)
            dtw_matrix[dist_matrix > self.config.thres] = 1
            np.savez_compressed(os.path.join(self.data_dir, "adj_mx.npz"), adj_mx=dtw_matrix)
        self.supports = [np.load(adj_mx_filename)["adj_mx"]]

    def _load_data(self):
        train_filename = os.path.join(self.data_dir, "train.npz")
        val_filename = os.path.join(self.data_dir, "val.npz")
        test_filename = os.path.join(self.data_dir, "test.npz")
        self.train_iter, self.train_data = self._get_iter(train_filename, define_scaler=True)
        self.val_iter, _ = self._get_iter(val_filename)
        self.test_iter, _ = self._get_iter(test_filename, shuffle=False)

    def _get_iter(self, filename, define_scaler=False, shuffle=True):
        data = np.load(filename)
        x, y = data['x'], data['y']
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        y = y[:, :, :, :self.config.c_out]
        if define_scaler:
            self.scaler = get_scaler(self.config.scaler, x)
        x = self.scaler.transform(x)
        return MyDatasetIterator((x, y), self.config.batch_size, self.device, shuffle), x


class MyDatasetIterator(object):
    def __init__(self, data, batch_size, device, shuffle):
        self.batch_size = batch_size
        # x: (N, L, V, C)
        # y: (N, L', V, C)
        self.x: torch.Tensor = data[0]
        self.y: torch.Tensor = data[1]
        self.N = self.x.size(0)

        self.n_batches = int(round(self.N // batch_size))
        self.device = device

        self.batches = list(np.arange(0, self.N))
        if shuffle:
            self.shuffle()

        self.index = 0

    def shuffle(self):
        random.shuffle(self.batches)
        self.index = 0

    def _to_tensor(self, indexes):
        x = torch.FloatTensor([self.x[i].numpy() for i in indexes]).to(self.device)  # NLVC
        y = torch.FloatTensor([self.y[i].numpy() for i in indexes]).to(self.device)  # NL'VC
        x = x.permute(0, 3, 2, 1)  # NCVL
        y = y.permute(0, 3, 2, 1)  # NCVL'
        return x, y

    def __next__(self):
        if self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                      self.index * self.batch_size: min((self.index + 1) * self.batch_size, len(self.batches))]
            self.index += 1
            x, y = self._to_tensor(batches)
            return x, y

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches
