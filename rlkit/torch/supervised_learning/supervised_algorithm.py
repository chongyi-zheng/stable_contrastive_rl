from __future__ import print_function
from torch import nn, optim
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger
import numpy as np

class SupervisedAlgorithm():
    def __init__(
            self,
            X_train,
            X_test,
            y_train,
            y_test,
            model,
            batch_size=128,
            lr=3e-4,
            weight_decay=0,
            num_batches = 128,
    ):
        self.batch_size = batch_size
        if ptu.gpu_enabled():
            model.to(ptu.device)
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.num_batches = num_batches

    def random_batch(self, inputs, labels, batch_size=64):
        idxs = np.random.choice(len(inputs), batch_size)
        return inputs[idxs], labels[idxs]

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        per_dim_losses = np.zeros((self.num_batches, self.y_train.shape[1]))
        for batch in range(self.num_batches):
            inputs_np, labels_np = self.random_batch(self.X_train, self.y_train, batch_size=self.batch_size)
            inputs, labels = ptu.Variable(ptu.from_numpy(inputs_np)), ptu.Variable(ptu.from_numpy(labels_np))
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data[0])
            per_dim_loss = np.mean(np.power(ptu.get_numpy(outputs-labels), 2), axis=0)
            per_dim_losses[batch] = per_dim_loss

        logger.record_tabular("train/epoch", epoch)
        logger.record_tabular("train/loss", np.mean(np.array(losses)))
        for i in range(self.y_train.shape[1]):
            logger.record_tabular("train/dim "+str(i)+" loss", np.mean(per_dim_losses[:, i]))


    def test_epoch(
            self,
            epoch,
    ):
        self.model.eval()
        val_losses = []
        per_dim_losses = np.zeros((self.num_batches, self.y_train.shape[1]))
        for batch in range(self.num_batches):
            inputs_np, labels_np = self.random_batch(self.X_test, self.y_test, batch_size=self.batch_size)
            inputs, labels = ptu.Variable(ptu.from_numpy(inputs_np)), ptu.Variable(ptu.from_numpy(labels_np))
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            val_losses.append(loss.data[0])
            per_dim_loss = np.mean(np.power(ptu.get_numpy(outputs - labels), 2), axis=0)
            per_dim_losses[batch] = per_dim_loss

        logger.record_tabular("test/epoch", epoch)
        logger.record_tabular("test/loss", np.mean(np.array(val_losses)))
        for i in range(self.y_train.shape[1]):
            logger.record_tabular("test/dim "+str(i)+" loss", np.mean(per_dim_losses[:, i]))
        logger.dump_tabular()