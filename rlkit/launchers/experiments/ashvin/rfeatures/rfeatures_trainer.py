from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
from rlkit.core.loss import LossFunction
from torch import optim
from torchvision.utils import save_image
from multiworld.core.image_env import normalize_image
from rlkit.core import logger
from rlkit.torch import pytorch_util as ptu
from rlkit.util.ml_util import ConstantSchedule
import collections

from rlkit.torch.core import np_to_pytorch_batch

import torchvision
import pickle
from torch.utils import data

import time

import rlkit.data_management.external.epic_kitchens_data as epic
import matplotlib.pyplot as plt

def get_data(variant):
    import numpy as np
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.dataset  import \
        ImageObservationDataset, InitialObservationDataset, EpicTimePredictionDataset

    dataset_name = variant.get("dataset_name")
    full_dataset_path = "/private/home/anair17/ashvindev/railrl/notebooks/outputs/%s_trajectories.p" % dataset_name
    data = pickle.load(open(full_dataset_path, "rb"))
    l = len(data)

    N = l
    test_p = variant.get('test_p', 0.9)

    n = int(N * test_p)

    train_dataset = EpicTimePredictionDataset(data[:n])
    test_dataset = EpicTimePredictionDataset(data[n:N])

    return train_dataset, test_dataset, {}

MAX_BATCH_SIZE = 20


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TimePredictionTrainer(LossFunction):
    def __init__(
            self,
            model,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            lr=1e-3,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            use_linear_dynamics=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
            batch_size=64,
    ):
        #TODO:steven fix pickling
        assert not use_parallel_dataloading, "Have to fix pickling the dataloaders first"

        if skew_config is None:
            skew_config = {}
        self.log_interval = log_interval
        self.beta = beta
        if is_auto_encoder:
            self.beta = 0
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None or is_auto_encoder:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.do_scatterplot = do_scatterplot
        model.to(ptu.device)

        self.model = model

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params,
            lr=self.lr,
            weight_decay=weight_decay,
        )

        self.batch_size = batch_size
        self.use_parallel_dataloading = use_parallel_dataloading
        self.train_data_workers = train_data_workers
        self.skew_dataset = skew_dataset
        self.skew_config = skew_config
        self.start_skew_epoch = start_skew_epoch
        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        self.linearity_weight = linearity_weight
        self.distance_weight = distance_weight
        self.loss_weights = loss_weights

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.log_softmax = torch.nn.LogSoftmax()

        self.use_linear_dynamics = use_linear_dynamics
        self._extra_stats_to_log = None

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)

        self.num_train_batches = 0
        self.num_test_batches = 0

        self.bin_midpoints = (torch.arange(0, self.model.output_classes).float() + 0.5) / self.model.output_classes
        self.bin_midpoints = self.bin_midpoints.to(ptu.device)

    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def get_dataset_stats(self, data):
        torch_input = ptu.from_numpy(normalize_image(data))
        mus, log_vars = self.model.encode(torch_input)
        mus = ptu.get_numpy(mus)
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def _kl_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
        mu, log_var = self.model.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def _reconstruction_squared_error_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
        recons, *_ = self.model(torch_input)
        error = torch_input - recons
        return ptu.get_numpy((error ** 2).sum(dim=1))

    def set_vae(self, vae):
        self.model = vae
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def to_device(self, batch):
        for key in batch:
            batch[key] = batch[key].float().to(ptu.device)

    def train_epoch(self, epoch, dataset, batches=100):
        start_time = time.time()
        start_loop = start_time
        for b in range(batches):
            batch = dataset.random_batch(self.batch_size)
            data_time = time.time()
            self.eval_statistics["train/data_duration"].append(data_time - start_loop)
            self.train_batch(epoch, batch)
            start_loop = time.time()
            self.eval_statistics["train/batch_duration"].append(start_loop - data_time)
        self.eval_statistics["train/epoch_duration"].append(time.time() - start_time)

    def test_epoch(self, epoch, dataset, batches=10):
        start_time = time.time()
        start_loop = start_time
        for b in range(batches):
            batch = dataset.random_batch(self.batch_size)
            data_time = time.time()
            self.eval_statistics["test/data_duration"].append(data_time - start_loop)
            self.test_batch(epoch, batch)
            start_loop = time.time()
            self.eval_statistics["test/batch_duration"].append(start_loop - data_time)
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    # def train_epoch(self, epoch, dataset_loader):
    #     start_loop = time.time()
    #     for batch in dataset_loader:
    #         data_time = time.time()
    #         self.eval_statistics["train/data_duration"].append(data_time - start_loop)
    #         self.to_device(batch)
    #         self.train_batch(epoch, batch)
    #         start_loop = time.time()
    #         self.eval_statistics["train/batch_duration"].append(start_loop - data_time)

    # def test_epoch(self, epoch, dataset_loader):
    #     start_loop = time.time()
    #     for batch in dataset_loader:
    #         data_time = time.time()
    #         self.eval_statistics["test/data_duration"].append(data_time - start_loop)
    #         self.to_device(batch)
    #         self.test_batch(epoch, batch)
    #         start_loop = time.time()
    #         self.eval_statistics["test/batch_duration"].append(start_loop - data_time)

    def compute_loss(self, batch, epoch=-1, test=False, slc=None):
        prefix = "test/" if test else "train/"

        if slc:
            x0, xt, xT = batch["x0"][slc], batch["xt"][slc], batch["xT"][slc]
            real_yt = batch["yt"][slc].flatten().long().to("cpu")
        else:
            x0, xt, xT = batch["x0"], batch["xt"], batch["xT"]
            real_yt = batch["yt"].flatten().long().to("cpu")
        pred_yt = self.model(x0, xt, xT).to("cpu")

        loss = self.loss_fn(pred_yt, real_yt)

        acc1, acc5 = accuracy(pred_yt, real_yt, topk=(1, 5))

        # e = pred_yt - real_yt

        # loss = torch.mean(torch.pow(e, 2))

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "acc1"].append(acc1.item())
        self.eval_statistics[prefix + "acc5"].append(acc5.item())
#         self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
#         self.eval_statistics[prefix + "kles"].append(kle.item())

#         encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
#         z_data = ptu.get_numpy(encoder_mean.cpu())
#         for i in range(len(z_data)):
#             self.eval_data[prefix + "zs"].append(z_data[i, :])
        current_batch = dict(x0=x0, xt=xt, xT=xT, yt=real_yt, )
        self.eval_data[prefix + "last_batch"] = (current_batch, pred_yt)

        return loss

    def train_batch(self, epoch, batch):
        self.model.train()

        bz = batch["x0"].shape[0]

        self.optimizer.zero_grad()
        for i in range(0, bz, MAX_BATCH_SIZE):
            loss = self.compute_loss(batch, epoch, False, slc=slice(i, i + MAX_BATCH_SIZE))
            loss.backward()
        self.optimizer.step()

        self.num_train_batches += 1
        self.eval_statistics['num_train_batches'] = self.num_train_batches

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()

        bz = batch["x0"].shape[0]

        start = time.time()

        for i in range(0, bz, MAX_BATCH_SIZE):
            loss = self.compute_loss(batch, epoch, True, slc=slice(i, i + MAX_BATCH_SIZE))

        end = time.time()
        self.eval_statistics["test/batch_duration"].append(end - start)

        self.num_test_batches += 1
        self.eval_statistics['num_test_batches'] = self.num_test_batches

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats

    def dump_scatterplot(self, z, epoch):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.log(__file__ + ": Unable to load matplotlib. Consider "
                                  "setting do_scatterplot to False")
            return
        dim_and_stds = [(i, np.std(z[:, i])) for i in range(z.shape[1])]
        dim_and_stds = sorted(
            dim_and_stds,
            key=lambda x: x[1]
        )
        dim1 = dim_and_stds[-1][0]
        dim2 = dim_and_stds[-2][0]
        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, dim1], z[:, dim2], marker='o', edgecolor='none')
        if self.model.dist_mu is not None:
            x1 = self.model.dist_mu[dim1:dim1 + 1]
            y1 = self.model.dist_mu[dim2:dim2 + 1]
            x2 = (
                    self.model.dist_mu[dim1:dim1 + 1]
                    + self.model.dist_std[dim1:dim1 + 1]
            )
            y2 = (
                    self.model.dist_mu[dim2:dim2 + 1]
                    + self.model.dist_std[dim2:dim2 + 1]
            )
        plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
        axes = plt.gca()
        axes.set_xlim([-6, 6])
        axes.set_ylim([-6, 6])
        axes.set_title('dim {} vs dim {}'.format(dim1, dim2))
        plt.grid(True)
        save_file = osp.join(self.log_dir, 'scatter%d.png' % epoch)
        plt.savefig(save_file)

    def histogram(self, logits, targets, ):
        n = logits.size(0)
        t = logits.size(1)
        P = torch.nn.functional.softmax(logits, dim=1)

        I = torch.ones(n, 3, epic.CROP_HEIGHT, epic.CROP_WIDTH)

        bin_width = epic.CROP_WIDTH // t

        I[:, :, :, bin_width*t:, ] = 0

        for i in range(n):
            x = int(targets[i] * bin_width)
            I[i, :2, :, x:x+bin_width, ] = 0
            for j in range(t):
                x = j * bin_width
                p = int(P[i, j] * epic.CROP_HEIGHT)
                y = epic.CROP_HEIGHT - p
                I[i, 1:, y:epic.CROP_HEIGHT, x:x+bin_width, ] = 0
        return I

    def dump_reconstructions(self, epoch):
        for prefix in ["train", "test"]:
            batch, pred_yt = self.eval_data["%s/last_batch" % prefix]
            x0, xt, xT, yt = batch["x0"], batch["xt"], batch["xT"], batch["yt"]
            n = min(x0.size(0), 8)
            x0 = x0[:n].cpu().view(-1, 3, epic.CROP_HEIGHT, epic.CROP_WIDTH) # .view(n, 3, 240, 240)[:, :, :224, :224]
            xt = xt[:n].cpu().view(-1, 3, epic.CROP_HEIGHT, epic.CROP_WIDTH) # .view(n, 3, 240, 240)[:, :, :224, :224]
            xT = xT[:n].cpu().view(-1, 3, epic.CROP_HEIGHT, epic.CROP_WIDTH) # .view(n, 3, 240, 240)[:, :, :224, :224]
            yViz = self.histogram(pred_yt[:n], yt[:n])
            comparison = torch.cat([
                x0, xt, xT, yViz
            ])
            save_dir = osp.join(self.log_dir, '%s_img%d.png' % (prefix, epoch))
            save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_trajectory_rewards(self, epoch, datasets, visualize=False):
        self.model.eval()

        for prefix in ["train", "test"]:
            dataset = datasets[prefix]

            i = np.random.randint(len(dataset))
            row = dataset[i]
            id = int(row[0])

            batch = epic.get_clip_as_batch(id, max_frames=50)
            batch = batch.to(ptu.device)

            N = len(batch)

            # latents = []
            # MAX_BATCH_SIZE = 50
            # for i in range(0, N, MAX_BATCH_SIZE):
            #     z = self.model.encoder(batch[i:i+MAX_BATCH_SIZE, :, :, :])
            #     latents.append(z)
            # latents = torch.cat(latents)
            latents = self.model.encoder(batch)

            ### Feature distances

            distances = self.get_feature_distances(latents)
            if visualize:
                plt.figure()
                plt.plot(distances)
                savefile = osp.join(self.log_dir, "traj_%s%s_id%d.png" % (prefix, str(epoch), id))
                plt.savefig(savefile)

            metric = self.get_diagonal_distance(distances, - distances[0] / N, distances[0])
            normalized_metric = metric / distances[0]
            self.eval_statistics["%s/diagonal_metric" % prefix] = metric
            self.eval_statistics["%s/normalized_diagonal_metric" % prefix] = normalized_metric

            z0 = latents[0, :].repeat(N, 1)
            zt = latents
            zT = latents[-1, :].repeat(N, 1)

            dt = zt - z0
            dT = zT - z0

            ### Expectation of classication

            z = torch.cat([dt, dT], dim=1).detach()
            # preds = []
            # for i in range(0, N, MAX_BATCH_SIZE):
            #     pred = self.model.predictor(z[i:i+MAX_BATCH_SIZE, :, ])
            #     preds.append(pred)
            # pred_yt = torch.cat(preds)
            pred_yt = self.model.predictor(z)

            P = torch.nn.functional.softmax(pred_yt, dim=1)
            ET = torch.sum(P * self.bin_midpoints, dim=1)

            classification_distances = ptu.get_numpy(ET.cpu())
            classification_diagonal_metric = self.get_diagonal_distance(classification_distances, 1 / N,  0)
            self.eval_statistics["%s/classification_diagonal_metric" % prefix] = classification_diagonal_metric

            if visualize:
                plt.figure()
                plt.plot(classification_distances)
                savefile = osp.join(self.log_dir, "traj_%s%s_id%d_classification.png" % (prefix, str(epoch), id))
                plt.savefig(savefile)

            ### Expectation of regression

            regression_pred_yt = (dt * dT).sum(dim=1) / ((dT ** 2).sum(dim=1) + eps)
            regression_distances = ptu.get_numpy(regression_pred_yt.cpu())
            regression_diagonal_metric = self.get_diagonal_distance(regression_distances, 1 / N,  0)
            self.eval_statistics["%s/regression_diagonal_metric" % prefix] = regression_diagonal_metric

            if visualize:
                plt.figure()
                plt.plot(regression_distances)
                savefile = osp.join(self.log_dir, "traj_%s%s_id%d_regression.png" % (prefix, str(epoch), id))
                plt.savefig(savefile)

    def get_diagonal_distance(self, signal, m, b):
        """Computes average distance between signal and y = mx + b"""
        diagonal = - signal[0] / len(signal) * np.arange(len(signal)) + signal[0]
        diagonal = m * np.arange(len(signal)) + b
        metric = np.mean(np.abs(signal - diagonal))
        return metric

    def get_feature_distances(self, z):
        z = ptu.get_numpy(z.cpu())

        z_goal = z[-1, :]
        distances = []
        for t in range(len(z)):
            d = np.linalg.norm(z[t, :] - z_goal)
            distances.append(d)
        return distances

    # def viz_rewards(self, prefix, id):
    #     self.model.eval()
    #     filename = osp.join(self.log_dir, '%s_id%d.png' % (prefix, id))
    #     distances = epic.viz_rewards(self.model, id, filename)
    #     return distances


class LatentPathPredictorTrainer(TimePredictionTrainer):
    def compute_loss(self, batch, epoch=-1, test=False, slc=None):
        prefix = "test/" if test else "train/"

        if slc:
            x0, xt, xT = batch["x0"][slc], batch["xt"][slc], batch["xT"][slc]
            real_yt = batch["yt"][slc].flatten().long() # .to("cpu")
            target_yt = batch["yt"][slc]
        else:
            x0, xt, xT = batch["x0"], batch["xt"], batch["xT"]
            real_yt = batch["yt"].flatten().long().to("cpu")
            target_yt = batch["yt"]

        z0, zt, zT = self.model.get_latents(x0, xt, xT)

        z0 = z0.detach()
        zT = zT.detach()

        z = torch.cat([z0, zt, zT], dim=1).detach()
        pred_yt = self.model.predictor(z)

        batch_size = len(real_yt)
        target_z = z0 + (zT - z0) * target_yt.view(batch_size, 1) / 100
        error = target_z - zt
        mse_loss = torch.mean(torch.pow(error, 2))
        bce_loss = self.loss_fn(pred_yt, real_yt)

        loss = bce_loss + mse_loss

        acc1, acc5 = accuracy(pred_yt, real_yt, topk=(1, 5))

        # e = pred_yt - real_yt

        # loss = torch.mean(torch.pow(e, 2))

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "mse_loss"].append(mse_loss.item())
        self.eval_statistics[prefix + "bce_loss"].append(bce_loss.item())
        self.eval_statistics[prefix + "acc1"].append(acc1.item())
        self.eval_statistics[prefix + "acc5"].append(acc5.item())
#         self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
#         self.eval_statistics[prefix + "kles"].append(kle.item())

#         encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
#         z_data = ptu.get_numpy(encoder_mean.cpu())
#         for i in range(len(z_data)):
#             self.eval_data[prefix + "zs"].append(z_data[i, :])
        current_batch = dict(x0=x0, xt=xt, xT=xT, yt=real_yt, )
        self.eval_data[prefix + "last_batch"] = (current_batch, pred_yt)

        return loss

eps = 1e-5

class GeometricTimePredictorTrainer(TimePredictionTrainer):
    def compute_loss(self, batch, epoch=-1, test=False, slc=None):
        prefix = "test/" if test else "train/"

        if slc:
            x0, xt, xT = batch["x0"][slc], batch["xt"][slc], batch["xT"][slc]
            real_yt = batch["yt"][slc].flatten().long() # .to("cpu")
            target_yt = batch["yt"][slc]
        else:
            x0, xt, xT = batch["x0"], batch["xt"], batch["xT"]
            real_yt = batch["yt"].flatten().long().to("cpu")
            target_yt = batch["yt"]

        z0, zt, zT = self.model.get_latents(x0, xt, xT)

        dt = zt - z0
        dT = zT - z0

        regression_pred_yt = (dt * dT).sum(dim=1) / ((dT ** 2).sum(dim=1) + eps)

        e = regression_pred_yt - target_yt / self.model.output_classes

        regression_t_difference = torch.mean(torch.abs(e))
        regression_t_mean = torch.mean(regression_pred_yt)

        mse_loss = torch.mean(torch.pow(e, 2))

        # import ipdb; ipdb.set_trace()

        # z0 = z0.detach()
        # zT = zT.detach()

        if self.model.delta_features:
            dt = zt - z0
            dT = zT - z0
            z = torch.cat([dt, dT], dim=1)
        else:
            z = torch.cat([z0, zt, zT], dim=1)

        if not self.loss_weights["classification_gradients"]:
            z = z.detach()
        pred_yt = self.model.predictor(z)

        P = torch.nn.functional.softmax(pred_yt, dim=1)
        ET = torch.sum(P * self.bin_midpoints, dim=1)

        classification_t_difference = torch.mean(torch.abs(ET - target_yt / self.model.output_classes))
        classification_t_mean = torch.mean(ET)

        # batch_size = len(real_yt)
        # target_z = z0 + (zT - z0) * target_yt.view(batch_size, 1) / 100
        # error = target_z - zt
        # mse_loss = torch.mean(torch.pow(error, 2))

        bce_loss = self.loss_fn(pred_yt, real_yt)

        loss = bce_loss + self.loss_weights["mse"] * mse_loss

        acc1, acc5 = accuracy(pred_yt, real_yt, topk=(1, 5))

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "mse_loss"].append(mse_loss.item())
        self.eval_statistics[prefix + "bce_loss"].append(bce_loss.item())
        self.eval_statistics[prefix + "acc1"].append(acc1.item())
        self.eval_statistics[prefix + "acc5"].append(acc5.item())
        self.eval_statistics[prefix + "regression_t_difference"].append(regression_t_difference.item())
        self.eval_statistics[prefix + "classification_t_difference"].append(classification_t_difference.item())
        self.eval_statistics[prefix + "regression_t_mean"].append(regression_t_mean.item())
        self.eval_statistics[prefix + "classification_t_mean"].append(classification_t_mean.item())
#         self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
#         self.eval_statistics[prefix + "kles"].append(kle.item())

#         encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
#         z_data = ptu.get_numpy(encoder_mean.cpu())
#         for i in range(len(z_data)):
#             self.eval_data[prefix + "zs"].append(z_data[i, :])
        current_batch = dict(x0=x0, xt=xt, xT=xT, yt=real_yt, )
        self.eval_data[prefix + "last_batch"] = (current_batch, pred_yt)

        return loss
