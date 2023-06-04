import ray.tune as tune
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.pytorch_util import set_gpu_mode
import os.path as osp
import pickle as cloudpickle
import logging


class SequentialRayExperiment(tune.Trainable):

    def _setup(self, config):
        """
        config contains:
            gpu_id (int) -- (default 0)
            use_gpu (bool) --
            init_algo_algo_functions_and_log_fnames ((function, str)[]) -- each
                element of this list is a tuple of a function that returns the
                next algorithm to train and the corresponding log filename.
            init_algo_kwargs (dict) -- the variant to pass into the
                init_algo_function call. This dict is the same for all
                init_algo_function calls. This also means that modification
                to the variant will propagate to future init_algo_function calls.
        """
        gpu_id = config.get('gpu_id', 0)
        use_gpu = config['use_gpu']
        set_gpu_mode(use_gpu, gpu_id)
        logging.info('Using GPU mode={}'.format(use_gpu))
        # import torch
        # if 'cpu' in config['resources_per_trial']:
            # num_threads = config['resources_per_trial']['cpu']
            # torch.set_num_threads(num_threads)
            # logging.info('Setting {} CPU threads'.format(num_threads))

        self.init_algo_functions_and_log_fnames = config['init_algo_functions_and_log_fnames']
        self.init_algo_functions = [
            init_func for init_func, _ in self.init_algo_functions_and_log_fnames
        ]
        self.log_fnames = [
            log_fname for _, log_fname in self.init_algo_functions_and_log_fnames
        ]
        self.init_algo_kwargs = config['algo_variant']
        self.cur_algo = None
        self.cur_algo_idx = -1
        self._setup_next_algo()


    def _train(self):
        self.cur_algo._begin_epoch()
        log_dict, done = self.cur_algo._train()
        self.cur_algo._end_epoch()
        log_dict['global_done'] = False
        log_dict['log_fname'] = self.log_fname
        if done:
            try:
                self._setup_next_algo()
            except StopIteration:
                log_dict['global_done'] = True
        # Convert from numpy objects to native python types. For example,
        # numpy.float32 to python float. This is necessary for the JSON logging
        for key, val in log_dict.items():
            if hasattr(val, 'item'):
                log_dict[key] = val.item()
        return log_dict

    def _setup_next_algo(self):
        """
        If we are getting the algorithm for the first experiment we basically
        just call
            cur_algo = algo1(init_algo_kwargs)
        otherwise, the previous algorithm will be passed in to the algo function
        as well as the first argument
            cur_algo = algo2(algo1, init_algo_kwargs)
        """
        self.cur_algo_idx += 1
        if self.cur_algo_idx >= len(self.init_algo_functions):
            raise StopIteration
        init_algo_function = self.init_algo_functions[self.cur_algo_idx]
        self.log_fname = self.log_fnames[self.cur_algo_idx]
        if self.cur_algo is None:
            self.cur_algo = init_algo_function(self.init_algo_kwargs)
        else:
            self.cur_algo = init_algo_function(self.cur_algo,
                                               self.init_algo_kwargs)
        self.cur_algo.to(ptu.device)

    def _save(self, checkpoint_dir):
        algo_path = osp.join(checkpoint_dir, 'exp.pkl')
        info_to_save = {
            'cur_algo':     self.cur_algo,
            'cur_algo_idx': self.cur_algo_idx
        }
        with open(algo_path, 'wb') as f:
            cloudpickle.dump(info_to_save, f)
        return algo_path

    def _restore(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            info = cloudpickle.load(f)
            self.cur_algo = info['cur_algo']
            self.cur_algo_idx = info['cur_algo_idx']
