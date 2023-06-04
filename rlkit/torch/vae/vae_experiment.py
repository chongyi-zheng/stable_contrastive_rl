class VAEExperiment:
    def __init__(self, vae_trainer, num_epochs, save_period=1,
                 dump_skew_debug_plots=False):
        self.vae_trainer = vae_trainer
        self.num_epochs = num_epochs
        self.save_period = save_period
        self.dump_skew_debug_plots = dump_skew_debug_plots
        self.epoch = 0

    def _train(self):
        log = dict()
        done = False
        if self.epoch == self.num_epochs:
            done = True
            return log, done
        should_save_imgs = (self.epoch % self.save_period == 0)
        self.vae_trainer.train_epoch(self.epoch)
        self.vae_trainer.test_epoch(self.epoch,
                                    save_reconstruction=should_save_imgs,
                                    save_scatterplot=should_save_imgs)
        if should_save_imgs:
            self.vae_trainer.dump_samples(self.epoch)
            if self.dump_skew_debug_plots:
                self.vae_trainer.dump_best_reconstruction(self.epoch)
                self.vae_trainer.dump_worst_reconstruction(self.epoch)
                self.vae_trainer.dump_sampling_histogram(self.epoch)
        self.vae_trainer.update_train_weights()
        self.epoch += 1
        return log, done

    def to(self, device):
        self.vae_trainer.model.to(device)
