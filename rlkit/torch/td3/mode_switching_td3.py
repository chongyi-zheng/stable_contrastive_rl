from rlkit.torch.td3.td3 import TD3

class ModeSwitchingTd3(TD3):
    def __init__(
            self,
            train_mode,
            test_mode,
            *args,
            **kwargs
    ):
        self.train_mode = train_mode
        self.test_mode = test_mode
        super().__init__(*args, **kwargs)
        # assert isinstance(self.replay_buffer, HerReplayBuffer)

    def _start_epoch(self, epoch):
        self.env.mode(self.train_mode)
        super()._start_epoch(epoch)

    def _try_to_eval(self, epoch):
        self.env.mode(self.test_mode)
        super()._start_epoch(epoch)
