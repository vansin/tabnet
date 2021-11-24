# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class UploadHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=1):
        self.interval = interval

    def before_run(self, runner):
        print('upload before_run')
        pass

    def after_run(self, runner):
        print('upload after_run')
        pass

    def before_epoch(self, runner):
        print('upload before_epoch')
        pass

    def after_epoch(self, runner):
        print('upload after_epoch')
        pass

    def before_iter(self, runner):
        print('upload before_iter')
        pass

    def after_iter(self, runner):
        print('upload after_iter')
        pass

    def before_train_epoch(self, runner):
        print('upload before_train_epoch')

    def before_val_epoch(self, runner):
        print('upload before_val_epoch')

    def after_train_epoch(self, runner):
        print('upload after_train_epoch')

    def after_val_epoch(self, runner):
        print('upload after_val_epoch')

    def before_train_iter(self, runner):
        print('upload before_train_iter')

    def before_val_iter(self, runner):
        print('upload before_val_iter')

    def after_train_iter(self, runner):

        print('upload loss after_train_iter')
        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')

    def after_val_iter(self, runner):
        print('upload after_val_iter')
