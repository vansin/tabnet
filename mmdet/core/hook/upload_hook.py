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

    def after_train_iter(self, runner):

        print('upload loss after_train_iter')

        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')

    def before_run(self, runner):

        print('before_run after_train_iter')
