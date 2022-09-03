from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EpochSetHook(Hook):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_train_epoch(self, runner):
        # CBGSDataset-->NuScenesDataset
        runner.data_loader.dataset.dataset.set_epoch_for_object_sample(runner._epoch)
