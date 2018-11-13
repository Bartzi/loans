from itertools import zip_longest

import os
import shutil

from chainer.training.extensions import LogReport


class Logger(LogReport):

    def __init__(self, model_files, log_dir, tensorboard_writer=None, keys=None, trigger=(1, 'epoch'), postprocess=None, log_name='log', dest_file_names=None):
        super(Logger, self).__init__(keys=keys, trigger=trigger, postprocess=postprocess, log_name=log_name)
        self.tensorboard_writer = tensorboard_writer
        self.backup_model(model_files, log_dir, dest_file_names)

    def backup_model(self, model_files, log_dir, dest_file_names):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        if dest_file_names is None:
            for model_file in model_files:
                shutil.copy(model_file, log_dir)
        else:
            assert len(model_files) == len(dest_file_names), "You did not specify enough file names for the files you want to create a backup of"
            for model_file, dest_file_name in zip(model_files, dest_file_names):
                shutil.copy(model_file, os.path.join(log_dir, dest_file_name))

    def __call__(self, trainer):
        observation = trainer.observation
        observation_keys = observation.keys()
        if self._keys is not None:
            observation_keys = filter(lambda x: x in self._keys, observation_keys)

        if self.tensorboard_writer is not None:
            for key in observation_keys:
                self.tensorboard_writer.add_scalar(key, observation[key].data, trainer.updater.iteration)

        super().__call__(trainer)
