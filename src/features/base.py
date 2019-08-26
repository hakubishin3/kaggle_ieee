import time
import inspect
import argparse
import pandas as pd
from pathlib import Path
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''

    def __init__(self, path='.'):
        self.name = self.__class__.__name__
        self.train_feature = pd.DataFrame()
        self.test_feature = pd.DataFrame()
        self.train_path = Path(path) / f'{self.name}_train.ftr'
        self.test_path = Path(path) / f'{self.name}_test.ftr'

    def run(self):
        """
        with timer(self.name):
        """
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else self.name + '_'
        suffix = '_' + self.suffix if self.suffix else ''
        self.train_feature.columns = prefix + self.train_feature.columns + suffix
        self.test_feature.columns = prefix + self.test_feature.columns + suffix
        return self

    @abstractmethod
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def categorical_features():
        raise NotImplementedError

    def save(self):
        self.train_feature.to_feather(str(self.train_path))
        self.test_feature.to_feather(str(self.test_path))


def load_features(config, debug_mode: bool):
    feathre_path = config['dataset']['feature_directory']

    dfs = [pd.read_feather(f'{feathre_path}/{f}_train.ftr', nthreads=-1) for f in config['features']]
    x_train = pd.concat(dfs, axis=1)

    if debug_mode:
        x_test = None
    else:
        dfs = [pd.read_feather(f'{feathre_path}/{f}_test.ftr', nthreads=-1) for f in config['features']]
        x_test = pd.concat(dfs, axis=1)

    return x_train, x_test
