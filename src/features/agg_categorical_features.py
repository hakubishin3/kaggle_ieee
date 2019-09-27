import gc
import sys
import time
import pandas as pd
import datetime
import numpy as np
from contextlib import contextmanager
from base import Feature
from joblib import Parallel, delayed

sys.path.append("../")
from utils.logger import get_logger
from utils.read_data import read_preprocessing_data


# ===============
# Constants
# ===============
DATA_DIR = "../../data/input/"
FE_DIR = "../../data/features/"


# ===============
# Settings
# ===============
logger = get_logger()
# logger = get_logger(out_file="label_encoding.log")

var_list = [
    'DeviceType',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_20',
    'D9'
]

groupby_dict = [
    {
        'key': ['card2'],
        'var': var_list
    },
    {
        'key': ['addr1'],
        'var': var_list
    },
    {
        'key': ['P_emaildomain'],
        'var': var_list
    },
    {
        'key': ['card5'],
        'var': var_list
    },
    {
        'key': ['addr1', 'Registered_at'],
        'var': var_list
    }
]



# ===============
# Functions
# ===============
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


def _calc_agg_category_func(df, key, value):
    new_fe_name_sample = f'{value}_groupby_{"_".join(key)}_n_sample'
    new_fe_name_ratio = f'{value}_groupby_{"_".join(key)}_ratio'
    tmp = df.groupby(key+[value]).size().rename(new_fe_name_sample).reset_index()
    tmp2 = tmp.groupby(key).size().rename("n_total").reset_index()
    result = pd.merge(tmp, tmp2, on=key, how="left")
    result[new_fe_name_ratio] = result["n_total"] / result[new_fe_name_sample]

    # concat
    df = pd.merge(df, result, on=key+[value], how="left")

    # return df[[new_fe_name_sample, new_fe_name_ratio]]
    return df[new_fe_name_ratio]


def calc_agg_category_func(df, groupby_dict):
    arg_list = []
    for grp in groupby_dict:
        key, value_list = grp['key'], grp['var']
        for value in value_list:
            arg_list.append([key, value])
            # _calc_rolling_func(df, key, value)

    result_list = Parallel(n_jobs=-1)([delayed(_calc_agg_category_func)(df[arg[0] + [arg[1]]], arg[0], arg[1]) for arg in arg_list])
    result = pd.concat(result_list, axis=1)
    return result


# ===============
# Main class
# ===============
class Agg_Category(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        with timer("load data"):
            train = read_preprocessing_data(DATA_DIR, "train", write_mode=False)
            test = read_preprocessing_data(DATA_DIR, "test", write_mode=False)

        with timer("get predicted user id"):
            predicted_user = pd.read_csv('../../data/interim/20190901_user_ids_share.csv')
            train = pd.merge(train, predicted_user[['TransactionID', 'predicted_user_id']], how='left', on='TransactionID')
            test = pd.merge(test, predicted_user[['TransactionID', 'predicted_user_id']], how='left', on='TransactionID')
            total = train.append(test).reset_index(drop=True)

        with timer("get original cols"):
            org_cols = total.columns

        with timer("aggregate categorical features"):
            total = calc_agg_category_func(total, groupby_dict)
            new_cols = [c for c in total.columns if c not in org_cols]
            total = total[new_cols]
            logger.info(f"n_features: {len(new_cols)}")

            train_result = total.iloc[:len(train)].reset_index(drop=True)
            test_result = total.iloc[len(train):].reset_index(drop=True)
            self.train_feature = train_result
            self.test_feature = test_result

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Agg_Category(FE_DIR)
    f.run().save()
