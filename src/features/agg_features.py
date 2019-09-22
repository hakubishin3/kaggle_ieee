import sys
import time
import datetime
import itertools
import numpy as np
import pandas as pd
from contextlib import contextmanager
from base import Feature

sys.path.append("../")
from utils.logger import get_logger
from utils.read_data import read_preprocessing_data
from utils.feature_module import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer, SinCos


# ===============
# Constants
# ===============
DATA_DIR = "../../data/input/"
FE_DIR = "../../data/features/"


# ===============
# Settings
# ===============
logger = get_logger()
var_list = ["TransactionAmt", "V258", "V257", "V201", "TransactionAmt_decimal", "D11", "V75_94_mean", "V95_137_mean", "V167_216_mean", "V242_263_mean"]
stats_list = ['mean', 'std', 'min', 'max']
stats_diff_list = ['mean', 'min', 'max']
groupby_dict = [
    {
        'key': ['predicted_user_id'],
        'var': var_list,
        'agg': stats_list
    },
    {
        'key': ['card1'],
        'var': var_list,
        'agg': stats_list
    },
    {
        'key': ['card2'],
        'var': var_list,
        'agg': stats_list
    },
    {
        'key': ['card4'],
        'var': var_list,
        'agg': stats_list
    },
    {
        'key': ['addr1', 'addr2'],
        'var': var_list,
        'agg': stats_list
    },
    {
        'key': ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'ProductCD', 'Registered_at'],
        'var': var_list,
        'agg': stats_list
    },
    {
        'key': ['predicted_user_id'],
        'var': ['D9', 'D9_sin', 'D9_cos', 'D9_LocalTime', 'D9_LocalTime_sin', 'D9_LocalTime_cos'],
        'agg': stats_list
    }
]
diff_dict = [
    {
        'key': ['predicted_user_id'],
        'var': var_list,
        'agg': stats_diff_list
    },
    {
        'key': ['card1'],
        'var': var_list,
        'agg': stats_diff_list
    },
    {
        'key': ['card2'],
        'var': var_list,
        'agg': stats_diff_list
    },
    {
        'key': ['card4'],
        'var': var_list,
        'agg': stats_diff_list
    },
    {
        'key': ['addr1', 'addr2'],
        'var': var_list,
        'agg': stats_diff_list
    },
    {
        'key': ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'ProductCD', 'Registered_at'],
        'var': var_list,
        'agg': stats_diff_list
    },
    {
        'key': ['predicted_user_id'],
        'var': ['D9', 'D9_sin', 'D9_cos', 'D9_LocalTime', 'D9_LocalTime_sin', 'D9_LocalTime_cos'],
        'agg': stats_diff_list
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


# ===============
# Main class
# ===============
class Agg(Feature):
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

            total['TransactionAmt_decimal'] = ((total['TransactionAmt'] - total['TransactionAmt'].astype(int)) * 1000).astype(int)

        with timer("make V features"):
            ### V75 ~~~ 94
            cols_V75_94 = [f'V{no}' for no in range(75, 95, 1)]
            cols_other = [f'V{no}' for no in [75, 88, 89, 90, 91, 94, 100, 104, 105, 106]]
            cols_V75_94 = list(set(cols_V75_94) - set(cols_other))
            total['V75_94_mean'] = total[cols_V75_94].mean(axis=1)

            ### V95 ~~~ 137
            cols_V95_137 = [f'V{no}' for no in range(95, 138, 1)]
            cols_V95_137 = list(set(cols_V95_137) - set([f'V{no}' for no in range(130, 138, 1)]))
            cols_other = [f'V{no}' for no in [96, 98, 99, 100, 104, 105, 106 , 120, 121, 122, 126, 127, 128]]
            cols_other_2 = [f'V{no}' for no in [117, 118, 119]]
            cols_V95_137 = sorted(list(set(cols_V95_137) - set(cols_other) -set(cols_other_2)))
            total['V95_137_mean'] = total[cols_V95_137].mean(axis=1)

            ### V167 ~~~ 216
            cols_V167_216 = [f'V{no}' for no in range(167, 217, 1)]
            cols_other = [f'V{no}' for no in range(186, 202, 1)]
            no_use_cols = [f'V{no}' for no in [169, 172, 173, 174, 175] + list(range(202, 217, 1))]
            cols_V167_216 = sorted(list(set(cols_V167_216) - set(cols_other) -set(no_use_cols)))
            total['V167_216_mean'] = total[cols_V167_216].mean(axis=1)

            ### V242 ~~~ 263
            cols_V242_263 = [f'V{no}' for no in list(range(242, 250, 1)) + list(range(252, 255, 1)) + list(range(257, 263, 1))]
            total['V242_263_mean'] = total[cols_V242_263].mean(axis=1)

            org_cols = total.columns

        with timer("sin/cos transformation"):
            total["D9_sin"] = np.sin(2 * np.pi * total["D9"] / 24).round(4)
            total["D9_cos"] = np.cos(2 * np.pi * total["D9"] / 24).round(4)
            total["D9_LocalTime_sin"] = np.sin(2 * np.pi * total["D9_LocalTime"] / 24).round(4)
            total["D9_LocalTime_cos"] = np.cos(2 * np.pi * total["D9_LocalTime"] / 24).round(4)

        with timer("group by features"):
            groupby = GroupbyTransformer(param_dict=groupby_dict)
            total = groupby.transform(total)
            diff = DiffGroupbyTransformer(param_dict=diff_dict)
            total = diff.transform(total)
            ratio = RatioGroupbyTransformer(param_dict=diff_dict)
            total = ratio.transform(total)

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
    f = Agg(FE_DIR)
    f.run().save()
