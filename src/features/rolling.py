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

stats_list = [np.nanmean, np.nanstd, np.nansum]
var_list = ["V258", "V257", "V201", "D11", "V75_94_mean", "V95_137_mean", "V167_216_mean", "V242_263_mean"]
window_size_list = ["1D", "7D", "30D"]

groupby_dict = [
    {
        'key': ['card2'],
        'var': var_list,
        'agg': stats_list,
        'window_size': window_size_list
    },
    {
        'key': ['addr1'],
        'var': var_list,
        'agg': stats_list,
        'window_size': window_size_list
    },
    {
        'key': ['P_emaildomain'],
        'var': var_list,
        'agg': stats_list,
        'window_size': window_size_list
    },
    {
        'key': ['card5'],
        'var': var_list,
        'agg': stats_list,
        'window_size': window_size_list
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


def _calc_rolling_func(df, key, value, window_size, stat):
    new_fe_name = f'window_{window_size}_{stat.__name__}_{value}_groupby_{"_".join(key)}'
    result = df.groupby(key)[value].rolling(window_size).apply(stat).rename(new_fe_name).reset_index()

    # 同一transactionDTのレコードがあるため、重複を削除
    cols = key + ["TransactionDT"]
    result = result[~result[cols].duplicated(keep="first")]

    # concat
    df.reset_index(inplace=True)
    df = pd.merge(df, result, on=cols, how="left")

    return df[new_fe_name]


def calc_rolling_func(df, groupby_dict):
    """
    dfのindexはdatetime型であること
    """
    arg_list = []
    for grp in groupby_dict:
        key, value_list, stat_list, window_size_list = grp['key'], grp['var'], grp['agg'], grp['window_size']
        for window_size in window_size_list:
            for stat in stat_list:
                for value in value_list:
                    arg_list.append([key, value, window_size, stat])

    result_list = Parallel(n_jobs=-1)([delayed(_calc_rolling_func)(df[arg[0] + [arg[1]]], arg[0], arg[1], arg[2], arg[3]) for arg in arg_list])
    result = pd.concat(result_list, axis=1)
    return result


# ===============
# Main class
# ===============
class Rolling(Feature):
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

        with timer("get original cols"):
            org_cols = total.columns

        with timer("Set TransactionDT"):
            total["TransactionDT"] = total["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            total.set_index("TransactionDT", inplace=True)

        with timer("get rolling features"):
            total = calc_rolling_func(total, groupby_dict)
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
    f = Rolling(FE_DIR)
    f.run().save()
