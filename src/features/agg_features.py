import sys
import time
import datetime
import itertools
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
var_list = ["TransactionAmt", "V258", "V257", "V201", "TransactionAmt_decimal"]
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
        'key': ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'ProductCD'],
        'var': var_list,
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
        'key': ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'ProductCD'],
        'var': var_list,
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
            org_cols = total.columns

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
