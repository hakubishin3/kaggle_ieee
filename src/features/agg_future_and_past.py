import sys
import time
import datetime
import numpy as np
import pandas as pd
from contextlib import contextmanager
from base import Feature
from itertools import combinations

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
n_days_list = [7, 30, 60, 120, 160]

# ===============
# Functions
# ===============
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


def sum_future(total, train_index, test_index, n_day, id_col, date_col, value):
    total['threshold'] = total[date_col] + datetime.timedelta(days=n_day)

    # narrow down the data.
    grp_result = total.groupby(id_col).size()
    more_one_visitor_id = grp_result[grp_result != 1].index.tolist()
    search_df = total[total[id_col].isin(more_one_visitor_id)][
        [id_col, date_col, 'threshold', value]].copy()
    search_df = search_df.sort_values([id_col, date_col], ascending=[False, True])

    max_iter = total.groupby(id_col).size().max()
    hits_col_list = []
    for n_lag in range(1, max_iter + 1):
        # shift hits
        hits_col = f'{value}_lag_{n_lag}'
        hits_col_list.append(hits_col)
        search_df[hits_col] = search_df.groupby(id_col)[date_col].shift(-1 * n_lag)
        search_df[hits_col] = (search_df[hits_col] <= search_df['threshold']).astype(int)
        search_df[hits_col] = search_df[hits_col] * search_df.groupby(id_col)[value].shift(-1 * n_lag)
    
    # summary count
    search_df[f'{value}_sum'] = search_df[hits_col_list].sum(axis=1)
    result_col = [f'{value}_sum']
    total = total.join(search_df[result_col], how='outer').fillna({f'{value}_sum': 0})
    train_result = total.loc[train_index][result_col].reset_index(drop=True)
    test_result = total.loc[test_index][result_col].reset_index(drop=True)

    return train_result, test_result


# ===============
# Main class
# ===============
class Agg_Future(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        feature_name_list = []

        with timer("load data"):
            train = read_preprocessing_data(DATA_DIR, "train", write_mode=False)
            test = read_preprocessing_data(DATA_DIR, "test", write_mode=False)

        with timer("get predicted user id"):
            predicted_user = pd.read_csv('../../data/interim/20190901_user_ids_share.csv')
            train = pd.merge(train, predicted_user[['TransactionID', 'predicted_user_id']], how='left', on='TransactionID')
            test = pd.merge(test, predicted_user[['TransactionID', 'predicted_user_id']], how='left', on='TransactionID')

        with timer("Set TransactionDT"):
            train["TransactionDT"] = train["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            test["TransactionDT"] = test["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            total = train.append(test).reset_index(drop=True)

        with timer("Get lag-time to future recoreds"):
            total['TransactionDT_lag1_future'] = total.groupby('predicted_user_id')['TransactionDT'].shift(-1)
            total["Diff_Time_To_lag1_future"] = (total["TransactionDT_lag1_future"] - total["TransactionDT"]).apply(lambda x: x.days*24+x.seconds/60/60 if x == x else x).astype(np.int64)
            total['TransactionDT_lag1_past'] = total.groupby('predicted_user_id')['TransactionDT'].shift(1)
            total["Diff_Time_To_lag1_past"] = (total["TransactionDT_lag1_past"] - total["TransactionDT"]).apply(lambda x: x.days*24+x.seconds/60/60 if x == x else x).astype(np.int64)
            feature_name_list.extend([
                "Diff_Time_To_lag1_future", "Diff_Time_To_lag1_past"
            ])

        with timer("Get lag-value"):
            # future value
            total['TransactionAmt_lag1_future'] = total.groupby('predicted_user_id')['TransactionAmt'].shift(-1)
            feature_name_list.extend([
                "TransactionAmt_lag1_future"
            ])

            # past value
            total['TransactionAmt_lag1_past'] = total.groupby('predicted_user_id')['TransactionAmt'].shift(1)
            feature_name_list.extend([
                "TransactionAmt_lag1_past"
            ])

            # current value - future value
            total['diff_TransactionAmt_lag1_future'] = total['TransactionAmt'] - total['TransactionAmt_lag1_future']
            feature_name_list.extend([
                "diff_TransactionAmt_lag1_future"
            ])

            # current value - past value
            total['diff_TransactionAmt_lag1_past'] = total['TransactionAmt'] - total['TransactionAmt_lag1_past']
            feature_name_list.extend([
                "diff_TransactionAmt_lag1_past"
            ])

            # current value / future value
            total['div_TransactionAmt_lag1_future'] = total['TransactionAmt'] / total['TransactionAmt_lag1_future'] 
            feature_name_list.extend([
                "div_TransactionAmt_lag1_future"
            ])

            # current value / past value
            total['div_TransactionAmt_lag1_past'] = total['TransactionAmt'] / total['TransactionAmt_lag1_past']
            feature_name_list.extend([
                "div_TransactionAmt_lag1_past"
            ])

            # slope
            total['slope_TransactionAmt_lag1_future'] = (total['TransactionAmt'] - total['TransactionAmt_lag1_future']) / total['Diff_Time_To_lag1_future']
            total['slope_TransactionAmt_lag1_past'] = (total['TransactionAmt'] - total['TransactionAmt_lag1_past']) / total['Diff_Time_To_lag1_past']
            feature_name_list.extend([
                "slope_TransactionAmt_lag1_future", "slope_TransactionAmt_lag1_past",
            ])

            # groupby value - future value
            grp_Amt = total.groupby('predicted_user_id')['TransactionAmt'].mean().reset_index()
            grp_Amt.columns = ['predicted_user_id', 'groupby_TransactionAmt']
            total = total.merge(grp_Amt, how='left', on='predicted_user_id')
            total['diff_grp_TransactionAmt_lag1_future'] = total['groupby_TransactionAmt'] - total['TransactionAmt_lag1_future']
            total['diff_grp_TransactionAmt_lag1_past'] = total['groupby_TransactionAmt'] - total['TransactionAmt_lag1_past']
            feature_name_list.extend([
                "diff_grp_TransactionAmt_lag1_future", "diff_grp_TransactionAmt_lag1_past",
            ])

            # アイデア
            """
            ・過去からの累積値
            ・未来からの累積値
            ・1個前と値が同じか否か
            ・1個後と値が同じか否か
            ・過去にない値が出現したかどうか
            ・過去からのdistinctの累積値
            """

        with timer("end"):
            train_result = total.iloc[:len(train)].reset_index(drop=True)
            test_result = total.iloc[len(train):].reset_index(drop=True)
            for fe in feature_name_list:
                self.train_feature[fe] = train_result[fe]
                self.test_feature[fe] = test_result[fe]

            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Agg_Future(FE_DIR)
    f.run().save()
