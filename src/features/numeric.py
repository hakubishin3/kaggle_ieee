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


# ===============
# Constants
# ===============
DATA_DIR = "../../data/input/"
FE_DIR = "../../data/features/"


# ===============
# Settings
# ===============
logger = get_logger()


# ===============
# Functions
# ===============
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")

def get_id_numeric_cols():
    return [
        'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11',
    ]

def get_C_cols():
    return [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    ]

def get_D_cols():
    return [
        'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15'
    ]

def get_V_cols_related_to_Amt():
    return [
        # V126 - V137
        'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137',
        # V306 - V321
        'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321'
    ]

def get_V_cols_others():
    """
    v_cols = [f'V{i+1}' for i in range(339)]

    # remove cols related to Amt
    v_cols_related_to_amt = get_V_cols_related_to_Amt()
    v_cols_others = [col for col in v_cols if col not in v_cols_related_to_amt]

    # remove cols
    # https://www.kaggle.com/duykhanh99/update-lgb-starter-with-r-0-9480-lb
    remove_cols = [
        'V300','V309','V111','V124','V106','V125','V315','V134','V102','V123','V316','V113',
        'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
        'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
        'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120',
        'V1','V14','V41','V65','V88','V107']
    v_cols_others = [col for col in v_cols_others if col not in v_cols_related_to_amt]
    """
    v_cols_others = [
        # important features
        'V143', 'V165', 'V189', 'V201', 'V243', 'V257', 'V258', 'V283', 'V294'
    ]
    return v_cols_others

def get_numeric_cols():
    main_numeric_cols = ["TransactionAmt", "dist1", "dist2"]
    id_numeric_cols = get_id_numeric_cols()
    c_cols = get_C_cols()
    d_cols = get_D_cols()
    v_cols_related_to_amt = get_V_cols_related_to_Amt()
    v_cols_others = get_V_cols_others()

    numeric_cols = main_numeric_cols + id_numeric_cols + c_cols + d_cols + v_cols_related_to_amt + v_cols_others
    remove_cols = []
    numeric_cols = [col for col in numeric_cols if col not in remove_cols]

    return numeric_cols

# ===============
# Main class
# ===============
class Numeric(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        with timer("load data"):
            train = read_preprocessing_data(DATA_DIR, "train", write_mode=False)
            test = read_preprocessing_data(DATA_DIR, "test", write_mode=False)

        with timer("get numeric features"):
            numeric_cols = get_numeric_cols()
            self.train_feature[numeric_cols] = train[numeric_cols]
            self.test_feature[numeric_cols] = test[numeric_cols]

        with timer("make features: V features related to TransactionAmt + TransactionAmt"):
            v_cols_related_to_amt = get_V_cols_related_to_Amt()
            for col in v_cols_related_to_amt:
                new_fe_col_name = col + "_add_Amt"
                self.train_feature[new_fe_col_name] = train[col] + train["TransactionAmt"]
                self.test_feature[new_fe_col_name] = test[col] + test["TransactionAmt"]

        with timer("make features: TransactionAmt * CXX"):
            c_cols = ['C1', 'C13', 'C14']
            for col in c_cols:
                new_fe_col_name = col + "_mul_Amt"
                self.train_feature[new_fe_col_name] = train[col] * train['TransactionAmt']
                self.test_feature[new_fe_col_name] = test[col] * test['TransactionAmt']

        with timer("numeric feature processings"):
            self.train_feature["day_of_week"] = train["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday())
            self.test_feature["day_of_week"] = test["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday())
            self.train_feature['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
            self.test_feature['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

        with timer("agg V features"):
            # V1 ~ V11
            vcol_names = [f'V{i}' for i in range(1, 12)]
            self.train_feature['sum_V1_V11'] = train[vcol_names].sum(axis=1)
            self.test_feature['sum_V1_V11'] = test[vcol_names].sum(axis=1)
            self.train_feature['null_sum_V1_V11'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V1_V11'] = test[vcol_names].isnull().sum(axis=1)

            # V12 ~ V34
            vcol_names = [f'V{i}' for i in range(12, 35)]
            self.train_feature['sum_V12_V34'] = train[vcol_names].sum(axis=1)
            self.test_feature['sum_V12_V34'] = test[vcol_names].sum(axis=1)
            self.train_feature['null_sum_V12_V34'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V12_V34'] = test[vcol_names].isnull().sum(axis=1)

            # V35 ~ V52
            vcol_names = [f'V{i}' for i in range(35, 53)]
            self.train_feature['sum_V35_V52'] = train[vcol_names].sum(axis=1)
            self.test_feature['sum_V35_V52'] = test[vcol_names].sum(axis=1)
            self.train_feature['null_sum_V35_V52'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V35_V52'] = test[vcol_names].isnull().sum(axis=1)

            # V53 ~ V74
            vcol_names = [f'V{i}' for i in range(53, 75)]
            self.train_feature['sum_V53_V74'] = train[vcol_names].sum(axis=1)
            self.test_feature['sum_V53_V74'] = test[vcol_names].sum(axis=1)
            self.train_feature['null_sum_V53_V74'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V53_V74'] = test[vcol_names].isnull().sum(axis=1)

            # V75 ~ V94
            vcol_names = [f'V{i}' for i in range(75, 95)]
            self.train_feature['sum_V75_V94'] = train[vcol_names].sum(axis=1)
            self.test_feature['sum_V75_V94'] = test[vcol_names].sum(axis=1)
            self.train_feature['null_sum_V75_V94'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V75_V94'] = test[vcol_names].isnull().sum(axis=1)

            # V95 ~ V125
            vcol_names = [f'V{i}' for i in range(95, 126)]
            self.train_feature['sum_V95_V125'] = train[vcol_names].sum(axis=1)
            self.test_feature['sum_V95_V125'] = test[vcol_names].sum(axis=1)
            self.train_feature['null_sum_V95_V125'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V95_V125'] = test[vcol_names].isnull().sum(axis=1)

            # V138 ~ V166
            vcol_names = [f'V{i}' for i in range(138, 167)]
            self.train_feature['null_sum_V138_V166'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V138_V166'] = test[vcol_names].isnull().sum(axis=1)

            # V167 ~ V216
            vcol_names = [f'V{i}' for i in range(167, 217)]
            self.train_feature['null_sum_V167_V216'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V167_V216'] = test[vcol_names].isnull().sum(axis=1)

            # V217 ~ V278
            vcol_names = [f'V{i}' for i in range(217, 279)]
            self.train_feature['null_sum_V217_V278'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V217_V278'] = test[vcol_names].isnull().sum(axis=1)

            # V279 ~ V321
            vcol_names = [f'V{i}' for i in range(279, 322)]
            self.train_feature['null_sum_V279_V321'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V279_V321'] = test[vcol_names].isnull().sum(axis=1)

            # V322 ~ V339
            vcol_names = [f'V{i}' for i in range(322, 340)]
            self.train_feature['null_sum_V322_V339'] = train[vcol_names].isnull().sum(axis=1)
            self.test_feature['null_sum_V322_V339'] = test[vcol_names].isnull().sum(axis=1)

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Numeric(FE_DIR)
    f.run().save()
