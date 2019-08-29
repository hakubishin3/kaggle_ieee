import sys
import time
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
# logger = get_logger(out_file="label_encoding.log")


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
    v_cols_related_to_amt = get_V_cols_related_to_Amt()
    v_cols = [f"V{i+1}" for i in range(339)]
    v_cols_others = [col for col in v_cols if col not in v_cols_related_to_amt]
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

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Numeric(FE_DIR)
    f.run().save()
