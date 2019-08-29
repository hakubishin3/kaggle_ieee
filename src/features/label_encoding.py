import sys
import time
import pandas as pd
from contextlib import contextmanager
from base import Feature
from sklearn.preprocessing import LabelEncoder

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


def label_encoding(col, train, test):
    le = LabelEncoder()
    train_label = list(train[col].astype(str).values)
    test_label = list(test[col].astype(str).values)
    total_label = train_label + test_label
    le.fit(total_label)
    train_feature = le.transform(train_label)
    test_feature = le.transform(test_label)

    return train_feature, test_feature


# ===============
# Main class
# ===============
class Label_Encoding(Feature):
    def categorical_features(self):
        categorical_cols_transaction = [
            'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'ProductCD', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
        ]
        categorical_cols_identity = [
            'DeviceType', 'DeviceInfo',
            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19',
            'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
        ]
        categorical_cols = categorical_cols_identity + categorical_cols_transaction

        # remove columns
        remove_cols = []
        categorical_cols = [col for col in categorical_cols if col not in remove_cols]

        # add columns
        add_cols = ["P_emaildomain_v2", "R_emaildomain_v2", "P_emaildomain_bin", "R_emaildomain_bin", "P_emaildomain_suffix", "R_emaildomain_suffix"]
        categorical_cols = categorical_cols + add_cols

        return categorical_cols

    def create_features(self):
        with timer("load data"):
            train = read_preprocessing_data(DATA_DIR, "train", write_mode=False)
            test = read_preprocessing_data(DATA_DIR, "test", write_mode=False)

        with timer("label encoding"):
            categorical_cols = self.categorical_features()
            for col in categorical_cols:
                train_result, test_result = label_encoding(col, train, test)
                self.train_feature[col] = train_result
                self.test_feature[col] = test_result

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Label_Encoding(FE_DIR)
    f.run().save()
