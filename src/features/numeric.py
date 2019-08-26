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


def get_numeric_cols():
    main_numeric_cols = ["TransactionAmt", "dist1", "dist2"]
    id_numeric_cols = get_id_numeric_cols()

    """
    numeric_cols = [
        # important
        'V143', 'V165', 'V189', 'V243', 'V257', 'V258',
        # V126 - V137
        'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137',
        # V306 - V321
        'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317',
        'V318', 'V319', 'V320', 'V321',  
    ]
    """

    numeric_cols = main_numeric_cols + id_numeric_cols
    remove_cols = ['id_02']
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

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Numeric(FE_DIR)
    f.run().save()
