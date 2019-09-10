import sys
import time
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
categorical_cols = [
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'ProductCD', 'P_emaildomain', 'R_emaildomain',
    'M4'
]

# ===============
# Functions
# ===============
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


def count_encoding(col, train, test):
    total = pd.concat([train, test], ignore_index=True, sort=False)
    count_map = total[col].value_counts().to_dict()
    train_feature = train[col].map(count_map)
    test_feature = test[col].map(count_map)

    return train_feature, test_feature


# ===============
# Main class
# ===============
class Count_Encoding_Nway(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        with timer("load data"):
            train = read_preprocessing_data(DATA_DIR, "train", write_mode=False)
            test = read_preprocessing_data(DATA_DIR, "test", write_mode=False)

        with timer("count encoding"):
            for col1, col2 in combinations(categorical_cols, 2):
                new_fe_col_name = f'{col1}_{col2}'
                train[new_fe_col_name] = train[col1].astype("str") + "_" + train[col2].astype("str")
                test[new_fe_col_name] = test[col1].astype("str") + "_" + test[col2].astype("str")
                train_result, test_result = count_encoding(new_fe_col_name, train, test)
                self.train_feature[new_fe_col_name] = train_result
                self.test_feature[new_fe_col_name] = test_result

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Count_Encoding_Nway(FE_DIR)
    f.run().save()
