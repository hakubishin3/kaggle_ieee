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
# logger = get_logger(out_file="label_encoding.log")


# ===============
# Functions
# ===============
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")

def load_os_release_date():
    os_info = pd.read_csv(DATA_DIR + "os_release_date.csv")
    return os_info

def load_browser_release_date():
    browser_info = pd.read_csv(DATA_DIR + "browser_release_date.csv")
    return browser_info

# ===============
# Main class
# ===============
class OS_and_Browser(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        with timer("load data"):
            train = read_preprocessing_data(DATA_DIR, "train", write_mode=False)
            test = read_preprocessing_data(DATA_DIR, "test", write_mode=False)

        with timer("concat os and browser info"):
            os_info = load_os_release_date()
            train = pd.merge(train, os_info, how="left", left_on="id_30", right_on="os_name").drop(columns="os_name")
            test = pd.merge(test, os_info, how="left", left_on="id_30", right_on="os_name").drop(columns="os_name")

            browser_info = load_browser_release_date()
            train = pd.merge(train, browser_info, how="left", left_on="id_31", right_on="browser_name").drop(columns="browser_name")
            test = pd.merge(test, browser_info, how="left", left_on="id_31", right_on="browser_name").drop(columns="browser_name")

        with timer("convert object-type to datetime-type"):
            train["TransactionDT"] = train["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            test["TransactionDT"] = test["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

            train["os_release_date"] = train["os_release_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d") if x==x else x)
            test["os_release_date"] = test["os_release_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d") if x==x else x)

            train["browser_release_date"] = train["browser_release_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d") if x==x else x)
            test["browser_release_date"] = test["browser_release_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d") if x==x else x)

        with timer("make features: elapsed times from release date"):
            self.train_feature["elapsed_days_from_os_release"] = (train["TransactionDT"] - train["os_release_date"]).apply(lambda x: x.days)
            self.test_feature["elapsed_days_from_os_release"] = (test["TransactionDT"] - test["os_release_date"]).apply(lambda x: x.days)

            self.train_feature["elapsed_days_from_browser_release"] = (train["TransactionDT"] - train["browser_release_date"]).apply(lambda x: x.days)
            self.test_feature["elapsed_days_from_browser_release"] = (test["TransactionDT"] - test["browser_release_date"]).apply(lambda x: x.days)

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = OS_and_Browser(FE_DIR)
    f.run().save()
