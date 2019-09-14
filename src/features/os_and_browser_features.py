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

def get_new_browser(df):
    df["new_browser_name"] = np.nan

    # samsung browser
    df.loc[(df["browser_type"] == "samsung browser") & (df["TransactionDT"] >= "2018-12-21"), "new_browser_name"] = "samsung browser 8.2"
    df.loc[(df["browser_type"] == "samsung browser") & (df["TransactionDT"] >= "2018-08-19") & (df["TransactionDT"] < "2018-12-21"), "new_browser_name"] = "samsung browser 7.4"
    df.loc[(df["browser_type"] == "samsung browser") & (df["TransactionDT"] >= "2018-06-07") & (df["TransactionDT"] < "2018-08-19"), "new_browser_name"] = "samsung browser 7.2"
    df.loc[(df["browser_type"] == "samsung browser") & (df["TransactionDT"] >= "2018-04-12") & (df["TransactionDT"] < "2018-06-07"), "new_browser_name"] = "samsung browser 7.0"
    df.loc[(df["browser_type"] == "samsung browser") & (df["TransactionDT"] >= "2018-02-19") & (df["TransactionDT"] < "2018-04-12"), "new_browser_name"] = "samsung browser 6.4"
    df.loc[(df["browser_type"] == "samsung browser") & (df["TransactionDT"] >= "2017-10-30") & (df["TransactionDT"] < "2018-02-19"), "new_browser_name"] = "samsung browser 6.2"

    # safari
    df.loc[(df["browser_type"] == "safari") & (df["TransactionDT"] >= "2018-09-17"), "new_browser_name"] = "safari 12.0"
    df.loc[(df["browser_type"] == "safari") & (df["TransactionDT"] >= "2017-09-19") & (df["TransactionDT"] < "2018-09-17"), "new_browser_name"] = "safari 11.0"

    # opera
    df.loc[(df["browser_type"] == "opera") & (df["TransactionDT"] >= "2018-09-25"), "new_browser_name"] = "opera 56.0"
    df.loc[(df["browser_type"] == "opera") & (df["TransactionDT"] >= "2018-08-16") & (df["TransactionDT"] < "2018-09-25"), "new_browser_name"] = "opera 55.0"
    df.loc[(df["browser_type"] == "opera") & (df["TransactionDT"] >= "2018-06-28") & (df["TransactionDT"] < "2018-08-16"), "new_browser_name"] = "opera 54.0"
    df.loc[(df["browser_type"] == "opera") & (df["TransactionDT"] >= "2018-05-10") & (df["TransactionDT"] < "2018-06-28"), "new_browser_name"] = "opera 53.0"
    df.loc[(df["browser_type"] == "opera") & (df["TransactionDT"] >= "2018-03-22") & (df["TransactionDT"] < "2018-05-10"), "new_browser_name"] = "opera 52.0"
    df.loc[(df["browser_type"] == "opera") & (df["TransactionDT"] >= "2018-02-07") & (df["TransactionDT"] < "2018-03-22"), "new_browser_name"] = "opera 51.0"
    df.loc[(df["browser_type"] == "opera") & (df["TransactionDT"] >= "2017-11-08") & (df["TransactionDT"] < "2018-02-07"), "new_browser_name"] = "opera 49.0"

    # mobile safari
    df.loc[(df["browser_type"] == "mobile safari") & (df["TransactionDT"] >= "2018-09-17"), "new_browser_name"] = "mobile safari 12.0"
    df.loc[(df["browser_type"] == "mobile safari") & (df["TransactionDT"] >= "2017-09-19") & (df["TransactionDT"] < "2018-09-17"), "new_browser_name"] = "mobile safari 11.0"

    # ie for desktop
    df.loc[(df["browser_type"] == "ie for desktop"), "new_browser_name"] = "ie 11.0 for desktop"

    # ie for tablet
    df.loc[(df["browser_type"] == "ie for tablet"), "new_browser_name"] = "ie 11.0 for tablet"

    # google search application
    df.loc[(df["browser_type"] == "google search application") & (df["TransactionDT"] >= "2018-03-06"), "new_browser_name"] = "google search application 65.0"
    df.loc[(df["browser_type"] == "google search application") & (df["TransactionDT"] >= "2018-01-24") & (df["TransactionDT"] < "2018-03-06"), "new_browser_name"] = "google search application 64.0"
    df.loc[(df["browser_type"] == "google search application") & (df["TransactionDT"] >= "2017-12-06") & (df["TransactionDT"] < "2018-01-24"), "new_browser_name"] = "google search application 63.0"
    df.loc[(df["browser_type"] == "google search application") & (df["TransactionDT"] >= "2017-10-17") & (df["TransactionDT"] < "2017-12-06"), "new_browser_name"] = "google search application 62.0"

    # firefox mobile
    df.loc[(df["browser_type"] == "firefox mobile") & (df["TransactionDT"] >= "2018-10-13"), "new_browser_name"] = "firefox mobile 63.0"
    df.loc[(df["browser_type"] == "firefox mobile") & (df["TransactionDT"] >= "2018-09-05") & (df["TransactionDT"] < "2018-10-13"), "new_browser_name"] = "firefox mobile 62.0"
    df.loc[(df["browser_type"] == "firefox mobile") & (df["TransactionDT"] >= "2018-06-25") & (df["TransactionDT"] < "2018-09-05"), "new_browser_name"] = "firefox mobile 61.0"

    # firefox
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2018-12-11"), "new_browser_name"] = "firefox 64.0"
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2018-10-23") & (df["TransactionDT"] < "2018-12-11"), "new_browser_name"] = "firefox 63.0"
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2018-09-05") & (df["TransactionDT"] < "2018-10-23"), "new_browser_name"] = "firefox 62.0"
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2018-06-25") & (df["TransactionDT"] < "2018-09-05"), "new_browser_name"] = "firefox 61.0"
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2018-05-09") & (df["TransactionDT"] < "2018-06-25"), "new_browser_name"] = "firefox 60.0"
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2018-03-13") & (df["TransactionDT"] < "2018-05-09"), "new_browser_name"] = "firefox 59.0"
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2018-01-23") & (df["TransactionDT"] < "2018-03-13"), "new_browser_name"] = "firefox 58.0"
    df.loc[(df["browser_type"] == "firefox") & (df["TransactionDT"] >= "2017-11-14") & (df["TransactionDT"] < "2018-01-23"), "new_browser_name"] = "firefox 57.0"

    # edge
    df.loc[(df["browser_type"] == "edge") & (df["TransactionDT"] >= "2018-11-13"), "new_browser_name"] = "edge 18.0"
    df.loc[(df["browser_type"] == "edge") & (df["TransactionDT"] >= "2018-04-30") & (df["TransactionDT"] < "2018-11-13"), "new_browser_name"] = "edge 17.0"
    df.loc[(df["browser_type"] == "edge") & (df["TransactionDT"] >= "2017-09-26") & (df["TransactionDT"] < "2018-04-30"), "new_browser_name"] = "edge 16.0"

    # chrome for ios
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-12-04"), "new_browser_name"] = "chrome 71.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-10-16") & (df["TransactionDT"] < "2018-12-04"), "new_browser_name"] = "chrome 70.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-09-04") & (df["TransactionDT"] < "2018-10-16"), "new_browser_name"] = "chrome 69.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-07-24") & (df["TransactionDT"] < "2018-09-04"), "new_browser_name"] = "chrome 68.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-05-29") & (df["TransactionDT"] < "2018-07-24"), "new_browser_name"] = "chrome 67.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-04-17") & (df["TransactionDT"] < "2018-05-29"), "new_browser_name"] = "chrome 66.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-03-06") & (df["TransactionDT"] < "2018-04-17"), "new_browser_name"] = "chrome 65.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2018-01-24") & (df["TransactionDT"] < "2018-03-06"), "new_browser_name"] = "chrome 64.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2017-12-05") & (df["TransactionDT"] < "2018-01-24"), "new_browser_name"] = "chrome 63.0 for ios"
    df.loc[(df["browser_type"] == "chrome for ios") & (df["TransactionDT"] >= "2017-10-18") & (df["TransactionDT"] < "2017-12-05"), "new_browser_name"] = "chrome 62.0 for ios"

    # chrome for android
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-12-04"), "new_browser_name"] = "chrome 71.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-10-17") & (df["TransactionDT"] < "2018-12-04"), "new_browser_name"] = "chrome 70.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-09-04") & (df["TransactionDT"] < "2018-10-17"), "new_browser_name"] = "chrome 69.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-07-24") & (df["TransactionDT"] < "2018-09-04"), "new_browser_name"] = "chrome 68.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-05-31") & (df["TransactionDT"] < "2018-07-24"), "new_browser_name"] = "chrome 67.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-04-17") & (df["TransactionDT"] < "2018-05-31"), "new_browser_name"] = "chrome 66.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-03-06") & (df["TransactionDT"] < "2018-04-17"), "new_browser_name"] = "chrome 65.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2018-01-23") & (df["TransactionDT"] < "2018-03-06"), "new_browser_name"] = "chrome 64.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2017-12-05") & (df["TransactionDT"] < "2018-01-23"), "new_browser_name"] = "chrome 63.0 for android"
    df.loc[(df["browser_type"] == "chrome for android") & (df["TransactionDT"] >= "2017-10-19") & (df["TransactionDT"] < "2017-12-05"), "new_browser_name"] = "chrome 62.0 for android"

    # chrome
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-12-04"), "new_browser_name"] = "chrome 71.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-10-16") & (df["TransactionDT"] < "2018-12-04"), "new_browser_name"] = "chrome 70.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-09-04") & (df["TransactionDT"] < "2018-10-16"), "new_browser_name"] = "chrome 69.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-07-24") & (df["TransactionDT"] < "2018-09-04"), "new_browser_name"] = "chrome 68.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-05-29") & (df["TransactionDT"] < "2018-07-24"), "new_browser_name"] = "chrome 67.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-04-17") & (df["TransactionDT"] < "2018-05-29"), "new_browser_name"] = "chrome 66.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-03-06") & (df["TransactionDT"] < "2018-04-17"), "new_browser_name"] = "chrome 65.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2018-01-24") & (df["TransactionDT"] < "2018-03-06"), "new_browser_name"] = "chrome 64.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2017-12-06") & (df["TransactionDT"] < "2018-01-24"), "new_browser_name"] = "chrome 63.0"
    df.loc[(df["browser_type"] == "chrome") & (df["TransactionDT"] >= "2017-10-17") & (df["TransactionDT"] < "2017-12-06"), "new_browser_name"] = "chrome 62.0"

    return df


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
            os_info.sort_values(["os_type", "os_release_date"], inplace=True)
            os_info["os_release_date_next_ver"] = os_info.groupby("os_type")["os_release_date"].shift(-1).fillna("2019-09-01")
            train = pd.merge(train, os_info, how="left", left_on="id_30", right_on="os_name").drop(columns="id_30")
            test = pd.merge(test, os_info, how="left", left_on="id_30", right_on="os_name").drop(columns="id_30")

            browser_info = load_browser_release_date()
            browser_info.sort_values(["browser_type", "browser_release_date"], inplace=True)
            browser_info["browser_release_date_next_ver"] = browser_info.groupby("browser_type")["browser_release_date"].shift(-1).fillna("2019-09-01")
            train = pd.merge(train, browser_info, how="left", left_on="id_31", right_on="browser_name").drop(columns="id_31")
            test = pd.merge(test, browser_info, how="left", left_on="id_31", right_on="browser_name").drop(columns="id_31")

            total = train.append(test).reset_index(drop=True)
            feature_name_list = []

        with timer("convert object-type to datetime-type"):
            total["TransactionDT"] = total["TransactionDT"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            total["os_release_date"] = total["os_release_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d") if x==x else x)
            total["browser_release_date"] = total["browser_release_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d") if x==x else x)

            total = get_new_browser(total)

        with timer("make features: elapsed times from release date"):
            total["elapsed_days_from_os_release"] = (total["TransactionDT"] - total["os_release_date"]).apply(lambda x: x.days)
            total["elapsed_days_from_browser_release"] = (total["TransactionDT"] - total["browser_release_date"]).apply(lambda x: x.days)
            feature_name_list.extend([
                "elapsed_days_from_os_release", "elapsed_days_from_browser_release"
            ])

        with timer("make features: elapsed times from new-version"):
            total = pd.merge(total, browser_info[["browser_name", "browser_release_date"]].rename(columns={"browser_name": "new_browser_name", "browser_release_date": "new_browser_release_date"}), how="left", on="new_browser_name")

            total["latest_browser"] = np.nan
            total.loc[total["browser_name"].notnull(), "latest_browser"] = (total.loc[total["browser_name"].notnull(), "browser_name"] == total.loc[total["browser_name"].notnull(), "new_browser_name"]).astype(int)

            total["new_browser_release_date"] = total["new_browser_release_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d") if x==x else x)
            total["elapsed_days_from_new_browser_release"] = (total["TransactionDT"] - total["new_browser_release_date"]).apply(lambda x: x.days)
            total.loc[total["latest_browser"] == 1, "elapsed_days_from_new_browser_release"] = 0

            total["elapsed_days_from_new_browser_release_v2"] = total["elapsed_days_from_new_browser_release"] + total["elapsed_days_from_browser_release"]
            total.loc[total["latest_browser"] == 1, "elapsed_days_from_new_browser_release_v2"] = 0

            feature_name_list.extend([
                "elapsed_days_from_new_browser_release", "latest_browser",
                "elapsed_days_from_new_browser_release_v2"
            ])

        with timer("end"):
            train_result = total.iloc[:len(train)].reset_index(drop=True)
            test_result = total.iloc[len(train):].reset_index(drop=True)
            for fe in feature_name_list:
                self.train_feature[fe] = train_result[fe]
                self.test_feature[fe] = test_result[fe]

            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = OS_and_Browser(FE_DIR)
    f.run().save()
