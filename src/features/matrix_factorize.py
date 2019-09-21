import sys
import time
import datetime
import itertools
import numpy as np
import pandas as pd
from contextlib import contextmanager
from base import Feature
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("../")
from utils.logger import get_logger
from utils.read_data import read_preprocessing_data
from utils.feature_module import CategoryVectorizer

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
    'card1', 'card2', 'card5', 'card6', 'addr1'
]
n_components = 5

# ===============
# Functions
# ===============
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")

def matrix_factorize(cols, train, test):
    total_features = pd.DataFrame()
    for col in cols:
        le = LabelEncoder()
        train_label = list(train[col].astype(str).values)
        test_label = list(test[col].astype(str).values)
        total_label = train_label + test_label
        le.fit(total_label)
        total_features[col] = le.transform(total_label)

    cv = CategoryVectorizer(cols, n_components,
             vectorizer=CountVectorizer(),
             transformer=LatentDirichletAllocation(n_components=n_components, n_jobs=-1, learning_method='online', random_state=777),
             name='CountLDA')
    features1 = cv.transform(total_features).astype(np.float32)

    cv = CategoryVectorizer(cols, n_components,
             vectorizer=CountVectorizer(),
             transformer=TruncatedSVD(n_components=n_components, random_state=777),
             name='CountSVD')
    features2 = cv.transform(total_features).astype(np.float32)

    total = pd.concat([features1, features2], axis=1)
    train_features = total.iloc[:len(train)].reset_index(drop=True)
    test_features = total.iloc[len(train):].reset_index(drop=True)

    return train_features, test_features

# ===============
# Main class
# ===============
class Matrix_Factorize(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        with timer("load data"):
            train = read_preprocessing_data(DATA_DIR, "train", write_mode=False)
            test = read_preprocessing_data(DATA_DIR, "test", write_mode=False)

        """
        with timer("get predicted user id"):
            predicted_user = pd.read_csv('../../data/interim/20190901_user_ids_share.csv')
            train = pd.merge(train, predicted_user[['TransactionID', 'predicted_user_id']], how='left', on='TransactionID')
            test = pd.merge(test, predicted_user[['TransactionID', 'predicted_user_id']], how='left', on='TransactionID')
        """

        with timer("create features"):
            train_result, test_result = matrix_factorize(categorical_cols, train, test)
            self.train_feature = train_result
            self.test_feature = test_result

        with timer("end"):
            self.train_feature.reset_index(drop=True, inplace=True)
            self.test_feature.reset_index(drop=True, inplace=True)


if __name__ == "__main__":
    f = Matrix_Factorize(FE_DIR)
    f.run().save()
