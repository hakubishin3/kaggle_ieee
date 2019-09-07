import pandas as pd
from sklearn.model_selection import GroupKFold


def get_user_id(df):
    key_list = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'Registered_at', 'ProductCD']
    df["predicted_user_id"] = ""
    for col in key_list:
        df["predicted_user_id"] += df[col].astype(str) + "_"
    return df


def get_folds_per_user(train):
    train = get_user_id(train)
    print("total user:", train["predicted_user_id"].nunique())

    # create folds
    kf = GroupKFold(n_splits=5)
    folds_ids = []
    for train_index, valid_index in kf.split(train, train, train["predicted_user_id"]):
        print((train.iloc[train_index]["predicted_user_id"].value_counts() == 1).sum(), "/", train.iloc[train_index]["predicted_user_id"].nunique())
        print((train.iloc[valid_index]["predicted_user_id"].value_counts() == 1).sum(), "/", train.iloc[valid_index]["predicted_user_id"].nunique())
        folds_ids.append((train_index, valid_index))

    return folds_ids


def get_DTM(df):
    train_dtm = pd.read_csv("data/interim/train_DT_M.csv")
    test_dtm = pd.read_csv("data/interim/test_DT_M.csv")
    total_dtm = train_dtm.append(test_dtm).reset_index(drop=True)
    df = pd.merge(df, total_dtm, on='TransactionID', how='left')
    return df


def get_folds_per_DTM(train):
    train = get_DTM(train)

    # create folds
    kf = GroupKFold(n_splits=5)
    folds_ids = []
    for train_index, valid_index in kf.split(train, train, train["DT_M"]):
        folds_ids.append((train_index, valid_index))
    return folds_ids
