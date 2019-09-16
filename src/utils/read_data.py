import sys
import numpy as np
import pandas as pd
import datetime
from pathlib import Path

if Path.cwd().name == 'kaggle_ieee':
    from src.utils.tools import reduce_mem_usage
elif Path.cwd().name == 'utils':
    sys.path.append("../")
    from utils.tools import reduce_mem_usage


def read_transaction(data_dir, data_type="train"):
    df = pd.read_csv(data_dir + f"{data_type}_transaction.csv")
    return df


def read_identity(data_dir, data_type="train"):
    df = pd.read_csv(data_dir + f"{data_type}_identity.csv")
    return df


def read_data(data_dir, data_type="train"):
    transaction = read_transaction(data_dir, data_type)
    identity = read_identity(data_dir, data_type)
    all_data = pd.merge(transaction, identity, on='TransactionID', how='left')
    all_data = reduce_mem_usage(all_data)
    return all_data


def change_TransactionDT(df):
    start_date = "2017-11-30"
    df["TransactionDT"] = df["TransactionDT"].apply(
        lambda x: datetime.datetime.strptime(start_date, "%Y-%m-%d") +\
        datetime.timedelta(seconds=int(x))
    )
    return df


def add_TransactionDT_LocalTime(df):
    df["TransactionDT_LocalTime"] = df.apply(
        lambda x: x["TransactionDT"] + datetime.timedelta(seconds=x["id_14"]*60)\
        if x["id_14"] == x["id_14"] else np.nan, axis=1
    )
    return df

def get_Registered_at(df):
    """
    add columns
    - Registered_at: datetime
    """
    df["Registered_at"] = df.apply(
        lambda x: (x["TransactionDT"] - datetime.timedelta(days=x["D1"])).strftime("%Y-%m-%d")\
        if x["D1"] == x["D1"] else np.nan, axis=1
    )
    return df


def get_D9(df):
    df["D9"] = df["TransactionDT"].apply(lambda x: x.hour)
    return df


def get_D9_LocalTime(df):
    df["D9_LocalTime"] = df["TransactionDT_LocalTime"].apply(lambda x: x.hour)
    return df


def get_emaildomain_v2(df):
    """
    - anonymous.comとmail.comはNULL
    - emaildomainの名寄せ
    https://www.kaggle.com/krishonaveen/xtreme-boost-and-feature-engineering/notebook

    add columns
    - P_emaildomain_v2: string
    - P_emaildomain_bin: string
    - P_emaildomain_suffix: string
    - R_emaildomain_v2: string
    - R_emaildomain_bin: string
    - R_emaildomain_suffix: string
    """
    df["P_emaildomain_v2"] = df["P_emaildomain"].replace("anonymous.com", np.nan).replace("mail.com", np.nan)
    df["R_emaildomain_v2"] = df["R_emaildomain"].replace("anonymous.com", np.nan).replace("mail.com", np.nan)
    emails = {
        'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other',
        'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other',
        'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum',
        'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
        'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
        'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',
        'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
        'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
        'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
        'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
        'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
        'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
        'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
        'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',
        'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'
    }
    us_emails = ['gmail', 'net', 'edu']
    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    return df


def read_preprocessing_data(data_dir, data_type="train", write_mode=False):
    if write_mode is True:
        data = read_data(data_dir, data_type)
        data = change_TransactionDT(data)
        data = add_TransactionDT_LocalTime(data)
        data = get_Registered_at(data)
        data = get_D9(data)
        data = get_D9_LocalTime(data)
        data = get_emaildomain_v2(data)
        data.to_csv(data_dir + f"{data_type}.csv", header=True, index=False)
    else:
        data = pd.read_csv(data_dir + f"{data_type}.csv")

    return data


def get_user_id(df):
    key_list = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'Registered_at', 'ProductCD']
    df["predicted_user_id"] = ""
    for col in key_list:
        df["predicted_user_id"] += df[col].astype(str) + "_"
    return df


if __name__ == "__main__":
    data_dir = "../../data/input/"
    train = read_preprocessing_data(data_dir, "train", write_mode=True)
    test = read_preprocessing_data(data_dir, "test", write_mode=True)
