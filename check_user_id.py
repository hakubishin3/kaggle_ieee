import numpy as np
import pandas as pd

train = pd.read_csv("data/input/train_transaction.csv")[["TransactionID", "TransactionDT", "P_emaildomain", "D1", "D2", "D3", "C1", "C14"]]
test = pd.read_csv("data/input/test_transaction.csv")[["TransactionID", "TransactionDT", "P_emaildomain", "D1", "D2", "D3", "C1", "C14"]]
predicted_user = pd.read_csv("data/interim/20190901_user_ids_share.csv")

total = pd.concat([train, test], axis=0, ignore_index=True)
total = pd.merge(total, predicted_user, how="left", on="TransactionID")

total["type"] = "train"
total.loc[len(train):, "type"] = "private"
total.loc[len(train):int(len(test)*0.2)+len(train), "type"] = "public"

total["C1_C14_ratio"] = total["C1"] / total["C14"]
total["C1_C14_ratio"] = total["C1_C14_ratio"].map(lambda x: np.round(x, 3)).astype(str).fillna("#")

import pdb; pdb.set_trace()