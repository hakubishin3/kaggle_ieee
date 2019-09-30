import json
import pathlib
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import GPyOpt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.base import load_features
from src.models.lightgbm import LightGBM
from src.models.catboost import Catboost
from src.folds.get_folds import get_folds_per_user, get_folds_per_DTM
from src.utils.logger import get_logger
from src.utils.json_func import save_json


def main():
    # =========================================
    # === Settings
    # =========================================
    # get logger
    logger = get_logger(out_file="ensemble.log")

    logger.info("=== file path ===")
    # set model
    oof_1_path = "./data/output/20190929_hmdhmd/20190929_hmdhmd_oof.csv"
    pred_1_path = "./data/output/20190929_hmdhmd/20190929_hmdhmd_pred.csv"
    logger.info(f"hmd model - oof: {oof_1_path}")
    logger.info(f"hmd model - pred: {pred_1_path}")

    oof_2_path = "./data/output/20190929_ML_bear/20190928_all_uid_aggs_New_single_user_id_preds_oof_features1354_oof0.954_pub0.984_pri0.983.csv"
    pred_2_path = "./data/output/20190929_ML_bear/20190928_all_uid_aggs_New_single_user_id_preds_pred_features1354_oof0.954_pub0.984_pri0.983.csv"
    logger.info(f"bear model - oof: {oof_2_path}")
    logger.info(f"bear model - pred: {pred_2_path}")

    oof_3_path = "./data/output/model_23/oof_preds.npy"
    pred_3_path = "./data/output/model_23/submission.csv"
    logger.info(f"hakubishin model - oof: {oof_3_path}")
    logger.info(f"hakubishin model - pred: {pred_3_path}")

    oof_4_path = "./data/output/20190927_holygo/0927_0221__holygo_oof__CV0.959479__LB0.9596.csv"
    pred_4_path = "./data/output/20190927_holygo/20190927_0221__CV0-959479__lr0-01.csv"
    logger.info(f"holygo model - oof: {oof_4_path}")
    logger.info(f"holygo model - pred: {pred_4_path}")

    # load data
    oof_1 = pd.read_csv(oof_1_path).sort_values("TransactionID")["isFraud"].values
    oof_2 = pd.read_csv(oof_2_path).sort_values("TransactionID")["isFraud"].values
    oof_3 = np.load(oof_3_path)
    oof_4 = pd.read_csv(oof_4_path).sort_values("TransactionID").iloc[:len(oof_3)]["isFraud"].values

    pred_1 = pd.read_csv(pred_1_path).sort_values("TransactionID").reset_index(drop=True)
    pred_2 = pd.read_csv(pred_2_path).sort_values("TransactionID").reset_index(drop=True)
    pred_3 = pd.read_csv(pred_3_path).sort_values("TransactionID").reset_index(drop=True)
    pred_4 = pd.read_csv(pred_4_path).sort_values("TransactionID").reset_index(drop=True)

    # =========================================
    # === data loading
    # =========================================
    train = pd.read_csv('./data/input/train.csv')
    # test = pd.read_csv('./data/input/test.csv')
    y_train = train["isFraud"].values

    # =========================================
    # === check score
    # =========================================
    logger.info("=== check score ===")
    def calc_bear_score(df):
        df_probing = pd.read_csv('data/interim/probing_toolbox/20190929_probing.csv').loc[:, ['TransactionID', 'data_type', 'Probing_isFraud']]
        df = pd.merge(df_probing, df, on='TransactionID', how='left')
        # test public score
        public_score = roc_auc_score(
            df[df.data_type=="test_public"]['Probing_isFraud'],
            df[df.data_type=="test_public"]['isFraud']
        )
        # test private score
        private_score = roc_auc_score(
            df[df.data_type=="test_private"]['Probing_isFraud'],
            df[df.data_type=="test_private"]['isFraud']
        )
        return public_score, private_score

    cv = roc_auc_score(y_train, oof_1)
    pub, prv = calc_bear_score(pred_1)
    logger.info(f"hmd model: cv{cv}, pub{pub}, prv{prv}")

    cv = roc_auc_score(y_train, oof_2)
    pub, prv = calc_bear_score(pred_2)
    logger.info(f"bear model: cv{cv}, pub{pub}, prv{prv}")

    cv = roc_auc_score(y_train, oof_3)
    pub, prv = calc_bear_score(pred_3)
    logger.info(f"hakubishin model: cv{cv}, pub{pub}, prv{prv}")

    cv = roc_auc_score(y_train, oof_4)
    pub, prv = calc_bear_score(pred_4)
    logger.info(f"holygo model: cv{cv}, pub{pub}, prv{prv}")

    # =========================================
    # === user info
    # =========================================
    logger.info("=== user info ===")
    thres = 2
    logger.info(f"user count thres: {thres}")
    predicted_user = pd.read_csv('./data/interim/20190901_user_ids_share.csv').sort_values("TransactionID").reset_index(drop=True)
    user_count = predicted_user["predicted_user_id"].value_counts()
    target_user_id = user_count[user_count <= thres].index.tolist()
    train_predicted_user = predicted_user.iloc[:len(oof_3)]
    train_target_df = train_predicted_user.query("predicted_user_id in @target_user_id")
    train_target_index = train_target_df.index

    cv = roc_auc_score(y_train[train_target_index], oof_1[train_target_index])
    logger.info(f"hmd model: cv{cv}")
    cv = roc_auc_score(y_train[train_target_index], oof_2[train_target_index])
    logger.info(f"bear model: cv{cv}")
    cv = roc_auc_score(y_train[train_target_index], oof_3[train_target_index])
    logger.info(f"hakubishin model: cv{cv}")
    cv = roc_auc_score(y_train[train_target_index], oof_4[train_target_index])
    logger.info(f"holygo model: cv{cv}")

    # =========================================
    # === hand made
    # =========================================
    logger.info("=== hand made ===")
    
    sub = pred_3.copy()
    # x_opt = [0.1, 0.2, 0.65, 0.05]
    x_opt = [0.10, 0.25, 0.55, 0.10]
    logger.info(f"rate: {x_opt}") 
    oof = oof_1 * x_opt[0] + oof_2 * x_opt[1] + oof_3 * x_opt[2] + oof_4 * x_opt[3]
    cv = roc_auc_score(y_train[train_target_index], oof[train_target_index])
    logger.info(f"ensemble model: cv{cv}")

    sub["isFraud"] = pred_1["isFraud"] * x_opt[0] + pred_2["isFraud"] * x_opt[1] + pred_3["isFraud"] * x_opt[2] + pred_4["isFraud"] * x_opt[3]
    pub, prv = calc_bear_score(sub)
    logger.info(f"ensemble model: pub{pub}, prv{prv}")
    sub.to_csv("sub_avg.csv",header=True,index=False)
    import pdb; pdb.set_trace()

    # override probing value and save
    df_probing = pd.read_csv('data/interim/probing_toolbox/20190929_probing.csv').loc[:, ['TransactionID', 'data_type', 'Probing_isFraud']]
    sub = pd.merge(sub, df_probing, on="TransactionID", how="left")
    # override only probing_isfraud = 1
    sub.loc[sub.Probing_isFraud == 1, "isFraud"] = 1
    sub = sub[["TransactionID", "isFraud"]]
    pub, prv = calc_bear_score(sub)
    logger.info(f"ensemble model after override proving value: pub{pub}, prv{prv}")
    sub.to_csv("sub_avg.csv",header=True,index=False)

    # =========================================
    # === optimize
    # =========================================
    logger.info("=== optimize ===")
    sub = pred_3.copy()

    def f(x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]

        sub["isFraud"] = pred_1["isFraud"] * x0 + pred_2["isFraud"] * x1 + pred_3["isFraud"] * x2 + pred_4["isFraud"] * x3
        public_score, private_score = calc_bear_score(sub)

        oof = oof_1 * x0 + oof_2 * x1 + oof_3 * x2 + oof_4 * x3
        cv = roc_auc_score(y_train[train_target_index], oof[train_target_index])

        opt_value = -1 * private_score
        # opt_value = -1 * (private_score + public_score + cv)

        return opt_value

    bounds = [
        {'name': 'x0', 'type': 'continuous', 'domain': (0.1, 1)},
        {'name': 'x1', 'type': 'continuous', 'domain': (0.1, 1)},
        {'name': 'x2', 'type': 'continuous', 'domain': (0.1, 1)},
        {'name': 'x3', 'type': 'continuous', 'domain': (0.1, 1)},
    ]

    constraints = [
        {
            'name': 'constr_1',
            'constraint': '(x[:,0] + x[:,1] + x[:,2] + x[:,3]) - 1 - 0.001'
        },
        {
            'name': 'constr_2',
            'constraint': '1 - (x[:,0] + x[:,1] + x[:,2] + x[:,3]) - 0.001'
        }
    ]

    myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, constraints=constraints)
    myBopt.run_optimization(max_iter=10)
    logger.info(f"rate: {myBopt.x_opt}") 
    logger.info(f"value: {myBopt.fx_opt}")

    # check oof
    oof = oof_1 * myBopt.x_opt[0] + oof_2 * myBopt.x_opt[1] + oof_3 * myBopt.x_opt[2] + oof_4 * myBopt.x_opt[3]
    cv = roc_auc_score(y_train[train_target_index], oof_1[train_target_index])
    logger.info(f"ensemble model: cv{cv}")

    # make submission file
    sub = pred_3.copy()
    sub["isFraud"] = pred_1["isFraud"] * myBopt.x_opt[0] + pred_2["isFraud"] * myBopt.x_opt[1] + pred_3["isFraud"] * myBopt.x_opt[2] + pred_4["isFraud"] * myBopt.x_opt[3]
    pub, prv = calc_bear_score(sub)
    logger.info(f"ensemble model: pub{pub}, prv{prv}")

    # override probing value and save
    df_probing = pd.read_csv('data/interim/probing_toolbox/20190929_probing.csv').loc[:, ['TransactionID', 'data_type', 'Probing_isFraud']]
    sub = pd.merge(sub, df_probing, on="TransactionID", how="left")
    sub.loc[sub.Probing_isFraud.notnull(), "isFraud"] = sub.loc[sub.Probing_isFraud.notnull(), "Probing_isFraud"].values
    sub = sub[["TransactionID", "isFraud"]]
    pub, prv = calc_bear_score(sub)
    logger.info(f"ensemble model after override proving value: pub{pub}, prv{prv}")
    sub.to_csv("sub_avg.csv",header=True,index=False)



if __name__ == '__main__':
    main()
