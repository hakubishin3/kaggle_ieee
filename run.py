import json
import pathlib
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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


model_map = {
    'lightgbm': LightGBM,
    'catboost': Catboost
}


def main():
    # =========================================
    # === Settings
    # =========================================
    # get logger
    logger = get_logger()

    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/model_0.json')
    parser.add_argument('--debug', '-d', action='store_true')
    args = parser.parse_args()
    logger.info(f'config: {args.config}')
    logger.info(f'debug: {args.debug}')

    # get config
    config = json.load(open(args.config))
    config.update({
        'args': {
            'config': args.config
        }
    })

    # make model-output-dir
    output_dir = config['dataset']['output_directory']
    model_no = args.config.split('/')[-1].split('.')[0]
    model_output_dir = pathlib.Path(output_dir + model_no + '/')
    if not model_output_dir.exists():
        model_output_dir.mkdir()
    logger.info(f'model_output_dir: {str(model_output_dir)}')
    logger.debug(f'model_output_dir exists: {model_output_dir.exists()}')
    config.update({
        'model_output_dir': str(model_output_dir)
    })

    # =========================================
    # === data loading
    # =========================================
    train = pd.read_csv('./data/input/train.csv')
    test = pd.read_csv('./data/input/test.csv')
    y_train = train["isFraud"].values

    # =========================================
    # === loading features
    # =========================================
    # load features
    logger.info('load features')
    x_train, x_test = load_features(config, args.debug)
    feature_name = x_test.columns
    logger.debug(f'number of features: {len(feature_name)}')

    """
    # =========================================
    # === Adversarial Validation
    # =========================================
    logger.info("adversarial validation")
    train_adv = x_train
    test_adv = x_test
    train_adv['target'] = 0
    test_adv['target'] = 1
    train_test_adv = pd.concat([train_adv, test_adv], axis=0, sort=False).reset_index(drop=True)
    target = train_test_adv['target'].values

    train_set, val_set = train_test_split(train_test_adv, test_size=0.33, random_state=71, shuffle=True)
    x_train_adv = train_set[feature_name]
    y_train_adv = train_set['target']
    x_val_adv = val_set[feature_name]
    y_val_adv = val_set['target']
    logger.debug(f'the number of train set: {len(x_train_adv)}')
    logger.debug(f'the number of valid set: {len(x_val_adv)}')

    train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
    val_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)
    lgb_model_params = config["adversarial_validation"]["lgb_model_params"]
    lgb_train_params = config["adversarial_validation"]["lgb_train_params"]
    clf = lgb.train(
        lgb_model_params,
        train_lgb,
        valid_sets=[train_lgb, val_lgb],
        valid_names=['train', 'valid'],
        **lgb_train_params
    )

    feature_imp = pd.DataFrame(
        sorted(zip(clf.feature_importance(importance_type='gain'), feature_name)), columns=['value', 'feature']
    )
    plt.figure(figsize=(20, 10))
    sns.barplot(x='value', y='feature', data=feature_imp.sort_values(by='value', ascending=False).head(20))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig(model_output_dir / "feature_importance_adv.png")

    config.update({
        'adversarial_validation_result': {
            'score': clf.best_score,
            'feature_importances': feature_imp.set_index("feature").sort_values(by="value", ascending=False).head(20).to_dict()["value"]
        }
    })
    """

    # =========================================
    # === train model
    # =========================================
    logger.info("train model")

    # get folds
    folds_ids = get_folds_per_DTM(train)

    x_train, x_test = load_features(config, args.debug)

    # remove features
    remove_features = [c for c in x_test.columns if c.find("_D3_") != -1]
    logger.info(f"remove features: {remove_features}")
    remain = [c for c in x_test.columns if c not in remove_features]
    x_train, x_test = x_train[remain], x_test[remain]

    feature_name = x_test.columns
    logger.debug(f'number of features: {len(feature_name)}')

    model_name = config['model']['name']
    model = model_map[model_name]()
    models, oof_preds, test_preds, feature_importance, evals_results = model.cv(
        y_train, x_train, x_test, feature_name, folds_ids, config
    )
    config.update(evals_results)

    feature_imp = feature_importance.reset_index().rename(columns={"index": "feature", 0: "value"})
    plt.figure(figsize=(20, 10))
    sns.barplot(x='value', y='feature', data=feature_imp.sort_values(by='value', ascending=False).head(20))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig(model_output_dir / "feature_importance_model.png")

    # ============================================
    # === Make Submission
    # ============================================
    test["isFraud"] = test_preds
    sub = test[["TransactionID", "isFraud"]]
    sub.to_csv(model_output_dir/ "submission.csv", index=False, header=True)

    # ============================================
    # === Check test score
    # ============================================
    df_probing = pd.read_csv('data/interim/probing_toolbox/20190929_probing.csv').loc[:, ['TransactionID', 'data_type', 'Probing_isFraud']]
    df = pd.merge(df_probing, sub, on='TransactionID', how='left')

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

    config.update({
        'proving_result': {
            'public_score': public_score,
            'private_score': private_score,
        }
    })
    logger.info(f"bear's public score: {public_score}")
    logger.info(f"bear's private score: {private_score}")

    # ============================================
    # === Save
    # ============================================
    save_path = model_output_dir / 'output.json'
    save_json(config, save_path)
    np.save(model_output_dir/ "oof_preds.npy", oof_preds)


if __name__ == '__main__':
    main()
