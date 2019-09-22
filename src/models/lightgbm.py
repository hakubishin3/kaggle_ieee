import lightgbm as lgb
from pathlib import Path
from .base import Base_Model


class LightGBM(Base_Model):
    def fit(self, x_train, y_train, x_valid, y_valid, config):
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        lgb_model_params = config["model"]["model_params"]
        lgb_train_params = config["model"]["train_params"]
        model = lgb.train(
            params=lgb_model_params,
            train_set=d_train,
            valid_sets=[d_valid],
            valid_names=['valid'],
            # categorical_feature = [col for col in x_train.columns if col.find("Label_En") != -1],
            **lgb_train_params
        )
        best_score = dict(model.best_score)
        return model, best_score
    
    def get_best_iteration(self, model):
        return model.best_iteration
    
    def predict(self, model, features):
        return model.predict(features)
        
    def get_feature_importance(self, model):
        return model.feature_importance(importance_type='gain')
