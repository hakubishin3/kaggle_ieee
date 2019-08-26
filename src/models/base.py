import numpy as np
import pandas as pd
from abc import abstractmethod
from sklearn.metrics import roc_auc_score


class Base_Model(object):
    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid, config):
        raise NotImplementedError
    
    @abstractmethod
    def get_best_iteration(self, model):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, features):
        raise NotImplementedError
        
    @abstractmethod
    def get_feature_importance(self, model):
        raise NotImplementedError    
        
    def cv(self, y_train, train_features, test_features, feature_name, folds_ids, config):
        # initialize
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        importances = pd.DataFrame(index=feature_name)
        best_iteration = 0
        cv_score_list = []
        models = []

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            # get train data and valid data
            x_trn = train_features.iloc[trn_idx]
            y_trn = y_train[trn_idx]
            x_val = train_features.iloc[val_idx]
            y_val = y_train[val_idx]
            
            # train model
            model, best_score = self.fit(x_trn, y_trn, x_val, y_val, config)
            cv_score_list.append(best_score)
            models.append(model)
            best_iteration += self.get_best_iteration(model) / len(folds_ids)
    
            # predict out-of-fold and test
            oof_preds[val_idx] = self.predict(model, x_val)
            test_preds += self.predict(model, test_features) / len(folds_ids)

            # get feature importances
            importances_tmp = pd.DataFrame(
                self.get_feature_importance(model),
                columns=[f'gain_{i_fold+1}'],
                index=feature_name
            )
            importances = importances.join(importances_tmp, how='inner')

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        # print oof score
        oof_score = roc_auc_score(y_train, oof_preds)
        print(f'oof score: {oof_score}')

        evals_results = {"evals_result": {
            "oof_score": oof_score,
            "cv_score": {f"cv{i+1}": cv_score for i, cv_score in enumerate(cv_score_list)},
            "n_data": len(train_features),
            "best_iteration": best_iteration,
            "n_features": len(train_features.columns),
            "feature_importance": feature_importance.sort_values(ascending=False).to_dict()
        }}

        return models, oof_preds, test_preds, feature_importance, evals_results
