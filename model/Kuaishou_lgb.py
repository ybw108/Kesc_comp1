import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import metrics


def offline(features):
    #train = pd.read_csv('../data/train.csv', index_col=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=2018)
    X = train[features]
    y = train['label']
    best_score = []
    best_iteration = []
    f1_score = []
    best_cutoff = 0
    feat_imp = np.zeros((1, train[features].shape[1]))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        gbm = lgb.LGBMRegressor(objective='binary', n_estimators=2000, seed=2018)
        model = gbm.fit(X_train, y_train, feature_name=features, categorical_feature=['register_type'], eval_set=[(X_test, y_test)],
                        eval_metric='auc', early_stopping_rounds=150)
        best_score.append(model.best_score_['valid_0']['auc'])
        best_iteration.append(model.best_iteration_)
        feat_imp_temp = model.feature_importances_
        feat_imp = feat_imp + feat_imp_temp
        # if best_cutoff == 0:
        #     i_range = np.arange(0.35, 0.55, 0.01)
        #     for i in i_range:
        #         y_pred = (gbm.predict(X_test, num_iteration=model.best_iteration_) >= i).astype(int)
        #         f1 = metrics.f1_score(y_test, y_pred)
        #         if f1 > best_f1:
        #             best_f1 = f1
        #             best_cutoff = i
        # else:
        #     y_pred = (gbm.predict(X_test, num_iteration=model.best_iteration_) >= best_cutoff).astype(int)
        #     f1 = metrics.f1_score(y_test, y_pred)
        # f1_score.append(f1)

        y_pred = (gbm.predict(X_test, num_iteration=model.best_iteration_) >= 0.398).astype(int)
        f1_score.append(metrics.f1_score(y_test, y_pred))

    used_features = [i for i in train[features].columns]
    feat_imp = pd.Series(feat_imp.reshape(-1), used_features).sort_values(ascending=False)
    print(feat_imp, 'number_of_features: ', len(feat_imp))
    print(best_score, '\n', f1_score, '\n', best_iteration)
    print('average of best auc: ' + str(sum(best_score)/len(best_score)))
    #print('best_cutoff = ', best_cutoff)
    print('average of best f1_score: ', str(sum(f1_score)/len(f1_score)))
    print('average of best iteration: ' + str(sum(best_iteration)/len(best_iteration)))


def online(features):
    #train = pd.read_csv('../data/train.csv', index_col=False)
    test = pd.read_csv('../data/test.csv', index_col=False)
    gbm = lgb.LGBMRegressor(objective='binary', seed=2018)

    gbm.fit(train[features], train['label'], feature_name=features, categorical_feature=['register_type'])
    test['predicted_score'] = gbm.predict(test[features])

    test = test[test['predicted_score'] >= 0.398]
    test[['user_id']].to_csv('../result/result.csv', header=False, index=False, sep=' ')


if __name__ == '__main__':
    train_1 = pd.read_csv('../data/train_1.csv', index_col=False)
    train_2 = pd.read_csv('../data/train_2.csv', index_col=False)
    train = pd.concat([train_1, train_2])

    features = [c for c in train if
                c not in ['label', 'user_id', ]]

    offline(features)
    online(features)



