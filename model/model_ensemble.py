import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import math


def offline(features):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)
    X = train[features]
    y = train['label']
    best_score = []
    best_iteration = []
    f1_score = []
    best_cutoff = 0
    feat_imp = np.zeros((1, train[features].shape[1]))
    y_valid = pd.DataFrame()
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        gbm1 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=2000, seed=2018)
        model1 = gbm1.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=150)
        gbm2 = lgb.LGBMClassifier(objective='binary', n_estimators=2000, seed=2018)
        model2 = gbm2.fit(X_train, y_train, feature_name=features, categorical_feature=['register_type'], eval_set=[(X_test, y_test)],
                         eval_metric='auc', early_stopping_rounds=150)

        y_test = y_test.to_frame()
        y_test['predict1'] = gbm1.predict_proba(X_test, ntree_limit=model1.best_iteration)[:, 1]
        y_test['predict2'] = gbm2.predict_proba(X_test, num_iteration=model2.best_iteration_)[:, 1]
        y_valid = pd.concat([y_valid, y_test])

    irange = np.arange(0.3, 0.7, 0.01)
    ensemble_score = {}
    for i in irange:
        y_valid['final_predict'] = ((float(i) * y_valid['predict1'] + (1-float(i)) * y_valid['predict2']) >= 0.396).astype(int)
        f1_score = metrics.f1_score(y_valid['label'], y_valid['final_predict'])
        ensemble_score[i] = f1_score
    print(max(zip(ensemble_score.values(), ensemble_score.keys())))


def online(features):
    # train = pd.read_csv('../data/train.csv', index_col=False)
    test = pd.read_csv('../data/test.csv', index_col=False)
    gbm1 = xgb.XGBClassifier(objective='binary:logistic', seed=2018)
    gbm1.fit(train[features], train['label'])
    test['predicted_score1'] = gbm1.predict_proba(test[features])[:, 1]

    gbm2 = lgb.LGBMClassifier(objective='binary', seed=2018)
    gbm2.fit(train[features], train['label'], feature_name=features, categorical_feature=['register_type'])
    test['predicted_score2'] = gbm2.predict(test[features])[:, 1]

    test = test[0.5 * test['predicted_score1'] + 0.5 * test['predicted_score2'] >= 0.396]
    test[['user_id']].to_csv('../result/result.csv', header=False, index=False, sep=' ')


def result_fusion(result1, result2):
    result_temp = pd.merge(result2, result1, how='left', on=['user_id'])
    result_temp = result_temp.fillna(-1)
    result_temp['final_predicted'] = result_temp.apply(lambda row: 0.45*row['predicted_score_x']+0.55*row['predicted_score_y'] if row['predicted_score_y'] !=-1 else row['predicted_score_x'],axis=1)
    # result_temp = result_temp[result_temp.final_predicted >= 0.396]
    result_temp = result_temp.sort_values(['final_predicted'], ascending=False).head(24800)
    return result_temp


def result_sigmoid2(result1, result2, sub_number):
    result_temp = pd.merge(result2, result1, how='left', on=['user_id'])
    result_temp = result_temp.fillna(-1)
    result_temp['predicted_score_y'] = result_temp.apply(lambda row: row['predicted_score_x'] if row['predicted_score_y'] == -1 else row['predicted_score_y'], axis=1)

    result_temp['predicted_score'] = result_temp['predicted_score_x'].apply(lambda x: math.log(x / (1 - x))) + result_temp['predicted_score_y'].apply(lambda x: math.log(x / (1 - x)))
    result_temp['predicted_score'] = (result_temp['predicted_score'] / 2).apply(lambda x: 1 / (1 + math.exp(-x)))

    result_temp[['user_id', 'predicted_score']].to_csv('../result/fusion_teammate.csv', index=False)
    # result_temp = result_temp[result_temp.final_predicted >= 0.396]
    result_temp = result_temp.sort_values(['predicted_score'], ascending=False).head(int(sub_number))
    return result_temp


def result_sigmoid3(result1, result2, result3):
    result_temp = pd.merge(result2, result1, how='left', on=['user_id'])
    result_temp = pd.merge(result_temp, result3, how='left', on=['user_id'])
    result_temp = result_temp.fillna(-1)
    result_temp['predicted_score_y'] = result_temp.apply(lambda row: row['predicted_score_x'] if row['predicted_score_y'] == -1 else row['predicted_score_y'], axis=1)
    result_temp['predicted_score'] = result_temp.apply(lambda row: row['predicted_score_x'] if row['predicted_score'] == -1 else row['predicted_score'], axis=1)

    result_temp['final_predicted'] = result_temp['predicted_score_x'].apply(lambda x: math.log(x / (1 - x))) + result_temp['predicted_score_y'].apply(lambda x: math.log(x / (1 - x))) + result_temp['predicted_score'].apply(lambda x: math.log(x / (1 - x)))
    result_temp['final_predicted'] = (result_temp['final_predicted'] / 3).apply(lambda x: 1 / (1 + math.exp(-x)))

    # result_temp = result_temp[result_temp.final_predicted >= 0.396]
    result_temp = result_temp.sort_values(['final_predicted'], ascending=False).head(24800)
    return result_temp


if __name__ == '__main__':
    # train_1 = pd.read_csv('../data/train_1.csv', index_col=False)
    # train_2 = pd.read_csv('../data/train_2.csv', index_col=False)
    # train = pd.concat([train_1, train_2])
    #result_1 = pd.read_csv('../result/lgb_highest.csv', index_col=False)
    #result_2 = pd.read_csv('../result/submit_xgb_818137_b.csv', index_col=False)


    result_1 = pd.read_csv('../result/fusion_highest.csv', index_col=False)
    result_2 = pd.read_csv('../result/fusion_teammate.csv', index_col=False)
    # result_3 = pd.read_csv('../result/xgb_highest.csv', index_col=False)
    result = result_sigmoid2(result_1, result_2, 24800)
    result[['user_id']].to_csv('../result/f_result.csv', header=False, index=False, sep=' ')

