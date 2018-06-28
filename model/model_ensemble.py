import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics


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


if __name__ == '__main__':
    # train_1 = pd.read_csv('../data/train_1.csv', index_col=False)
    # train_2 = pd.read_csv('../data/train_2.csv', index_col=False)
    # train = pd.concat([train_1, train_2])
    result_1 = pd.read_csv('../result/lgb_highest.csv', index_col=False)
    result_2 = pd.read_csv('../result/submit_xgb_817847_b.csv', index_col=False)
    result = result_fusion(result_1, result_2)
    result[['user_id']].to_csv('../result/f_result.csv', header=False, index=False, sep=' ')

    # features = [c for c in train if
    #             c not in ['label', 'user_id', 'launch_diff_target_day', 'act_diff_target_day', 'create_diff_target_day', 'continuous_launch_ratio', 'continuous_launch_times',
    #                       'day_act_var/n', 'last_second_diff_var/n', 'last_second_diff_var', 'last_second_diff_max', 'last_second_diff_min', 'last_second_diff_avg',
    #                       'last_second_diff', 'last_avg_diff',
    #                       'last_second_trend', 'second_trend_count', 'second_trend_ratio', 'last_avg_trend', 'avg_trend_count', 'avg_trend_ratio',
    #                       'create_diff_max', 'create_diff_min', 'create_diff_var', 'create_diff_avg', 'total_create_count',
    #                       # 重要性小于 1%的特征
    #                       # 'create_in_1', 'create_in_3', 'create_in_5', 'create_in_7', 'create_in_9', 'create_in_11', 'create_in_14', 'launch_diff_min', 'launch_diff_max', 'launch_diff_median',
    #                       # 重要性高但导致过拟合？（目前未确定）的特征
    #                       'launch_day_var', 'launch_day_avg', 'avg_launch_after_reg','device_reg_type',
    #                       'avg_act_after_reg', 'avg_adv_act_after_reg',
    #                       # 还没有试过的特征
    #                       'launch_day_median', 'day_act_median', 'launch_diff_median', 'day_adv_act_median', 'day_act_max_distance', 'day_adv_act_max_distance',
    #                       'is_author', 'is_act_page4',
    #                       ]]
    # 'last_second_trend', 'last_avg_trend', 'second_trend_count', 'avg_trend_count', 'second_trend_ratio', 'avg_trend_ratio' 趋势特征组
    # 'launch_diff_min', 'total_launch_count', 'continuous_launch_ratio', 'last_launch_day', 'last_act_day', 'last_create_day',
    # 'last_second_diff_var/n', 'last_second_diff_var', 'last_second_diff_max', 'last_second_diff_min', 'last_second_diff_avg',  每日act数目差值特征组
    #offline(features)