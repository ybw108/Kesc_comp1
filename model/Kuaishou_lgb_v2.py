import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics


def offline(features):
    gbm = lgb.LGBMClassifier(objective='binary', n_estimators=2000, seed=2018)
    model = gbm.fit(train_1[features], train_1['label'], feature_name=features, categorical_feature=['register_type'], eval_set=[(train_2[features], train_2['label'])],
                    eval_metric='auc', early_stopping_rounds=150)
    best_score = model.best_score_['valid_0']['auc']
    best_iteration = model.best_iteration_
    feat_imp = model.feature_importances_

    y_pred = gbm.predict_proba(train_2[features], num_iteration=model.best_iteration_)[:, 1]
    y_pred = pd.DataFrame(y_pred, columns=['predicted_score'])
    top = y_pred.sort_values(['predicted_score'], ascending=False).head(int(len(y_pred)*0.577137))
    y_pred.loc[y_pred.predicted_score.isin(top.predicted_score), ['predicted_score']] = 1
    y_pred = y_pred['predicted_score'].apply(lambda x: 0 if x < 1 else x)
    f1_score = metrics.f1_score(train_2['label'], y_pred)

    used_features = [i for i in train[features].columns]
    feat_imp = pd.Series(feat_imp.reshape(-1), used_features).sort_values(ascending=False)
    feat_imp = (feat_imp/feat_imp.sum())*100
    print(feat_imp, 'number_of_features: ', len(feat_imp))
    feat_imp.to_csv('../result/feat_imp.csv')
    print('auc: ' + str(best_score))
    print('f1_score: ', str(f1_score))
    print('iteration: ' + str(best_iteration))


def online(features):
    # train = pd.read_csv('../data/train.csv', index_col=False)
    test = pd.read_csv('../data/test.csv', index_col=False)
    gbm = lgb.LGBMClassifier(objective='binary', seed=2018)

    gbm.fit(train[features], train['label'], feature_name=features, categorical_feature=['register_type'])
    test['predicted_score'] = gbm.predict_proba(test[features])[:, 1]

    # test = test[test['predicted_score'] >= 0.396]
    test = test.sort_values(['predicted_score'], ascending=False).head(24900)
    test[['user_id']].to_csv('../result/result.csv', header=False, index=False, sep=' ')


if __name__ == '__main__':
    train_1 = pd.read_csv('../data/train_1.csv', index_col=False)
    train_2 = pd.read_csv('../data/train_2.csv', index_col=False)
    train = pd.concat([train_1, train_2])

    features = [c for c in train if
                c not in ['label', 'user_id', 'launch_diff_target_day', 'act_diff_target_day', 'create_diff_target_day','device_reg_type','continuous_launch_times','continuous_launch_ratio',
                          'last_second_diff_var', 'last_second_diff_max', 'last_second_diff_min', 'last_second_diff_avg',
                          'last_second_diff', 'last_avg_diff',
                          'last_second_trend', 'second_trend_count', 'second_trend_ratio', 'last_avg_trend', 'avg_trend_count', 'avg_trend_ratio',
                          'create_diff_max', 'create_diff_min', 'create_diff_var', 'create_diff_avg', 'total_create_count',
                          # 重要性小于 1%的特征
                          #'create_in_1', 'create_in_3', 'create_in_5', 'create_in_7', 'create_in_9', 'create_in_11', 'create_in_14', 'launch_diff_min', 'launch_diff_max', 'launch_diff_median',
                          'act_diff_max', 'act_diff_min', 'act_diff_var', 'act_diff_avg',
                          # 重要性高但导致过拟合？（目前未确定）的特征
                          'launch_day_var', 'launch_day_avg',
                          'avg_act_after_reg', 'avg_launch_after_reg','avg_adv_act_after_reg',
                          # 还没有试过的特征
                          'launch_day_median', 'day_act_median', 'launch_diff_median', 'day_adv_act_median', 'day_act_max_distance', 'day_adv_act_max_distance',
                           'is_act_page4',#'is_author',
                          ]]
    # 'last_second_trend', 'last_avg_trend', 'second_trend_count', 'avg_trend_count', 'second_trend_ratio', 'avg_trend_ratio' 趋势特征组
    # 'launch_diff_min', 'total_launch_count', 'continuous_launch_ratio', 'last_launch_day', 'last_act_day', 'last_create_day',
    # 'last_second_diff_var/n', 'last_second_diff_var', 'last_second_diff_max', 'last_second_diff_min', 'last_second_diff_avg',  每日act数目差值特征组
    offline(features)
    online(features)



