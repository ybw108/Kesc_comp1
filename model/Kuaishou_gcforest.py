import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from GCForest import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split


def online(features):
    X = train[features].fillna(-1)
    y = train['label']
    X = np.array(X)
    y = np.array(y)
    gcf = gcForest(shape_1X=39, n_mgsRFtree=100, window=10, stride=2,
                 cascade_test_size=0.2, n_cascadeRF=4, n_cascadeRFtree=101, cascade_layer=np.inf,
                 min_samples_mgs=0.1, min_samples_cascade=0.1, tolerance=0.0,)
    gcf.fit(X, y)
    #gcf = gcForest(tolerance=0.0, min_samples_cascade=20)
    #_ = gcf.cascade_forest(train[features], train['label'])

    X_test = test[features].fillna(-1)
    X_test = np.array(X_test)
    pred_proba = gcf.predict_proba(X_test)
    test['predict'] = pred_proba[:, 1]
    # preds = np.argmax(tmp, axis=1)
    test[['user_id', 'predict']].to_csv('../result/gcforest_predict.csv', index=False)
    #tmp = np.mean(pred_proba, axis=0)
    #preds = np.argmax(tmp, axis=1)


if __name__ == '__main__':
    train_1 = pd.read_csv('../data/train_1.csv', index_col=False)
    train_2 = pd.read_csv('../data/train_2.csv', index_col=False)
    train = pd.concat([train_1, train_2])
    test = pd.read_csv('../data/test.csv', index_col=False)
    features = [c for c in train if
                c not in ['label', 'user_id', 'launch_diff_target_day', 'act_diff_target_day', 'create_diff_target_day', 'continuous_launch_ratio', 'continuous_launch_times'
                          'day_act_var/n', 'last_second_diff_var/n', 'last_second_diff_var', 'last_second_diff_max', 'last_second_diff_min', 'last_second_diff_avg',
                          'last_second_diff', 'last_avg_diff',
                          'last_second_trend', 'second_trend_count', 'second_trend_ratio', 'last_avg_trend', 'avg_trend_count', 'avg_trend_ratio',
                          'create_diff_max', 'create_diff_min', 'create_diff_var', 'create_diff_avg', 'total_create_count',
                          # 重要性小于 1%的特征
                          # 'create_in_1', 'create_in_3', 'create_in_5', 'create_in_7', 'create_in_9', 'create_in_11', 'create_in_14', 'launch_diff_min', 'launch_diff_max', 'launch_diff_median',
                          # 重要性高但导致过拟合？（目前未确定）的特征
                          'launch_day_var', 'launch_day_avg', 'avg_launch_after_reg',
                          'avg_act_after_reg', 'avg_adv_act_after_reg', 'device_reg_type',
                          # 还没有试过的特征
                          'launch_day_median', 'day_act_median', 'launch_diff_median', 'day_adv_act_median', 'day_act_max_distance', 'day_adv_act_max_distance',
                          'is_author', 'is_act_page4',
                          ]]
    # 'last_second_trend', 'last_avg_trend', 'second_trend_count', 'avg_trend_count', 'second_trend_ratio', 'avg_trend_ratio' 趋势特征组
    # 'launch_diff_min', 'total_launch_count', 'continuous_launch_ratio', 'last_launch_day', 'last_act_day', 'last_create_day',
    # 'last_second_diff_var/n', 'last_second_diff_var', 'last_second_diff_max', 'last_second_diff_min', 'last_second_diff_avg',  每日act数目差值特征组
    # offline(features)
    #online(features)

    result = pd.read_csv('../result/gcforest_predict.csv', index_col=False)
    result = result.sort_values(['predict'], ascending=False).head(24900)
    result['user_id'].to_csv('../result/gcforest_result.csv', index=False, header=False)

