import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics


def train_lgb(features, col_sample, row_sample, r_seed):
    # train = pd.read_csv('../data/train.csv', index_col=False)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)
    X = train[features]
    y = train['label']
    best_score = []
    best_iteration = []
    f1_score = []
    feat_imp = np.zeros((1, train[features].shape[1]))
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        gbm = lgb.LGBMClassifier(objective='binary', n_estimators=2000, seed=int(r_seed), colsample_bytree=col_sample, subsample=row_sample)  # , num_leaves=13, max_depth=4, max_bin=90, colsample_bytree=0.9
        model = gbm.fit(X_train, y_train, feature_name=features, categorical_feature=['register_type'], eval_set=[(X_test, y_test)],
                        eval_metric='auc', early_stopping_rounds=150, verbose=False)
        best_score.append(model.best_score_['valid_0']['auc'])
        best_iteration.append(model.best_iteration_)
        feat_imp_temp = model.feature_importances_
        feat_imp = feat_imp + feat_imp_temp

        y_pred = (gbm.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1] >= 0.396).astype(int)
        f1_score.append(metrics.f1_score(y_test, y_pred))

    used_features = [i for i in train[features].columns]
    feat_imp = pd.Series(feat_imp.reshape(-1), used_features).sort_values(ascending=False)
    feat_imp = (feat_imp/feat_imp.sum())*100
    print(feat_imp, 'number_of_features: ', len(feat_imp))
    # feat_imp.to_csv('../result/feat_imp.csv')

    avg_best_auc = sum(best_score)/len(best_score)
    avg_best_f1 = sum(f1_score)/len(f1_score)
    avg_best_iteration = sum(best_iteration)/len(best_iteration)
    return avg_best_auc, avg_best_f1, avg_best_iteration


def multi(r_seed, col_sample, row_sample,number):
    auc_dict = {}
    f1_dict = {}
    iteration_dict = {}
    for i in range(0, int(number)):
        auc, f1, iteration = train_lgb(features, col_sample, row_sample, int(r_seed)+int(i))
        auc_dict[int(r_seed)+int(i)] = auc
        f1_dict[int(r_seed)+int(i)] = f1
        iteration_dict[int(r_seed)+int(i)] = iteration
    return auc_dict, f1_dict, iteration_dict


def online(features):
    # train = pd.read_csv('../data/train.csv', index_col=False)
    test = pd.read_csv('../data/test.csv', index_col=False)
    for i in range(2018, 2028):
        gbm = lgb.LGBMClassifier(objective='binary', seed=int(i), colsample_bytree=0.9, subsample=0.9)
        gbm.fit(train[features], train['label'], feature_name=features, categorical_feature=['register_type'])
        test['predicted_score'+str(i-2017)] = gbm.predict_proba(test[features])[:, 1]

    test['predicted_score'] = 0
    for i in range(1, 11):
        test['predicted_score'] = test['predicted_score'] + test['predicted_score'+str(i)]
    test['predicted_score'] = test['predicted_score'] / 10
    test = test[test['predicted_score'] >= 0.396]
    # test = test.sort_values(['predicted_score'], ascending=False).head(24800)
    test[['user_id']].to_csv('../result/result.csv', header=False, index=False, sep=' ')


if __name__ == '__main__':
    train_1a = pd.read_csv('../data/train_1a.csv', index_col=False)
    train_2a = pd.read_csv('../data/train_2a.csv', index_col=False)
    train_1b = pd.read_csv('../data/train_1b.csv', index_col=False)
    train_2b = pd.read_csv('../data/train_2b.csv', index_col=False)
    train = pd.concat([train_1a, train_2a, train_1b, train_2b])
    # train = pd.concat([train_1a, train_1b, train_2a,  train_2b])

    features = [c for c in train if
                c not in ['label', 'user_id', 'continuous_launch_ratio', 'continuous_launch_times',
                          'day_act_var/n', 'last_second_diff_var/n', 'last_second_diff_var', 'last_second_diff_max', 'last_second_diff_min', 'last_second_diff_avg',
                          'last_second_diff', 'last_avg_diff',
                          'last_second_trend', 'second_trend_count', 'second_trend_ratio', 'last_avg_trend', 'avg_trend_count', 'avg_trend_ratio',
                          'create_diff_max', 'create_diff_min', 'create_diff_var', 'create_diff_avg', 'total_create_count',
                          # 重要性小于 1%的特征
                          # 'create_in_1', 'create_in_3', 'create_in_5', 'create_in_7', 'create_in_9', 'create_in_11', 'create_in_14', 'launch_diff_min', 'launch_diff_max', 'launch_diff_median',
                          'act_diff_max', 'act_diff_min', 'act_diff_var', 'act_diff_avg',
                          # 重要性高但导致过拟合？（目前未确定）的特征
                          #'last_launch_day', 'last_act_day', 'last_create_day',
                          'launch_diff_target_day', 'act_diff_target_day', 'create_diff_target_day',
                          'launch_day_var', 'launch_day_avg', 'avg_launch_after_reg',
                          'avg_act_after_reg', 'avg_adv_act_after_reg', 'device_reg_type',
                          # 还没有试过的特征
                          'launch_day_median', 'day_adv_act_median', 'day_act_max_distance', 'day_adv_act_max_distance',
                          'is_author', 'is_act_page4',
                          'day_act_median', 'launch_diff_median', 'launch_diff', 'distance', 'create_diff',
                          ]]

    # offline
    # auc_dict = {}
    # f1_dict = {}
    # iteration_dict = {}
    # auc_dict, f1_dict, iteration_dict = multi(2018, 0.9, 0.9, 10)
    # print('auc: ' + str(sum(auc_dict.values())/len(auc_dict)))
    # print('F1: ' + str(sum(f1_dict.values()) / len(f1_dict)))
    # print('iteration: ' + str(sum(iteration_dict.values()) / len(iteration_dict)))

    online(features)



