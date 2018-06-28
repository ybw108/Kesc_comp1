import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics


def offline(features):
    train_2 = pd.concat([train_1a,  train_2a])
    train_1 = pd.concat([train_1b,  train_2b])

    gbm = lgb.LGBMClassifier(objective='binary', n_estimators=2000, seed=2018)
    model = gbm.fit(train_1[features], train_1['label'], feature_name=features, categorical_feature=['register_type'],
                    eval_set=[(train_2[features], train_2['label'])], eval_metric='auc', early_stopping_rounds=150)
    best_score = model.best_score_['valid_0']['auc']
    best_iteration = model.best_iteration_
    feat_imp = model.feature_importances_

    # train_2['predicted_label'] = (gbm.predict_proba(train_2[features], num_iteration=model.best_iteration_)[:, 1] >= 0.416).astype(int)
    train_2['predicted_score'] = gbm.predict_proba(train_2[features], num_iteration=model.best_iteration_)[:, 1]
    top = train_2.sort_values(['predicted_score'], ascending=False)
    min_res = 20000
    max_res = 40000
    pp = len(top[top.label == 1])
    dict = {}
    while(min_res <= max_res):
        t_data = top[:min_res]
        one = len(t_data[t_data.label == 1])
        s_l = len(t_data)
        precision = one/s_l
        recall = one/pp
        print(min_res, (2*precision*recall)/(precision+recall))
        dict[min_res] = (2 * precision * recall) / (precision + recall)
        min_res += 200
    best_offline = max(zip(dict.values(), dict.keys()))
    t_data = top[:best_offline[1]]
    train_2['predicted_label'] = (train_2['predicted_score'] >= t_data.iloc[-1]['predicted_score']).astype(int)
    train_2.loc[train_2.label != train_2.predicted_label].to_csv('../result/cuofen.csv', index=False)

    used_features = [i for i in train[features].columns]
    feat_imp = pd.Series(feat_imp.reshape(-1), used_features).sort_values(ascending=False)
    feat_imp = (feat_imp/feat_imp.sum())*100
    print(feat_imp, 'number_of_features: ', len(feat_imp))
    feat_imp.to_csv('../result/feat_imp.csv')
    print('auc: ' + str(best_score))
    print('best_f1: ', best_offline[0])
    print('best_f1_cutoff: ', t_data.iloc[-1]['predicted_score'])
    print('iteration: ' + str(best_iteration))


def online(features):
    test = pd.read_csv('../data/test.csv', index_col=False)
    gbm = lgb.LGBMClassifier(objective='binary', seed=2018)
    gbm.fit(train[features], train['label'], feature_name=features, categorical_feature=['register_type'])
    test['predicted_score'] = gbm.predict_proba(test[features])[:, 1]

    test = test[test['predicted_score'] >= 0.4]
    # test = test.sort_values(['predicted_score'], ascending=False).head(24900)
    test[['user_id']].to_csv('../result/result.csv', header=False, index=False, sep=' ')


if __name__ == '__main__':
    train_1a = pd.read_csv('../data/train_1a.csv', index_col=False)
    train_2a = pd.read_csv('../data/train_2a.csv', index_col=False)
    train_1b = pd.read_csv('../data/train_1b.csv', index_col=False)
    train_2b = pd.read_csv('../data/train_2b.csv', index_col=False)
    train = pd.concat([train_1a,  train_2a, train_1b, train_2b])

    features = [c for c in train if
                c not in ['label', 'user_id', 'device_reg_type',# 'register_day', 'register_length',
                          'launch_in_1', 'launch_in_2', 'launch_in_3', 'launch_in_5', 'launch_in_7', 'launch_in_9', 'launch_in_11', 'launch_in_14', 'launch_in_16',
                          'last_create_distance', 'last_act_distance', 'last_launch_distance',
                          'avg_act_after_reg', 'avg_launch_after_reg',
                          'last_launch_day', 'last_create_day', 'last_act_day',
                          ]]
    offline(features)
    #online(features)



