import pandas as pd
import numpy as np
import gc
from math import log
from sklearn.preprocessing import LabelEncoder


def get_lx_day(now):
    k1 = np.array(now)
    k2 = np.where(np.diff(k1) == 1)[0]
    i = 0
    ans = []
    while i < len(k2)-1:
        l1 = 1
        while k2[i+1]-k2[i] == 1:
            l1 += 1
            i += 1
            if i == len(k2)-1:
                break
        if l1 == 1:
            i += 1
            ans.append(2)
        else:
            i += 1
            ans.append(l1 + 1)
    if len(k2) == 1:
        ans.append(2)
    return ans


def read_data(first_day, last_day):
    user_reg_temp = pd.read_csv('../data/user_reg.csv', index_col=False)
    act_temp = pd.read_csv('../data/act.csv', index_col=False)
    launch_temp = pd.read_csv('../data/launch.csv', index_col=False)
    create_temp = pd.read_csv('../data/create.csv', index_col=False)
    act_temp = act_temp.loc[(act_temp.day >= first_day) & (act_temp.day <= last_day)]
    launch_temp = launch_temp.loc[(launch_temp.day >= first_day) & (launch_temp.day <= last_day)]
    create_temp = create_temp.loc[(create_temp.day >= first_day) & (create_temp.day <= last_day)]

    return user_reg_temp, act_temp, launch_temp, create_temp


def reg_features(data, day):
    data = pd.merge(data, user_reg, how='left', on=['user_id'])
    # 合并出现次数过少的项
    data['register_type'] = data['register_type'].apply(lambda x: 9 if x >= 9 else x)
    # 注册时长
    data['register_length'] = day - data['register_day']

    return data


def launch_features(data, day):
    # last launch date
    last_launch = launch.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_launch_day'})
    data = pd.merge(data, last_launch, how='left', on=['user_id'])
    data['last_launch_distance'] = day - data['last_launch_day']
    data = data.fillna(-1)
    # launch times in X days
    for i in [1, 3, 5, 7, 9, 11, 14, 16]:  # 1,3,5,7
        temp = launch.loc[launch.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)

    # 启动时间差特征
    for i in [5, 7, 9, 11, 14, 16]:
        launch_temp = launch.loc[launch.day >= day-int(i)]
        launch_temp['last_launch_day'] = launch_temp.sort_values(['user_id', 'day']).groupby(['user_id'])['day'].shift(1)
        launch_temp['launch_diff'] = launch_temp['day'] - launch_temp['last_launch_day'] - 1
        del(launch_temp['last_launch_day'])
        temp = launch_temp.groupby(['user_id'])['launch_diff'].agg({'launch_diff_max_in_'+str(i): 'max', 'launch_diff_min_in_'+str(i): 'min', 'launch_diff_avg_in_'+str(i): 'sum', 'launch_diff_var_in_'+str(i): 'var', 'total_launch_count_in_'+str(i): 'size'}).reset_index()
        temp['launch_diff_avg_in_'+str(i)] = temp['launch_diff_avg_in_'+str(i)]/temp['total_launch_count_in_'+str(i)]   # 这里要不要减一？？？
        data = pd.merge(data, temp, how='left', on=['user_id'])

    # # 连续启动特征
    # temp = launch.groupby(['user_id'])['day'].apply(get_lx_day).reset_index()
    # temp['con_max'] = temp['day'].apply(lambda x: np.max(x) if len(x) > 0 else 0)
    # temp['con_min'] = temp['day'].apply(lambda x: np.min(x) if len(x) > 0 else 0)
    # temp['con_sum'] = temp['day'].apply(lambda x: np.sum(x) if len(x) > 0 else 0)
    # temp['con_avg'] = temp['day'].apply(lambda x: np.sum(x)/len(x) if len(x) > 0 else 0)
    # temp['con_std'] = temp['day'].apply(lambda x: np.std(x) if len(x) > 0 else 0)
    # del(temp['day'])
    # data = pd.merge(data, temp, how='left', on=['user_id'])

    # 启动日期距离考察日的距离特征
    launch['distance'] = day - launch['day']    # 日期or和考察日的距离？
    temp = launch.groupby(['user_id'])['distance'].agg(
        {'launch_day_median': 'median', 'launch_day_avg': 'sum', 'launch_day_var': 'var', 'launch_day_count': 'size'}).reset_index()
    temp['launch_day_avg'] = temp['launch_day_avg']/temp['launch_day_count']    # 日期的暴力加和除以启动次数
    del(temp['launch_day_count'])
    data = pd.merge(data, temp, how='left', on=['user_id'])

    # 考察日的前一天是否连续启动
    launch['last_launch_day'] = launch.sort_values(['user_id', 'day']).groupby(['user_id'])['day'].shift(1)
    launch['launch_diff'] = launch['day'] - launch['last_launch_day'] - 1
    del (launch['last_launch_day'])
    temp = launch.loc[(launch.day == 16) & (launch.launch_diff == 0)]
    data['is_in_con'] = 0
    data.loc[data.user_id.isin(temp.user_id), ['is_in_con']] = 1

    # 最近两次启动的时间差
    temp = launch.sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='last').rename(columns={'launch_diff': 'launch_diff_last'})
    data = pd.merge(data, temp[['user_id', 'launch_diff_last']], how='left', on=['user_id'])

    # # 注册和第一次启动的时间差
    # temp = launch.sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='first')
    # temp = pd.merge(temp, user_reg[['user_id', 'register_day']], how='left', on=['user_id'])
    # temp['reg_launch_diff'] = temp['register_day'] - temp['day']
    # data = pd.merge(data, temp[['user_id', 'reg_launch_diff']], how='left', on=['user_id'])


    data = data.fillna(-1)
    return data


def act_features(data, day):
    last_act = act[['user_id', 'day']].sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_act_day'})
    data = pd.merge(data, last_act, how='left', on=['user_id'])
    data['last_act_distance'] = day - data['last_act_day']
    data = data.fillna(-1)

    # 用户N天内的act特征
    for i in [1, 3, 5, 7, 9, 11, 14, 16]:
        temp = act.loc[act.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'act_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
        temp = act.loc[(act.day >= day-int(i)) & (act.action_type >= 1)].groupby(['user_id']).size().reset_index().rename(columns={0: 'adv_act_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
        data['adv_act_ratio_in_'+str(i)] = data['adv_act_in_'+str(i)]/data['act_in_'+str(i)]  # n天内的进阶互动率
        for p in [0, 1, 2, 3, 4]:
            act1 = act.loc[act.page == int(p)]
            temp = act1.loc[act1.day >= day - int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'page_' + str(p) + '_act_in_' + str(i)})
            data = pd.merge(data, temp, how='left', on=['user_id'])
            # data['page_' + str(p) + '_ratio_in_' + str(i)] = data['page_' + str(p) + '_act_in_' + str(i)] / data['act_in_' + str(i)]
    data = data.fillna(0)

    for i in [5, 7, 9, 11, 14, 16]:
        # 每日act数目特征
        act_temp = act.loc[act.day >= day - int(i)]
        temp = act_temp.groupby(['user_id', 'day']).size().rename('day_act_times').reset_index()
        act_temp = pd.merge(act_temp, temp, how='left', on=['user_id', 'day'])
        temp = act_temp.drop_duplicates(['user_id', 'day']).groupby(['user_id'])['day_act_times'].agg({'day_act_max_in_'+str(i): 'max', 'day_act_min_in_'+str(i): 'min',
                                                                                                       'day_act_avg_in_'+str(i): 'sum', 'day_act_median_in_'+str(i): 'median',
                                                                                                       'day_act_var_in_'+str(i): 'var', 'act_day_count_in_'+str(i): 'size'}).reset_index()
        temp['day_act_avg_in_'+str(i)] = temp['day_act_avg_in_'+str(i)] / temp['act_day_count_in_'+str(i)]
        data = pd.merge(data, temp, how='left', on=['user_id'])
        # adv_act数目特征
        adv_act_temp = act.loc[(act.day >= day - int(i)) & (act.action_type >= 1)]
        temp = adv_act_temp.groupby(['user_id', 'day']).size().rename('day_act_times').reset_index()
        adv_act_temp = pd.merge(adv_act_temp, temp, how='left', on=['user_id', 'day'])
        temp = adv_act_temp.drop_duplicates(['user_id', 'day']).groupby(['user_id'])['day_act_times'].agg({'day_adv_act_max_in_' + str(i): 'max', 'day_adv_act_min_in_' + str(i): 'min',
                                                                                                       'day_adv_act_avg_in_' + str(i): 'sum', 'day_adv_act_median_in_' + str(i): 'median',
                                                                                                       'day_adv_act_var_in_' + str(i): 'var', 'adv_act_day_count_in_' + str(i): 'size'}).reset_index()
        temp['day_adv_act_avg_in_'+str(i)] = temp['day_adv_act_avg_in_'+str(i)] / temp['adv_act_day_count_in_'+str(i)]
        data = pd.merge(data, temp, how='left', on=['user_id'])

    # # 发生act的日期时间差特征
    # temp = act_temp.drop_duplicates(['user_id', 'day'], keep='last').sort_values(['user_id', 'day'])
    # temp['last_act_day'] = temp.groupby(['user_id'])['day'].shift(1)
    # temp['act_diff'] = temp['day'] - temp['last_act_day'] - 1
    # del(temp['last_act_day'])
    # temp = temp.groupby(['user_id'])['act_diff'].agg({'act_diff_max': 'max', 'act_diff_min': 'min', 'act_diff_avg': 'sum', 'act_diff_var': 'var', 'total_act_count': 'size'}).reset_index()
    # temp['act_diff_avg'] = temp['act_diff_avg']/temp['total_act_count']
    # del(temp['total_act_count'])
    # data = pd.merge(data, temp, how='left', on=['user_id'])


    # # 注册时间内平均每天adv_act数目
    # temp = pd.merge(temp, data[['user_id', 'register_length']], how='left', on=['user_id'])
    # temp['register_length'] = temp['register_length'].apply(lambda x: 16 if x > 16 else x)
    # temp['avg_adv_act_after_reg'] = temp['day_adv_act_avg'] / temp['register_length']
    # temp['day_adv_act_avg'] = temp['day_adv_act_avg'] / temp['act_adv_day_count']
    # # temp['day_act_var/n'] = temp['day_act_var'] / temp['act_day_count']
    # del (temp['register_length'])
    # data = pd.merge(data, temp, how='left', on=['user_id'])
    # data = pd.merge(data, temp2[['user_id', 'day_adv_act_max_distance']], how='left', on=['user_id'])
    # data['day_adv_act_max_distance'] = day - data['day_adv_act_max_distance']
    # data['adv_act_ratio'] = data['day_adv_act_avg']/data['day_act_avg']
    #
    # # 用户在每个page上的act特征
    # for p in [0, 1, 2, 3]:
    #     act1 = act.loc[act.page == int(p)]
    #     for i in [1, 3, 5, 7, 9, 11, 14]:
    #         temp = act1.loc[act1.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'act_in_'+str(i)+'_page'+str(p)})
    #         data = pd.merge(data, temp, how='left', on=['user_id'])
    #     data = data.fillna(0)
    #
    #     # 每日act数目特征
    #     temp = act1.groupby(['user_id', 'day']).size().rename('day_act_times'+'_page'+str(p)).reset_index()
    #     act_temp = pd.merge(act1, temp, how='left', on=['user_id', 'day'])
    #     # temp2 = act_temp.sort_values(['user_id', 'day']).drop_duplicates(['user_id', 'day'], keep='last')
    #     # temp3 = temp2.reset_index(drop=True).groupby(['user_id'])['day_act_times'].idxmax()                  # act 最大日的索引
    #     # temp2 = temp2.iloc[temp3]
    #     # temp2['day_act_max_distance'] = day - temp2['day']
    #     temp = act_temp.drop_duplicates(['user_id', 'day']).groupby(['user_id'])['day_act_times'+'_page'+str(p)].agg({'day_act_max'+'_page'+str(p): 'max',
    #             'day_act_min'+'_page'+str(p): 'min', 'day_act_avg'+'_page'+str(p): 'sum', 'day_act_median'+'_page'+str(p): 'median', 'day_act_var'+'_page'+str(p): 'var', 'act_day_count'+'_page'+str(p): 'size'}).reset_index()
    #
    #     # 注册时间内平均每天act数目
    #     temp = pd.merge(temp, data[['user_id', 'register_length']], how='left', on=['user_id'])
    #     temp['register_length'] = temp['register_length'].apply(lambda x: 16 if x > 16 else x)
    #     temp['avg_act_after_reg'+'_page'+str(p)] = temp['day_act_avg'+'_page'+str(p)] / temp['register_length']
    #     temp['day_act_avg'+'_page'+str(p)] = temp['day_act_avg'+'_page'+str(p)] / temp['act_day_count'+'_page'+str(p)]
    #     # temp['day_act_var/n'] = temp['day_act_var'] / temp['act_day_count']
    #     del(temp['register_length'])
    #     data = pd.merge(data, temp, how='left', on=['user_id'])
    #     #data = pd.merge(data, temp2[['user_id', 'day_act_max_distance']], how='left', on=['user_id'])
    #     #data['day_act_max_distance'] = day - data['day_act_max_distance']
    #     data['page'+str(p)+'_ratio'] = (data['day_act_avg_page'+str(p)]*data['act_day_count_page'+str(p)])/(data['day_act_avg']*data['act_day_count'])
    #

    # # 最近两次act/adv_act的时间差  没写完
    # temp = act.sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='last').rename(columns={'launch_diff': 'launch_diff_last'})
    # data = pd.merge(data, temp[['user_id', 'launch_diff_last']], how='left', on=['user_id'])

    # 注册和第一次act/adv_act的时间差
    temp = act.sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='first')
    temp = pd.merge(temp, user_reg[['user_id', 'register_day']], how='left', on=['user_id'])
    temp['reg_act_diff'] = temp['register_day'] - temp['day']
    data = pd.merge(data, temp[['user_id', 'reg_act_diff']], how='left', on=['user_id'])

    temp = act.loc[act.action_type >= 1].sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='first')
    temp = pd.merge(temp, user_reg[['user_id', 'register_day']], how='left', on=['user_id'])
    temp['reg_adv_act_diff'] = temp['register_day'] - temp['day']
    data = pd.merge(data, temp[['user_id', 'reg_adv_act_diff']], how='left', on=['user_id'])

    data = data.fillna(-1)
    gc.collect()
    return data


def create_features(data, day):
    last_create = create.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id','day': 'last_create_day'})
    data = pd.merge(data, last_create, how='left', on=['user_id'])
    data['last_create_distance'] = day - data['last_create_day']
    data = data.fillna(-1)
    for i in [1, 3, 5, 7, 9, 11, 14, 16]:
        temp = create.loc[create.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'create_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    # # create时间差特征
    # create['last_create_day'] = create.sort_values(['user_id', 'day']).groupby(['user_id'])['day'].shift(1)
    # create['create_diff'] = create['day'] - create['last_create_day'] - 1
    # del(create['last_create_day'])
    # temp = create.groupby(['user_id'])['create_diff'].agg({'create_diff_max': 'max', 'create_diff_min': 'min', 'create_diff_avg': 'sum', 'create_diff_var': 'var', 'total_create_count': 'size'}).reset_index()
    # temp['create_diff_avg'] = temp['create_diff_avg']/temp['total_create_count']
    # data = pd.merge(data, temp, how='left', on=['user_id'])
    # data = data.fillna(-1)

    return data


if __name__ == '__main__':
    user_reg, act, launch, create = read_data(1, 16)  # 1号到16号的用户启动、行为、上传日志
    train_1 = pd.read_csv('../data/train_1_list.csv', index_col=False)
    train_1 = reg_features(train_1, 17)
    train_1 = launch_features(train_1, 17)            # 输入考察日第一天的日期
    train_1 = create_features(train_1, 17)
    train_1 = act_features(train_1, 17)
    train_1.to_csv('../data/train_1.csv', index=False)

    user_reg, act, launch, create = read_data(8, 23)  # 8号到23号的用户启动、行为、上传日志
    train_2 = pd.read_csv('../data/train_2_list.csv', index_col=False)
    train_2 = reg_features(train_2, 24)
    train_2 = launch_features(train_2, 24)
    train_2 = create_features(train_2, 24)
    train_2 = act_features(train_2, 24)
    train_2.to_csv('../data/train_2.csv', index=False)

    user_reg, act, launch, create = read_data(15, 30)
    test = pd.read_csv('../data/test_list.csv', index_col=False)
    test = reg_features(test, 31)
    test = launch_features(test, 31)
    test = create_features(test, 31)
    test = act_features(test, 31)
    test.to_csv('../data/test.csv', index=False)