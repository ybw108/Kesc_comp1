import pandas as pd
import numpy as np
import gc
from math import log
from sklearn.preprocessing import LabelEncoder


def fill(row, tarday):
    if row['register_day'] < int(tarday):
        row = row.fillna(0)
    else:
        row = row.fillna(-1)
    return row


def read_data(first_day, last_day):
    act_temp = act.loc[(act.day >= first_day) & (act.day <= last_day)]
    launch_temp = launch.loc[(launch.day >= first_day) & (launch.day <= last_day)]
    create_temp = create.loc[(create.day >= first_day) & (create.day <= last_day)]

    return user_reg, act_temp, launch_temp, create_temp


def reg_features(data, day):
    data = pd.merge(data, user_reg, how='left', on=['user_id'])
    # 合并出现次数过少的项





    data['register_type'] = data['register_type'].apply(lambda x: 9 if x >= 9 else x)
    # 注册时长
    data['register_length'] = day - data['register_day']
    return data


def launch_features(data, day):
    # # launch times in X days
    # for i in [1, 2, 3, 5, 7, 9, 11, 14, 16]:  # 1,3,5,7
    #     temp = launch.loc[launch.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_in_'+str(i)})
    #     data = pd.merge(data, temp, how='left', on=['user_id'])
    # data = data.fillna(0)

    #序列化
    for i in range(1, 17):
        temp = launch.loc[launch.day == day - int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_on_day'+str(17 - int(i))})
        data = pd.merge(data, temp, how='left', on=['user_id'])
        data = data.apply(fill, axis=1, tarday=day - int(i))
    #带权加和
    for i in [3, 5, 7, 9, 11, 14, 16]:
        value = 0
        for j in range(1, i+1):
            data['wei_launch_on_day'+str(17 - j)] = data['launch_on_day'+str(17 - j)].apply(lambda x: x*int(17-j) if x>0 else 0)
            value = value + data['wei_launch_on_day'+str(17 - j)]
            del(data['wei_launch_on_day'+str(17 - j)])
        data['wlaunch_in_' + str(i)] = value
    data = data.fillna(0)

    # 启动时间差特征
    for i in [3, 5, 7, 9, 11, 14, 16]:
        launch_temp = launch.loc[launch.day >= day-int(i)]
        launch_temp['last_launch_day'] = launch_temp.sort_values(['user_id', 'day']).groupby(['user_id'])['day'].shift(1)
        launch_temp['launch_diff'] = launch_temp['day'] - launch_temp['last_launch_day'] - 1
        del(launch_temp['last_launch_day'])
        temp = launch_temp.groupby(['user_id'])['launch_diff'].agg({'launch_diff_max_in_'+str(i): 'max', 'launch_diff_min_in_'+str(i): 'min', 'launch_diff_mean_in_'+str(i): 'mean', 'launch_diff_var_in_'+str(i): 'var'}).reset_index()
        data = pd.merge(data, temp, how='left', on=['user_id'])

    # # 启动时间差特征
    # launch['last_launch_day'] = launch.sort_values(['user_id', 'day']).groupby(['user_id'])['day'].shift(1)
    # launch['launch_diff'] = launch['day'] - launch['last_launch_day'] - 1
    # del (launch['last_launch_day'])
    # temp = launch.groupby(['user_id'])['launch_diff'].agg({'launch_diff_max': 'max', 'launch_diff_min': 'min', 'launch_diff_mean': 'mean', 'launch_diff_var': 'var'}).reset_index()
    # data = pd.merge(data, temp, how='left', on=['user_id'])

    # 注册时间内平均每天启动次数
    temp = launch.sort_values(['user_id', 'day']).groupby(['user_id']).size().rename('total_launch_count').reset_index()
    temp = pd.merge(temp, data[['user_id', 'register_length']], how='left', on=['user_id'])
    temp['register_length'] = temp['register_length'].apply(lambda x: 16 if x > 16 else x)  # 注册时长超过窗口长度的，降为窗口长度
    temp['avg_launch_after_reg'] = temp['total_launch_count'] / temp['register_length']      # 注册后平均每天启动次数
    data = pd.merge(data, temp[['user_id', 'avg_launch_after_reg']], how='left', on=['user_id'])

    # 最后一次启动距离考察日时间差
    last_launch = launch.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_launch_day'})
    data = pd.merge(data, last_launch, how='left', on=['user_id'])
    data['last_launch_distance'] = day - data['last_launch_day']
    del(data['last_launch_day'])

    data = data.fillna(-1)
    return data


def act_features(data, day):
    # 用户N天内的act特征
    for i in [1, 2, 3, 5, 7, 9, 11, 14, 16]:
        act_temp = act.loc[act.day >= day-int(i)]
        temp = act_temp.groupby(['user_id']).size().reset_index().rename(columns={0: 'act_in_'+str(i)})
        temp['type_in_' + str(i)] = act_temp.groupby(['user_id'])['action_type'].nunique().rename('type_in_' + str(i)).reset_index()['type_in_' + str(i)]
        temp['page_in_'+str(i)] = act_temp.groupby(['user_id'])['page'].nunique().rename('page_in_'+str(i)).reset_index()['page_in_'+str(i)]
        temp['author_in_' + str(i)] = act_temp.groupby(['user_id'])['author_id'].nunique().rename('author_in_' + str(i)).reset_index()['author_in_' + str(i)]
        temp['video_in_' + str(i)] = act_temp.groupby(['user_id'])['video_id'].nunique().rename('video_in_' + str(i)).reset_index()['video_in_' + str(i)]
        data = pd.merge(data, temp, how='left', on=['user_id'])

        # # 用户对作者的重复点击情况
        # temp2 = act_temp.groupby(['user_id', 'day', 'author_id']).size().rename('act_on_author').reset_index()
        # temp2 = temp2.groupby(['user_id'])['act_on_author'].agg({'act_on_author_max_in_'+str(i): 'max', 'act_on_author_mean_in_'+str(i): 'mean', 'act_on_author_var_in_'+str(i): 'var'}).reset_index()
        # data = pd.merge(data, temp2, how='left', on=['user_id'])
        #
        # # 用户对视频的重复点击情况
        # temp2 = act_temp.groupby(['user_id', 'day', 'video_id']).size().rename('act_on_video').reset_index()
        # temp2 = temp2.groupby(['user_id'])['act_on_video'].agg({'act_on_video_max_in_' + str(i): 'max', 'act_on_video_mean_in_' + str(i): 'mean','act_on_video_var_in_' + str(i): 'var'}).reset_index()
        # data = pd.merge(data, temp2, how='left', on=['user_id'])
    data = data.fillna(0)

    # # 用户对作者的重复点击情况
    # temp2 = act.groupby(['user_id', 'day', 'author_id']).size().rename('act_on_author').reset_index()
    # temp2 = temp2.groupby(['user_id'])['act_on_author'].agg({'act_on_author_max': 'max', 'act_on_author_mean': 'mean', 'act_on_author_var': 'var'}).reset_index()
    # data = pd.merge(data, temp2, how='left', on=['user_id'])
    #
    # # 用户对视频的重复点击情况
    # temp2 = act.groupby(['user_id', 'day', 'video_id']).size().rename('act_on_video').reset_index()
    # temp2 = temp2.groupby(['user_id'])['act_on_video'].agg({'act_on_video_max': 'max', 'act_on_video_mean': 'mean', 'act_on_video_var': 'var'}).reset_index()
    # data = pd.merge(data, temp2, how='left', on=['user_id'])

    # 每日act数目特征
    for i in [3, 5, 7, 9, 11, 14, 16]:
        act_temp = act.loc[act.day >= day - int(i)]
        temp = act_temp.groupby(['user_id', 'day']).size().rename('day_act_times').reset_index()
        act_temp = pd.merge(act_temp, temp, how='left', on=['user_id', 'day'])
        temp = act_temp.drop_duplicates(['user_id', 'day']).groupby(['user_id'])['day_act_times'].agg({'day_act_max_in_'+str(i): 'max', 'day_act_min_in_'+str(i): 'min',
                                                                                                       'day_act_mean_in_'+str(i): 'mean', 'day_act_median_in_'+str(i): 'median',
                                                                                                       'day_act_var_in_'+str(i): 'var', 'act_day_count'+str(i): 'size'}).reset_index()
        data = pd.merge(data, temp, how='left', on=['user_id'])


    # temp = act.groupby(['user_id', 'day']).size().rename('day_act_times').reset_index()
    # act_temp = pd.merge(act, temp, how='left', on=['user_id', 'day'])
    # temp = act_temp.drop_duplicates(['user_id', 'day']).groupby(['user_id'])['day_act_times'].agg({'day_act_max': 'max', 'day_act_min': 'min', 'day_act_avg': 'sum', 'day_act_median': 'median', 'day_act_var': 'var', 'act_day_count': 'size'}).reset_index()
    # data = pd.merge(data, temp, how='left', on=['user_id'])

    last_act = act[['user_id', 'day']].sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_act_day'})
    data = pd.merge(data, last_act, how='left', on=['user_id'])
    data['last_act_distance'] = day - data['last_act_day']
    del (data['last_act_day'])

    # 注册时间内平均每天act次数
    temp = act.sort_values(['user_id', 'day']).groupby(['user_id']).size().rename('total_act_count').reset_index()
    temp = pd.merge(temp, data[['user_id', 'register_length']], how='left', on=['user_id'])
    temp['register_length'] = temp['register_length'].apply(lambda x: 16 if x > 16 else x)  # 注册时长超过窗口长度的，降为窗口长度
    temp['avg_act_after_reg'] = temp['total_act_count'] / temp['register_length']      # 注册后平均每天启动次数
    data = pd.merge(data, temp[['user_id', 'avg_act_after_reg']], how='left', on=['user_id'])

    data = data.fillna(-1)
    gc.collect()

    return data


def create_features(data, day):
    # for i in [1, 2, 3, 5, 7, 9, 11, 14, 16]:
    #     temp = create.loc[create.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'create_in_'+str(i)})
    #     data = pd.merge(data, temp, how='left', on=['user_id'])
    # data = data.fillna(0)
    #序列化
    for i in range(1, 17):
        temp = create.loc[create.day == day - int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'create_on_day'+str(17 - int(i))})
        data = pd.merge(data, temp, how='left', on=['user_id'])
        data = data.apply(fill, axis=1, tarday=day - int(i))
    #带权加和
    for i in [3, 5, 7, 9, 11, 14, 16]:
        value = 0
        for j in range(1, i+1):
            data['wei_create_on_day'+str(17 - j)] = data['create_on_day'+str(17 - j)].apply(lambda x: x*int(17-j) if x>0 else 0)
            value = value + data['wei_create_on_day'+str(17 - j)]
            del(data['wei_create_on_day'+str(17 - j)])
        data['wcreate_in_' + str(i)] = value
    data = data.fillna(0)

    last_create = create.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id','day': 'last_create_day'})
    data = pd.merge(data, last_create, how='left', on=['user_id'])
    data['last_create_distance'] = day - data['last_create_day']
    del(data['last_create_day'])
    return data


if __name__ == '__main__':
    # A榜
    user_reg = pd.read_csv('../data/user_reg.csv', index_col=False)
    act = pd.read_csv('../data/act.csv', index_col=False)
    launch = pd.read_csv('../data/launch.csv', index_col=False)
    create = pd.read_csv('../data/create.csv', index_col=False)

    user_reg, act, launch, create = read_data(1, 16)  # 1号到16号的用户启动、行为、上传日志
    train_1 = pd.read_csv('../data/train_1a_list.csv', index_col=False)
    train_1 = reg_features(train_1, 17)
    train_1 = launch_features(train_1, 17)            # 输入考察日第一天的日期
    train_1 = create_features(train_1, 17)
    train_1 = act_features(train_1, 17)
    train_1.to_csv('../data/train_1a.csv', index=False)

    user_reg, act, launch, create = read_data(8, 23)  # 8号到23号的用户启动、行为、上传日志
    train_2 = pd.read_csv('../data/train_2a_list.csv', index_col=False)
    train_2 = reg_features(train_2, 24)
    train_2 = launch_features(train_2, 24)
    train_2 = create_features(train_2, 24)
    train_2 = act_features(train_2, 24)
    train_2.to_csv('../data/train_2a.csv', index=False)
    gc.collect()

    # B榜
    user_reg = pd.read_csv('../data/user_reg_b.csv', index_col=False)
    act = pd.read_csv('../data/act_b.csv', index_col=False)
    launch = pd.read_csv('../data/launch_b.csv', index_col=False)
    create = pd.read_csv('../data/create_b.csv', index_col=False)

    user_reg, act, launch, create = read_data(1, 16)  # 1号到16号的用户启动、行为、上传日志
    train_1 = pd.read_csv('../data/train_1b_list.csv', index_col=False)
    train_1 = reg_features(train_1, 17)
    train_1 = launch_features(train_1, 17)  # 输入考察日第一天的日期
    train_1 = create_features(train_1, 17)
    train_1 = act_features(train_1, 17)
    train_1.to_csv('../data/train_1b.csv', index=False)

    user_reg, act, launch, create = read_data(8, 23)  # 8号到23号的用户启动、行为、上传日志
    train_2 = pd.read_csv('../data/train_2b_list.csv', index_col=False)
    train_2 = reg_features(train_2, 24)
    train_2 = launch_features(train_2, 24)
    train_2 = create_features(train_2, 24)
    train_2 = act_features(train_2, 24)
    train_2.to_csv('../data/train_2b.csv', index=False)

    user_reg, act, launch, create = read_data(15, 30)
    test = pd.read_csv('../data/test_list.csv', index_col=False)
    test = reg_features(test, 31)
    test = launch_features(test, 31)
    test = create_features(test, 31)
    test = act_features(test, 31)
    test.to_csv('../data/test.csv', index=False)
    gc.collect()