import pandas as pd
import gc
from math import log


# def merge_device_type(row):
#     count = row['device_count']
#     if count == 1:
#         row['device_type'] = 99999
#     return row


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
    # temp = data.groupby(['device_type']).size().reset_index().rename(columns={0: 'device_count'})
    # data = pd.merge(data, temp, 'left', ['device_type'])
    # data = data.apply(merge_device_type, axis=1)
    # del(data['device_count'])
    # data['device_count2'] = data['device_count'].apply(lambda x: log(1 + x))
    # 注册时长
    data['register_length'] = day - data['register_day']

    return data


def launch_features(data, day):
    # last launch date
    last_launch = launch.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_launch_day'})
    data = pd.merge(data, last_launch, how='left', on=['user_id'])
    data['launch_diff_target_day'] = day - data['last_launch_day']
    data = data.fillna(-1)
    # launch times in X days
    for i in [1, 3, 5, 7, 9, 11, 14]:  # 1,3,5,7
        temp = launch.loc[launch.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    # 启动时间差特征
    launch['last_launch_day'] = launch.sort_values(['user_id', 'day']).groupby(['user_id'])['day'].shift(1)
    launch['launch_diff'] = launch['day'] - launch['last_launch_day'] - 1
    del(launch['last_launch_day'])
    temp = launch.groupby(['user_id'])['launch_diff'].agg({'launch_diff_max': 'max', 'launch_diff_min': 'min', 'launch_diff_avg': 'sum', 'launch_diff_var': 'var', 'total_launch_count': 'size'}).reset_index()
    temp['launch_diff_avg'] = temp['launch_diff_avg']/temp['total_launch_count']
    temp2 = launch.loc[launch.launch_diff == 0].groupby(['user_id'])['launch_diff'].size().reset_index().rename(columns={'launch_diff': 'continuous_launch_times'})
    temp = pd.merge(temp, temp2, how='left', on=['user_id'])
    temp['continuous_launch_ratio'] = temp['continuous_launch_times']/temp['total_launch_count']
    data = pd.merge(data, temp, how='left', on=['user_id'])
    data['continuous_launch_ratio'] = data['continuous_launch_ratio'].fillna(0)
    del(data['continuous_launch_times'])
    data = data.fillna(-1)

    return data


def act_features(data, day):
    last_act = act[['user_id', 'day']].sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_act_day'})
    data = pd.merge(data, last_act, how='left', on=['user_id'])
    data['act_diff_target_day'] = day - data['last_act_day']
    data = data.fillna(-1)
    for i in [1, 3, 5, 7, 9, 11, 14]:
        temp = act.loc[act.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'act_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    # 每日act数目特征
    temp = act.groupby(['user_id', 'day']).size().rename('day_act_times').reset_index()
    act_temp = pd.merge(act, temp, how='left', on=['user_id', 'day'])
    temp = act_temp.drop_duplicates(['user_id', 'day']).groupby(['user_id'])['day_act_times'].agg({'day_act_max': 'max', 'day_act_min': 'min', 'day_act_avg': 'sum', 'day_act_var': 'var', 'act_day_count': 'size'}).reset_index()
    temp['day_act_avg'] = temp['day_act_avg'] / temp['act_day_count']
    temp['day_act_var/n'] = temp['day_act_var'] / temp['act_day_count']
    data = pd.merge(data, temp, how='left', on=['user_id'])

    # 每日act数目差值特征
    temp2 = act_temp.drop_duplicates(['user_id', 'day'], keep='last').sort_values(['user_id', 'day'])
    temp2['last_day_act_times'] = temp2.sort_values(['user_id', 'day']).groupby(['user_id'])['day_act_times'].shift(1)
    temp2['last_second_diff'] = temp2['day_act_times'] - temp2['last_day_act_times']
    temp2 = pd.merge(temp2, temp[['user_id', 'day_act_avg']], how='left', on=['user_id'])
    temp2['last_avg_diff'] = temp2['day_act_times'] - temp2['day_act_avg']
    temp2['last_second_trend'] = temp2['last_second_diff'].apply(lambda x: 1 if x > 0 else 0)
    temp2['last_avg_trend'] = temp2['last_avg_diff'].apply(lambda x: 1 if x > 0 else 0)
    temp = temp2.groupby(['user_id'])['last_second_diff'].agg(
        {'last_second_diff_max': 'max', 'last_second_diff_min': 'min', 'last_second_diff_avg': 'sum', 'last_second_diff_var': 'var', 'act_day_count': 'size'}).reset_index()
    temp['last_second_diff_avg'] = temp['last_second_diff_avg']/temp['act_day_count']
    temp['last_second_diff_var/n'] = temp['last_second_diff_var']/temp['act_day_count']
    # 每日act数目差值趋势特征
    temp['last_second_trend'] = temp2.sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='last').reset_index()['last_second_trend']  # 最后一天和前一天相比的趋势
    temp['last_avg_trend'] = temp2.sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='last').reset_index()['last_avg_trend']        # 最后一天和平均数相比的趋势
    temp['second_trend_ratio'] = temp2.groupby(['user_id'])['last_second_trend'].mean().reset_index()['last_second_trend']                          # 最后与前一天呈增长趋势的天数占有act行为的天数的比例
    temp['avg_trend_ratio'] = temp2.groupby(['user_id'])['last_avg_trend'].mean().reset_index()['last_avg_trend']                                   # 最后与平均值呈增长趋势的天数占有act行为的天数的比例

    temp3 = temp2.loc[temp2.last_second_trend == 1].groupby(['user_id'])['last_second_trend'].size().rename('second_trend_count').reset_index()
    temp = pd.merge(temp, temp3, how='left', on=['user_id'])
    temp3 = temp2.loc[temp2.last_avg_trend == 1].groupby(['user_id'])['last_avg_trend'].size().rename('avg_trend_count').reset_index()
    temp = pd.merge(temp, temp3, how='left', on=['user_id'])
    temp['second_trend_count'] = temp['second_trend_count'].fillna(0)
    temp['avg_trend_count'] = temp['avg_trend_count'].fillna(0)

    del(data['act_day_count'])
    data = pd.merge(data, temp, how='left', on=['user_id'])
    temp2 = temp2.sort_values(['user_id', 'day']).drop_duplicates(['user_id'], keep='last')
    data = pd.merge(data, temp2[['user_id', 'last_avg_diff', 'last_second_diff']], how='left', on=['user_id'])   # 两个中间生成的特征是否要加入训练？？

    # data['last_second_trend'] = data['last_second_trend'].fillna(0)
    # data['last_avg_trend'] = data['last_avg_trend'].fillna(0)
    # data['second_trend_count'] = data['second_trend_count'].fillna(0)
    # data['avg_trend_count'] = data['avg_trend_count'].fillna(0)
    # data['second_trend_ratio'] = data['second_trend_ratio'].fillna(0)
    # data['avg_trend_ratio'] = data['avg_trend_ratio'].fillna(0)


    data = data.fillna(-1)
    # 每日act_diff 特征 即当天和前一天act数量差值的特征
    gc.collect()
    return data


def create_features(data, day):
    last_create = create.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id','day': 'last_create_day'})
    data = pd.merge(data, last_create, how='left', on=['user_id'])
    data['create_diff_target_day'] = day - data['last_create_day']
    data = data.fillna(-1)
    for i in [1, 3, 5, 7, 9, 11, 14]:
        temp = create.loc[create.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'create_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    # create时间差特征
    create['last_create_day'] = create.sort_values(['user_id', 'day']).groupby(['user_id'])['day'].shift(1)
    create['create_diff'] = create['day'] - create['last_create_day'] - 1
    del(create['last_create_day'])
    temp = create.groupby(['user_id'])['create_diff'].agg({'create_diff_max': 'max', 'create_diff_min': 'min', 'create_diff_avg': 'sum', 'create_diff_var': 'var', 'total_create_count': 'size'}).reset_index()
    temp['create_diff_avg'] = temp['create_diff_avg']/temp['total_create_count']
    # temp2 = create.loc[create.create_diff == 0].groupby(['user_id'])['create_diff'].size().reset_index().rename(columns={'launch_diff': 'continuous_launch_times'})
    # temp = pd.merge(temp, temp2, how='left', on=['user_id'])
    # temp['continuous_launch_ratio'] = temp['continuous_launch_times']/temp['total_launch_count']
    data = pd.merge(data, temp, how='left', on=['user_id'])
    # data['continuous_launch_ratio'] = data['continuous_launch_ratio'].fillna(0)
    # del(data['continuous_launch_times'])
    data = data.fillna(-1)

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