import pandas as pd


def launch_features(data, mode):
    if mode == 'train_mode':
        f = 24
    else:
        f = 31
    # last launch date
    last_launch = launch.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_launch_day'})
    data = pd.merge(data, last_launch, how='left', on=['user_id'])
    data = data.fillna(-1)
    # launch_difftime 第一次、最后一次启动距考察日的时间差
    data['last_launch_difftime'] = int(f) - data['last_launch_day']
    first_launch = launch.sort_values(by=['day']).drop_duplicates(['user_id'], keep='first').rename(columns={0: 'user_id', 'day': 'first_launch_day'})
    data = pd.merge(data, first_launch, how='left', on=['user_id'])
    data['first_launch_difftime'] = int(f) - data['first_launch_day']
    del (data['first_launch_day'])
    # launch times in X days
    for i in [2, 3, 5, 7, 14]:
        temp = launch.loc[launch.day >= f-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    temp = launch.groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_total'})
    data = pd.merge(data, temp, 'left', ['user_id'])
    data = data.fillna(0)
    #
    return data


def act_features(data):
    last_act = act[['user_id', 'day']].sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_act_day'})
    data = pd.merge(data, last_act, how='left', on=['user_id'])
    data = data.fillna(-1)
    temp = act.loc[act.day >= 21].groupby(['user_id']).size().reset_index().rename(columns={0: 'act_in_3'})
    data = pd.merge(data, temp, how='left', on=['user_id'])
    temp = act.loc[act.day >= 17].groupby(['user_id']).size().reset_index().rename(columns={0: 'act_in_7'})
    data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    return data


def create_features(data):
    last_create = create.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_create_day'})
    data = pd.merge(data, last_create, how='left', on=['user_id'])
    data = data.fillna(-1)
    temp = create.loc[create.day >= 21].groupby(['user_id']).size().reset_index().rename(columns={0: 'create_in_3'})
    data = pd.merge(data, temp, how='left', on=['user_id'])
    temp = create.loc[create.day >= 17].groupby(['user_id']).size().reset_index().rename(columns={0: 'create_in_7'})
    data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    return data


if __name__ == '__main__':
    act = pd.read_csv('../data/act.csv', index_col=False)
    launch = pd.read_csv('../data/launch.csv', index_col=False)
    create = pd.read_csv('../data/create.csv', index_col=False)
    act = act.loc[act.day < 24]                      # 线下验证用前23天的数据，线上提交用前30天
    launch = launch.loc[launch.day < 24]
    create = create[create.day < 24]
    train = pd.read_csv('../data/train_list.csv', index_col=False)
    train = launch_features(train, 'train_mode')
    train = create_features(train)
    train = act_features(train)
    train.to_csv('../data/train.csv', index=False)

    act = pd.read_csv('../data/act.csv', index_col=False)
    launch = pd.read_csv('../data/launch.csv', index_col=False)
    create = pd.read_csv('../data/create.csv', index_col=False)
    test = pd.read_csv('../data/test_list.csv', index_col=False)
    test = launch_features(test, 'test_mode')
    test = create_features(test)
    test = act_features(test)
    test.to_csv('../data/test.csv', index=False)