import pandas as pd


def read_data(first_day, last_day):
    user_reg_temp = pd.read_csv('../data/user_reg.csv', index_col=False)
    act_temp = pd.read_csv('../data/act.csv', index_col=False)
    launch_temp = pd.read_csv('../data/launch.csv', index_col=False)
    create_temp = pd.read_csv('../data/create.csv', index_col=False)
    act_temp = act_temp.loc[(act_temp.day >= first_day) & (act_temp.day <= last_day)]
    launch_temp = launch_temp.loc[(launch_temp.day >= first_day) & (launch_temp.day <= last_day)]
    create_temp = create_temp.loc[(create_temp.day >= first_day) & (create_temp.day <= last_day)]

    return user_reg_temp, act_temp, launch_temp, create_temp


def reg_features(data):
    data = pd.merge(data, user_reg, how='left', on=['user_id'])
    return data


def launch_features(data, day):
    # last launch date
    last_launch = launch.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_launch_day'})
    data = pd.merge(data, last_launch, how='left', on=['user_id'])
    data = data.fillna(-1)
    # launch times in X days
    for i in [3, 7]:  # 1,3,5,7
        temp = launch.loc[launch.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    #temp = launch.groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_total'})
    #data = pd.merge(data, temp, 'left', ['user_id'])
    data = data.fillna(0)

    return data


def act_features(data, day):
    last_act = act[['user_id', 'day']].sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_act_day'})
    data = pd.merge(data, last_act, how='left', on=['user_id'])
    data = data.fillna(-1)
    for i in [3, 7]:
        temp = act.loc[act.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'act_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    return data


def create_features(data, day):
    last_create = create.sort_values(by=['day']).drop_duplicates(['user_id'], keep='last').rename(columns={0: 'user_id', 'day': 'last_create_day'})
    data = pd.merge(data, last_create, how='left', on=['user_id'])
    data = data.fillna(-1)
    for i in [3, 7]:
        temp = create.loc[create.day >= day-int(i)].groupby(['user_id']).size().reset_index().rename(columns={0: 'create_in_'+str(i)})
        data = pd.merge(data, temp, how='left', on=['user_id'])
    data = data.fillna(0)
    return data


if __name__ == '__main__':
    user_reg, act, launch, create = read_data(1, 16)  # 1号到16号的用户启动、行为、上传日志
    train_1 = pd.read_csv('../data/train_1_list.csv', index_col=False)
    train_1 = reg_features(train_1)
    train_1 = launch_features(train_1, 17)            # 输入考察日第一天的日期
    train_1 = create_features(train_1, 17)
    train_1 = act_features(train_1, 17)
    train_1.to_csv('../data/train_1.csv', index=False)

    user_reg, act, launch, create = read_data(8, 23)  # 8号到23号的用户启动、行为、上传日志
    train_2 = pd.read_csv('../data/train_2_list.csv', index_col=False)
    train_2 = reg_features(train_2)
    train_2 = launch_features(train_2, 24)
    train_2 = create_features(train_2, 24)
    train_2 = act_features(train_2, 24)
    train_2.to_csv('../data/train_2.csv', index=False)

    user_reg, act, launch, create = read_data(15, 30)
    test = pd.read_csv('../data/test_list.csv', index_col=False)
    test = reg_features(test)
    test = launch_features(test, 31)
    test = create_features(test, 31)
    test = act_features(test, 31)
    test.to_csv('../data/test.csv', index=False)