import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder

def merge_device_type(row):
    count = row['device_count']
    if count == 1:
        row['device_type'] = -1

    return row


def split_data(first_day, last_day):
    reg_temp = user_reg.loc[(user_reg.register_day >= first_day) & (user_reg.register_day <= last_day)]
    act_temp = act.loc[(act.day >= first_day) & (act.day <= last_day)]
    launch_temp = launch.loc[(launch.day >= first_day) & (launch.day <= last_day)]
    create_temp = create.loc[(create.day >= first_day) & (create.day <= last_day)]

    train_temp = pd.concat([reg_temp['user_id'], act_temp['user_id'], launch_temp['user_id'], create_temp['user_id']])
    train_temp.drop_duplicates(inplace=True, keep='last')
    return train_temp


def give_label(first_day, last_day, data):
    data = data.to_frame()
    active_user = launch.loc[(launch.day >= first_day) & (launch.day <= last_day)]['user_id'].drop_duplicates()
    data['label'] = data['user_id'].isin(active_user).apply(lambda x: 1 if x == True else 0)
    return data


if __name__ == '__main__':
    # A榜数据
    user_reg = pd.read_csv('../data/user_reg.csv', index_col=False)
    act = pd.read_csv('../data/act.csv', index_col=False)
    launch = pd.read_csv('../data/launch.csv', index_col=False)
    create = pd.read_csv('../data/create.csv', index_col=False)
    # 划分数据、打标签
    train_1 = split_data(1, 16)
    train_2 = split_data(8, 23)
    train_1 = give_label(17, 23, train_1)
    train_2 = give_label(24, 30, train_2)
    gc.collect()
    # 输出
    train_1.to_csv('../data/train_1a_list.csv', index=False)
    train_2.to_csv('../data/train_2a_list.csv', index=False)

    # B榜数据
    user_reg = pd.read_csv('../data/user_reg_b.csv', index_col=False)
    act = pd.read_csv('../data/act_b.csv', index_col=False)
    launch = pd.read_csv('../data/launch_b.csv', index_col=False)
    create = pd.read_csv('../data/create_b.csv', index_col=False)
    # 划分数据、打标签
    train_1 = split_data(1, 16)
    train_2 = split_data(8, 23)
    test = split_data(15, 30)
    train_1 = give_label(17, 23, train_1)
    train_2 = give_label(24, 30, train_2)
    gc.collect()
    # 输出
    train_1.to_csv('../data/train_1b_list.csv', index=False)
    train_2.to_csv('../data/train_2b_list.csv', index=False)
    test = test.to_frame()
    test.to_csv('../data/test_list.csv', index=False)





