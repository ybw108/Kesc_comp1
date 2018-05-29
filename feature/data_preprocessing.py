import pandas as pd

user_reg = pd.read_csv('../data/user_reg.csv', index_col=False)
train = user_reg.loc[user_reg.register_day < 24 ]
train['label'] = 0

user_act = pd.read_csv('../data/act.csv', index_col=False)
app_launch = pd.read_csv('../data/launch.csv', index_col=False)
video_create = pd.read_csv('../data/create.csv', index_col=False)

# active_user = pd.concat([user_act.loc[user_act.day >= 24]['user_id'].drop_duplicates(), app_launch.loc[app_launch.day >= 24]['user_id'].drop_duplicates(), video_create.loc[video_create.day >= 24]['user_id'].drop_duplicates()])
# active_user.drop_duplicates(inplace=True)

active_user = app_launch.loc[app_launch.day >= 24]['user_id'].drop_duplicates()
train['label'] = train['user_id'].isin(active_user).apply(lambda x: 1 if x == True else 0)
train.to_csv('../data/train_list.csv', index=False)
user_reg.to_csv('../data/test_list.csv', index=False)
print('done')