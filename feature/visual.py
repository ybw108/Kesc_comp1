import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

launch = pd.read_csv('../data/launch.csv', index_col=False)
act = pd.read_csv('../data/act.csv', index_col=False)
create = pd.read_csv('../data/create.csv', index_col=False)
train = pd.read_csv('../data/train.csv', index_col=False)



temp = pd.merge(launch, train[['user_id', 'label', 'register_day', 'last_launch_day', 'launch_in_3', 'launch_in_7']], how='left', on=['user_id'])
temp = temp.loc[temp.day < 24].astype(int)

a = temp.groupby(['user_id']).size().reset_index().rename(columns={0: 'launch_ratio'})
a['launch_ratio'] = a['launch_ratio']/23
temp = pd.merge(temp, a, how='left', on=['user_id'])
a = temp.groupby(['last_launch_day'])['label'].mean()

temp.sort_values(['user_id', 'day'], ascending=True, inplace=True)
launch.sort_values(['user_id', 'day'], ascending=True, inplace=True)
act.sort_values(['user_id', 'day'], ascending=True, inplace=True)
create.sort_values(['user_id', 'day'], ascending=True, inplace=True)

color = sns.color_palette()
plt.figure(figsize=(12, 6))
sns.barplot(a.index, a.values, alpha=0.8, color=color[0])
plt.ylabel('reg_num', fontsize=12)
plt.xlabel('day', fontsize=12)
plt.show()