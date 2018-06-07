import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

launch = pd.read_csv('../data/launch.csv', index_col=False)
act = pd.read_csv('../data/act.csv', index_col=False)
create = pd.read_csv('../data/create.csv', index_col=False)
train_1 = pd.read_csv('../data/train_1.csv', index_col=False)
train_2 = pd.read_csv('../data/train_2.csv', index_col=False)

launch.sort_values(['user_id', 'day'], ascending=True, inplace=True)
act.sort_values(['user_id', 'day'], ascending=True, inplace=True)
create.sort_values(['user_id', 'day'], ascending=True, inplace=True)


color = sns.color_palette()
plt.figure(figsize=(12, 6))
sns.barplot(a.index, a.values, alpha=0.8, color=color[0])
plt.ylabel('reg_num', fontsize=12)
plt.xlabel('day', fontsize=12)
plt.show()