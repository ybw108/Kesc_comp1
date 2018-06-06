import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import seaborn as sns

start = time.time()
app_launch = pd.read_table('E:/Kesci_Kuaishou/data/raw/app_launch_log.txt', names=['user_id', 'day'], encoding='utf-8', sep='\t',)
user_act = pd.read_table('E:/Kesci_Kuaishou/data/raw/user_activity_log.txt', names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'], encoding='utf-8', sep='\t')
user_reg = pd.read_table('E:/Kesci_Kuaishou/data/raw/user_register_log.txt', names=['user_id', 'register_day', 'register_type', 'device_type'], encoding='utf-8', sep='\t')
video_create = pd.read_table('E:/Kesci_Kuaishou/data/raw/video_create_log.txt', names=['user_id', 'day'], encoding='utf-8', sep='\t')

#
# temp = user_reg.groupby(['register_day']).size().reset_index().rename(columns={0:'day_reg_count'})
# color = sns.color_palette()
# plt.figure(figsize=(12, 6))
# sns.barplot(temp.register_day, temp.day_reg_count, alpha=0.8, color=color[0])
# plt.ylabel('reg_num', fontsize=12)
# plt.xlabel('day', fontsize=12)
# plt.show()

for i in ['app_launch', 'user_act', 'user_reg', 'video_create']:
    locals()[i].to_csv('E:/Kesci_Kuaishou/data/' + str(i) + '.csv', encoding='utf-8', index=False)

# split train and valid
#train_act,valid_act = user_act[user_act.day < 24],user_act[user_act.day >= 24]
#train_reg,valid_reg = user_reg[user_reg.register_day < 24],user_reg[user_reg.register_day >= 24]
# Your Feature Engineering Work

# 取最后一个星期有action的用户即可，Baseline F1: 0.790+
# userPre = user_act[user_act.day >= 24]
# userPre = pd.concat([userPre, video_create[video_create.day >= 15]])
# sub = userPre[['user_id']].drop_duplicates()
# sub.to_csv('E:/Kesci_Kuaishou/result/5.26 v1.csv', encoding='utf-8', index=None, header=None)

