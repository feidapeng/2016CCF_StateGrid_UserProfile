'''
构造高敏感度用户的特征
“高敏感度用户”定义：在表1 “95598工单” 中有2条及以上通话记录的用户

@Author：Fei peng
@E-mail：dlutfeipeng@gmail.com
'''
import pandas as pd
import numpy as np
import csv
import re
import os
import jieba
import codecs
import pickle
from numpy import log
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

data_path = '../rawdata/'
#95598 train 
file_jobinfo_train = '01_arc_s_95598_wkst_train.tsv' 
# 95598 test
file_jobinfo_test = '01_arc_s_95598_wkst_test.tsv'    
# 通话信息记录
file_comm = '02_s_comm_rec.tsv'
# 应收电费信息表 train
file_flow_train = '09_arc_a_rcvbl_flow.tsv'
# 应收电费信息表 test
file_flow_test = '09_arc_a_rcvbl_flow_test.tsv'
# 训练集正例
file_label = 'train_label.csv'
# 测试集
file_test = 'test_to_predict.csv'

print('Part II:')
# --------------------------------------------load data---------------------------------------------------
print('@loading data...')
train_info = pd.read_csv(data_path + 'processed_' + file_jobinfo_train, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
# 过滤CUST_NO为空的用户
train_info = train_info.loc[~train_info.CUST_NO.isnull()]
train_info['CUST_NO'] = train_info.CUST_NO.astype(np.int64)
# 构建用户索引
train = train_info.CUST_NO.value_counts().to_frame().reset_index()
train.columns = ['CUST_NO', 'counts_of_jobinfo']
temp = pd.read_csv(data_path + file_label, header=None)
temp.columns = ['CUST_NO']
train['label'] = 0
train.loc[train.CUST_NO.isin(temp.CUST_NO), 'label'] = 1
train = train[['CUST_NO', 'label', 'counts_of_jobinfo']]

test_info = pd.read_csv(data_path + 'processed_' + file_jobinfo_test, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
test = test_info.CUST_NO.value_counts().to_frame().reset_index()
test.columns = ['CUST_NO', 'counts_of_jobinfo']
test['label'] = -1
test = test[['CUST_NO', 'label', 'counts_of_jobinfo']]

df = train.append(test).copy()
labels = df.copy()
del temp, train, test
##################################
##################################
# 只保留多条工单信息的高敏感度用户
##################################
##################################
df = df.loc[df.counts_of_jobinfo != 1].copy()
# 同时去掉表1记录太多的离群点
df = df.loc[df.counts_of_jobinfo <= 10].copy()

df.reset_index(drop=True, inplace=1)
train = df.loc[df.label != -1]
test = df.loc[df.label == -1]
print('原始数据中的高敏感度用户分布情况如下：')
print('训练集：',train.shape[0])
print('正样本:',train.loc[train.label == 1].shape[0])
print('负样本:',train.loc[train.label == 0].shape[0])
print('-----------------------')
print('测试集：',test.shape[0])

# --------------------------------------------create features---------------------------------------------------
print('@creating features...')
# 合并工单
jobinfo = train_info.append(test_info).copy()
jobinfo = jobinfo.merge(labels[['CUST_NO', 'label']], on='CUST_NO', how='left')
jobinfo['date'] = jobinfo.HANDLE_TIME.apply(lambda x:(str(x).split()[0]))
print('构造表1 date和topic的敏感度特征...')
##############
# 1.date score
##############
train = jobinfo[jobinfo.label != -1]
ratio = {}
a = 0.001
for i in train.date.unique():
    ratio[i] = (len(train.loc[(train.date == i) & (train.label == 1)]) + a) / (len(train.loc[train.date == i]) + 2*a)
jobinfo['date_ratio'] = jobinfo.date.map(ratio)

df['sum_date_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date_ratio.sum())
df['mean_date_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date_ratio.mean())
df['min_date_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date_ratio.min())
df['max_date_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date_ratio.max())
df['std_date_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date_ratio.std())
df['median_date_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date_ratio.median())
##############
# 2.topic score
##############
pattern = re.compile('【[^【】]*】')
def get_topic(x):
    finds = pattern.findall(x)
    if len(finds) == 0:
        return '-1'
    else:
        return finds[0]
jobinfo['topic'] = jobinfo.ACCEPT_CONTENT.apply(lambda x: get_topic(x))

# ratio
train = jobinfo[jobinfo.label != -1]
ratio = {}
a = 0.001
for i in train.topic.unique():
    ratio[i] = (len(train.loc[(train.topic == i) & (train.label == 1)]) + a) / (len(train.loc[train.topic == i]) + 2*a)
    
topics = jobinfo.topic.value_counts().to_frame().reset_index()
topics.columns = ['topic', 'counts']
topics['topic_ratio'] = topics.topic.map(ratio)
topics = topics.loc[(topics.counts > 4) & (~topics.topic_ratio.isnull())]
jobinfo = jobinfo.merge(topics[['topic', 'topic_ratio']], on='topic', how='left')
jobinfo.topic_ratio.fillna(topics.topic_ratio.median(), inplace=1)

df['sum_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').topic_ratio.sum())
df['mean_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').topic_ratio.mean())
df['min_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').topic_ratio.min())
df['max_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').topic_ratio.max())
df['std_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').topic_ratio.std())
df['median_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').topic_ratio.median())

print('处理表2...')
# 1. 没有表2信息的用户全是非敏感用户,剔除掉，其中训练集181个，测试集43个
# 2. 6个训练集的用户，REQ_BEGIN_DATE > REQ_FINISH_DATE . 异常数据，剔除掉

comm = pd.read_csv(data_path + file_comm, sep='\t')
comm.drop_duplicates(inplace=1)
comm = comm.loc[comm.APP_NO.isin(jobinfo.ID)]
comm = comm.rename(columns={'APP_NO':'ID'})
comm = comm.merge(jobinfo[['ID', 'CUST_NO']], on='ID', how='left')
comm['REQ_BEGIN_DATE'] = comm.REQ_BEGIN_DATE.apply(lambda x:pd.to_datetime(x))
comm['REQ_FINISH_DATE'] = comm.REQ_FINISH_DATE.apply(lambda x:pd.to_datetime(x))

# 过滤
comm = comm.loc[~(comm.REQ_BEGIN_DATE > comm.REQ_FINISH_DATE)].copy()
df = df.loc[df.CUST_NO.isin(comm.CUST_NO)].copy()

comm['holding_time'] = comm['REQ_FINISH_DATE'] - comm['REQ_BEGIN_DATE']
comm['holding_time_seconds'] = comm.holding_time.apply(lambda x:x.seconds)

df['counts_of_comm'] = df.CUST_NO.map(comm.groupby('CUST_NO').size())
df['min_holding_time_seconds'] = df.CUST_NO.map(comm.groupby('CUST_NO').holding_time_seconds.min())
df['min_holding_time_seconds'] = df.min_holding_time_seconds.apply(lambda x:log(x+1))
df['max_holding_time_seconds'] = df.CUST_NO.map(comm.groupby('CUST_NO').holding_time_seconds.max())
df['max_holding_time_seconds'] = df.max_holding_time_seconds.apply(lambda x:log(x+1))
df['sum_holding_time_seconds'] = df.CUST_NO.map(comm.groupby('CUST_NO').holding_time_seconds.sum())
df['sum_holding_time_seconds'] = df.sum_holding_time_seconds.apply(lambda x:log(x+1))
df['std_holding_time_seconds'] = df.CUST_NO.map(comm.groupby('CUST_NO').holding_time_seconds.std())
df['std_holding_time_seconds'] = df.std_holding_time_seconds.apply(lambda x:log(x+1))
df['median_holding_time_seconds'] = df.CUST_NO.map(comm.groupby('CUST_NO').holding_time_seconds.median())
df['median_holding_time_seconds'] = df.median_holding_time_seconds.apply(lambda x:log(x+1))
df['mean_holding_time_seconds'] = df['sum_holding_time_seconds'] / df['counts_of_comm']

df['comm_not_equal_jobinfo'] = 0
df.loc[df.counts_of_jobinfo != df.counts_of_comm, 'comm_not_equal_jobinfo'] = 1

df['counts_jobinfo_jianqu_comm'] = df['counts_of_jobinfo'] - df['counts_of_comm']
del comm
# ************************************************************************************************************
print('处理表1...')
jobinfo = jobinfo.loc[jobinfo.CUST_NO.isin(df.CUST_NO)].copy()
jobinfo.reset_index(drop=True, inplace=1)
################
# CUST_NO
################
# rank
df['rank_CUST_NO'] = df.CUST_NO.rank(method='max')
df['rank_CUST_NO'] = MinMaxScaler().fit_transform(df.rank_CUST_NO)
################
# BUSI_TYPE_CODE
################
df['nunique_BUSI_TYPE'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').BUSI_TYPE_CODE.nunique())
df['counts_divide_busi'] =  df['counts_of_jobinfo'] / df['nunique_BUSI_TYPE']
df.drop(['nunique_BUSI_TYPE'], axis=1, inplace=1)
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.BUSI_TYPE_CODE, prefix='count_BUSI_TYPE_CODE')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.BUSI_TYPE_CODE.unique(): 
    df['ratio_BUSI_TYPE_CODE_{}'.format(i)] = df['count_BUSI_TYPE_CODE_{}'.format(i)] / df.counts_of_jobinfo
################
# URBAN_RURAL_FLAG
################
jobinfo['URBAN_RURAL_FLAG'].fillna(-1, inplace=1)
df['nunique_URBAN'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').URBAN_RURAL_FLAG.nunique())
df['ratio_urban'] =  df['counts_of_jobinfo'] / df['nunique_URBAN']
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.URBAN_RURAL_FLAG, prefix='count_URBAN_RURAL_FLAG')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.URBAN_RURAL_FLAG.unique(): 
    df['ratio_URBAN_RURAL_FLAG_{}'.format(i)] = df['count_URBAN_RURAL_FLAG_{}'.format(i)] / df.counts_of_jobinfo
################
## ORG_NO
################
df['nunique_ORG_NO'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').ORG_NO.nunique())
df['nunique_ORG_NO_divide_counts'] = df['nunique_ORG_NO'] / df['counts_of_jobinfo']
df['counts_divide_nunique_ORG_NO'] = df['counts_of_jobinfo'] / df['nunique_ORG_NO']
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.ORG_NO, prefix='count_ORG_NO')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.ORG_NO.unique(): 
    df['ratio_ORG_NO_{}'.format(i)] = df['count_ORG_NO_{}'.format(i)] / df.counts_of_jobinfo
########################
# len_of_ORG_NO
########################
jobinfo['len_of_ORG_NO'] = jobinfo.ORG_NO.apply(lambda x:len(str(x)))
df['nunique_len_of_ORG_NO'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').len_of_ORG_NO.nunique())
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.len_of_ORG_NO, prefix='count_len_of_ORG_NO')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.len_of_ORG_NO.unique(): 
    df['ratio_len_of_ORG_NO_{}'.format(i)] = df['count_len_of_ORG_NO_{}'.format(i)] / df.counts_of_jobinfo
###############
# ELEC_TYPE
###############
jobinfo['ELEC_TYPE'].fillna(0, inplace=1)
df['nunique_ELEC_TYPE'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').ELEC_TYPE.nunique())
df['ratio_ELEC_TYPE'] =  df['counts_of_jobinfo'] / df['nunique_ELEC_TYPE']
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.ELEC_TYPE, prefix='count_ELEC_TYPE')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.ELEC_TYPE.unique(): 
    df['ratio_ELEC_TYPE_{}'.format(i)] = df['count_ELEC_TYPE_{}'.format(i)] / df.counts_of_jobinfo
###############
# head_of_ELEC_TYPE
###############
jobinfo['head_of_ELEC_TYPE'] = jobinfo.ELEC_TYPE.apply(lambda x: int(str(x)[0]))
df['nunique_head_of_ELEC_TYPE'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').head_of_ELEC_TYPE.nunique())
df['ratio_head_of_ELEC_TYPE'] =  df['counts_of_jobinfo'] / df['nunique_head_of_ELEC_TYPE']
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.head_of_ELEC_TYPE, prefix='count_head_of_ELEC_TYPE')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.head_of_ELEC_TYPE.unique(): 
    df['ratio_head_of_ELEC_TYPE_{}'.format(i)] = df['count_head_of_ELEC_TYPE_{}'.format(i)] / df.counts_of_jobinfo
####################
# time
####################
# 1. month12维
jobinfo['date'] = jobinfo.HANDLE_TIME.apply(lambda x:pd.to_datetime(str(x).split()[0]))
jobinfo['month'] = jobinfo.date.apply(lambda x:x.month)
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.month, prefix='count_month')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.month.unique(): 
    df['ratio_month_{}'.format(i)] = df['count_month_{}'.format(i)] / df.counts_of_jobinfo
# 2. 几个不同的日期
df['nunique_date'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date.nunique())
# 3.第一个电话和最后一个电话间隔几天
df['dates'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').date.apply(lambda x:[i for i in sorted(x)]))
df['how_many_days_interval'] = df.dates.apply(lambda x:x[-1] - x[0])
df['how_many_days_interval'] = df.how_many_days_interval.apply(lambda x:x.days)
df.drop(['dates'], axis=1, inplace=1)
# 4.平均几天打一个电话
df['how_many_days_one_call'] = df['how_many_days_interval'] / df['counts_of_jobinfo']
# 5.平均一天打几个电话
df['how_many_calls_in_oneday'] = df['counts_of_jobinfo'] / df['nunique_date']
# 6.平均多少天会打电话
df['mean_how_many_days_call'] = df['how_many_days_interval'] / df['nunique_date']
# 7. 间隔
jobinfo['time'] = jobinfo.HANDLE_TIME.apply(lambda x:pd.to_datetime(x))
df['times'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').time.apply(lambda x:[i for i in sorted(x)]))
def get_gaps(x):
    gaps = []
    for i in range(len(x)-1):
        gap = pd.to_datetime(x[i+1]) - pd.to_datetime(x[i])
        gaps.append(gap.days)
    return gaps
df['gaps'] = df.times.apply(lambda x:get_gaps(x))
df['min_gap'] = df.gaps.apply(lambda x:min(x))
df['max_gap'] = df.gaps.apply(lambda x:max(x))
df['mean_gap'] = df.gaps.apply(lambda x:np.mean(x))
df['std_gap'] = df.gaps.apply(lambda x:np.std(x))
df['median_gap'] = df.gaps.apply(lambda x:np.median(x))
df.drop(['times', 'gaps'], axis=1, inplace=1)
# 8. 一个月中的哪一天
jobinfo['day'] = jobinfo.date.apply(lambda x:x.day)
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.day, prefix='day')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# 9. 小时
jobinfo['hour'] = jobinfo.time.apply(lambda x:x.hour)
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.hour, prefix='hour')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# 10. 最多一个月打了几个电话
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.month, prefix='month')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
temp = pd.concat([temp, temp.drop(['CUST_NO'], axis=1).max(axis=1).to_frame(name='most_times_jobinfo_in_one_month')], axis=1)
df = df.merge(temp[['CUST_NO', 'most_times_jobinfo_in_one_month']], on='CUST_NO', how='left')
# 11.打电话日期的标准差
jobinfo['day_of_year'] = jobinfo.date.apply(lambda x:x.dayofyear)
df['std_day_of_year'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').day_of_year.std())
###############
###############
# CITY_ORG_NO
###############
###############
df['nunique_CITY_ORG_NO'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').CITY_ORG_NO.nunique())
df['ratio_CITY_ORG_NO'] =  df['counts_of_jobinfo'] / df['nunique_CITY_ORG_NO']
# count
temp = jobinfo[['CUST_NO']]
temp = pd.concat([temp, pd.get_dummies(jobinfo.CITY_ORG_NO, prefix='count_CITY_ORG_NO')], axis=1)
temp = temp.groupby('CUST_NO').sum()
temp.reset_index(inplace=1)
df = df.merge(temp, on='CUST_NO', how='left')
# ratio
for i in jobinfo.CITY_ORG_NO.unique(): 
    df['ratio_CITY_ORG_NO_{}'.format(i)] = df['count_CITY_ORG_NO_{}'.format(i)] / df.counts_of_jobinfo
###############
# topic
###############
# ratio
train = jobinfo[jobinfo.label != -1]
ratio = {}
a = 0.001
for i in train.topic.unique():
    ratio[i] = (len(train.loc[(train.topic == i) & (train.label == 1)]) + a) / (len(train.loc[train.topic == i]) + 2*a)
    
topics = jobinfo.topic.value_counts().to_frame().reset_index()
topics.columns = ['topic', 'counts']

topics['multi_topic_ratio'] = topics.topic.map(ratio)
topics = topics.loc[(topics.counts > 4) & (~topics.multi_topic_ratio.isnull())]

jobinfo = jobinfo.merge(topics[['topic', 'multi_topic_ratio']], on='topic', how='left')
jobinfo.multi_topic_ratio.fillna(topics.multi_topic_ratio.median(), inplace=1)

df['sum_multi_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').multi_topic_ratio.sum())
df['mean_multi_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').multi_topic_ratio.mean())
df['max_multi_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').multi_topic_ratio.max())
df['min_multi_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').multi_topic_ratio.min())
df['std_multi_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').multi_topic_ratio.std())
df['median_multi_topic_ratio'] = df.CUST_NO.map(jobinfo.groupby('CUST_NO').multi_topic_ratio.median())
# ************************************************************************************************************
print('处理表9...')
train_flow = pd.read_csv(data_path + file_flow_train, sep='\t')
test_flow = pd.read_csv(data_path + file_flow_test, sep='\t')
flow = train_flow.append(test_flow).copy()
flow.rename(columns={'CONS_NO':'CUST_NO'}, inplace=1)
flow.drop_duplicates(inplace=1)
flow = flow.loc[flow.CUST_NO.isin(df.CUST_NO)].copy()

flow['T_PQ'] = flow.T_PQ.apply(lambda x:-x if x<0 else x)
flow['RCVBL_AMT'] = flow.RCVBL_AMT.apply(lambda x:-x if x<0 else x)
flow['RCVED_AMT'] = flow.RCVED_AMT.apply(lambda x:-x if x<0 else x)
flow['OWE_AMT'] = flow.OWE_AMT.apply(lambda x:-x if x<0 else x)
# 是否有表9
df['has_biao9'] = 0
df.loc[df.CUST_NO.isin(flow.CUST_NO), 'has_biao9'] = 1

df['counts_of_09flow'] = df.CUST_NO.map(flow.groupby('CUST_NO').size())

# 应收金额
df['sum_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.sum()) + 1)
df['mean_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.mean()) + 1)
df['max_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.max()) + 1)
df['min_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.min()) + 1)
df['std_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.std()) + 1)
# 实收金额
df['sum_shishoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_AMT.sum()) + 1)
# 少交了多少
df['qianfei'] = df['sum_yingshoujine'] - df['sum_shishoujine']

# 总电量
df['sum_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.sum()) + 1)
df['mean_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.mean()) + 1)
df['max_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.max()) + 1)
df['min_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.min()) + 1)
df['std_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.std()) + 1)

# 电费金额
df['sum_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.sum()) + 1)
df['mean_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.mean()) + 1)
df['max_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.max()) + 1)
df['min_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.min()) + 1)
df['std_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.std()) + 1)

# 电费金额和应收金额差多少
df['dianfei_jian_yingshoujine'] = df['sum_OWE_AMT'] - df['sum_yingshoujine']

# 应收违约金
df['sum_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.sum()) + 1)
df['mean_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.mean()) + 1)
df['max_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.max()) + 1)
df['min_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.min()) + 1)
df['std_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.std()) + 1)

# 实收违约金
df['sum_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.sum()) + 1)
df['mean_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.mean()) + 1)
df['max_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.max()) + 1)
df['min_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.min()) + 1)
df['std_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.std()) + 1)

df['chaduoshao_weiyuejin'] = df['sum_RCVBL_PENALTY'] - df['sum_RCVED_PENALTY']

# 每个用户有几个月的记录
df['nunique_RCVBL_YM'] = df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_YM.nunique())

# 平均每个月几条
df['mean_RCVBL_YM'] = df['counts_of_09flow'] / df['nunique_RCVBL_YM']
del train_flow, test_flow, flow

print('统计特征处理完成！')
pickle.dump(df, open('../myfeatures/statistical_features_2.pkl', 'wb'))
# ************************************************************************************************************
print('开始处理表1中的文本特征...')
###############
# ACCEPT_CONTENT
###############
mywords = ['户号', '分时', '抄表', '抄表示数', '工单', '单号', '工单号', '空气开关', '脉冲灯', '计量表', '来电', '报修']
for word in mywords:
    jieba.add_word(word)

stops = set()
with open('../stopwords.txt', encoding='utf-8')as f:
    for word in f:
        word = word.strip()
        stops.add(word)

def fenci(line):
    res = []
    words = jieba.cut(line)
    for word in words:
        if word not in stops:
            res.append(word)
    return ' '.join(res)
print('分词ing...')
jobinfo['contents'] = jobinfo.ACCEPT_CONTENT.apply(lambda x:fenci(x))
def hash_number(x):
    shouji_pattern = re.compile('\s1\d{10}\s|\s1\d{10}\Z')
    if shouji_pattern.findall(x):
        x = re.sub(shouji_pattern, ' 手机number ', x)
    
    huhao_pattern = re.compile('\s\d{10}\s|\s\d{10}\Z')
    if huhao_pattern.findall(x):
        x = re.sub(huhao_pattern, ' 户号number ', x)
            
    tuiding_pattern = re.compile('\s\d{11}\s|\s\d{11}\Z')
    if tuiding_pattern.findall(x):
        x = re.sub(tuiding_pattern, ' 退订number ', x)
            
    gongdan_pattern = re.compile('\s201\d{13}\s|\s201\d{13}\Z')
    if gongdan_pattern.findall(x):
        x = re.sub(gongdan_pattern, ' 工单number ', x)
            
    tingdian_pattern = re.compile('\s\d{12}\s|\s\d{12}\Z')
    if tingdian_pattern.findall(x):
        x = re.sub(tingdian_pattern, ' 停电number ', x)
        
    return x.strip()
jobinfo['contents'] = jobinfo['contents'].apply(lambda x:hash_number(x))

text = df[['CUST_NO', 'counts_of_jobinfo']].copy()
text['contents'] = text.CUST_NO.map(jobinfo.groupby('CUST_NO').contents.apply(lambda x:' '.join(x)))

jobinfo['len_of_contents'] = jobinfo.contents.apply(lambda x:len(x.split()))
jobinfo['counts_of_words'] = jobinfo.contents.apply(lambda x:len(set(x.split())))

text['max_len_of_content'] = text.CUST_NO.map(jobinfo.groupby('CUST_NO').len_of_contents.max())
text['min_len_of_content'] = text.CUST_NO.map(jobinfo.groupby('CUST_NO').len_of_contents.min())
text['sum_len_of_content'] = text.CUST_NO.map(jobinfo.groupby('CUST_NO').len_of_contents.sum())
text['std_len_of_content'] = text.CUST_NO.map(jobinfo.groupby('CUST_NO').len_of_contents.std())
text['median_len_of_content'] = text.CUST_NO.map(jobinfo.groupby('CUST_NO').len_of_contents.median())
text['mean_len_of_content'] = text.sum_len_of_content / text.counts_of_jobinfo

text['sum_counts_of_words'] = text.contents.apply(lambda x:len(set(x.split())))
text['mean_counts_of_words'] = text.sum_counts_of_words / text.counts_of_jobinfo

text.drop(['counts_of_jobinfo'], axis=1, inplace=1)

pickle.dump(text, open('../myfeatures/text_features_2.pkl', 'wb'))
print('done!')