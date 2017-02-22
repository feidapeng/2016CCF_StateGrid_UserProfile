'''
构造低敏感度用户的特征
“低敏感度用户”定义：在表1 “95598工单” 中只有1条通话记录的用户

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

print('Part I:')
# --------------------------------------------load data---------------------------------------------------
print('@processing data...')
with codecs.open(data_path + file_jobinfo_train, 'r', encoding='utf-8') as fin,\
    codecs.open(data_path + 'processed_' + file_jobinfo_train, 'w', encoding='utf-8') as fout:
        for index, line in enumerate(fin):
            items = line.strip().split('\t')
            for i, item in enumerate(items):
                item = item.strip()
                if i < 12:
                    fout.write(item + '\t')
                else:
                    fout.write(item + '\n')
        print('{} lines in train_95598'.format(index))
with codecs.open(data_path + file_jobinfo_test, 'r', encoding='utf-8') as fin,\
    codecs.open(data_path + 'processed_' + file_jobinfo_test, 'w', encoding='utf-8') as fout:
        for index, line in enumerate(fin):
            items = line.strip().split('\t')
            for i, item in enumerate(items):
                item = item.strip()
                if i < 12:
                    fout.write(item + '\t')
                else:
                    fout.write(item + '\n')
        print('{} lines in test_95598'.format(index))
        
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
del temp, train, test
##################################
##################################
# 只保留一条工单信息的低敏感度用户
##################################
##################################
df = df.loc[df.counts_of_jobinfo == 1].copy()
df.reset_index(drop=True, inplace=1)
train = df.loc[df.label != -1]
test = df.loc[df.label == -1]
print('原始数据中的低敏感度用户分布情况如下：')
print('训练集：',train.shape[0])
print('正样本:',train.loc[train.label == 1].shape[0])
print('负样本:',train.loc[train.label == 0].shape[0])
print('-----------------------')
print('测试集：',test.shape[0])
df.drop(['counts_of_jobinfo'], axis=1, inplace=1)


# --------------------------------------------create features---------------------------------------------------
# 读取表2
# 1. 没有表2信息的用户全是非敏感用户,剔除掉，其中训练集 1548个，测试集 1267个
# 2. 6个用户，其中训练集3个，测试集3个，REQ_BEGIN_DATE > REQ_FINISH_DATE. 属于数据异常，剔除掉
print('@creating features...')
# 合并工单
jobinfo = train_info.append(test_info).copy()
jobinfo = jobinfo.loc[jobinfo.CUST_NO.isin(df.CUST_NO)].copy()
jobinfo.reset_index(drop=True, inplace=1)
jobinfo = jobinfo.merge(df[['CUST_NO', 'label']], on='CUST_NO', how='left')
######################################################################################################
print('处理表2...')
comm = pd.read_csv(data_path + file_comm, sep='\t')
comm.drop_duplicates(inplace=1)
comm = comm.loc[comm.APP_NO.isin(jobinfo.ID)]
comm = comm.rename(columns={'APP_NO':'ID'})
comm = comm.merge(jobinfo[['ID', 'CUST_NO']], on='ID', how='left')
comm['REQ_BEGIN_DATE'] = comm.REQ_BEGIN_DATE.apply(lambda x:pd.to_datetime(x))
comm['REQ_FINISH_DATE'] = comm.REQ_FINISH_DATE.apply(lambda x:pd.to_datetime(x))

# 过滤
comm = comm.loc[~(comm.REQ_BEGIN_DATE > comm.REQ_FINISH_DATE)]
df = df.loc[df.CUST_NO.isin(comm.CUST_NO)].copy()
comm['holding_time'] = comm['REQ_FINISH_DATE'] - comm['REQ_BEGIN_DATE']
comm['holding_time_seconds'] = comm.holding_time.apply(lambda x:x.seconds)
# add
df = df.merge(comm[['CUST_NO', 'holding_time_seconds']], how='left', on='CUST_NO')
# 通话时间归一化，秒
df['holding_time_seconds'] = MinMaxScaler().fit_transform(df['holding_time_seconds'])
del comm
#####################################################################################################
print('处理表1...')
jobinfo = jobinfo.loc[jobinfo.CUST_NO.isin(df.CUST_NO)].copy()
jobinfo.reset_index(drop=True, inplace=1)
###############
#  CUST_NO
###############
# rank
df['rank_CUST_NO'] =df.CUST_NO.rank(method='max')
df['rank_CUST_NO'] = MinMaxScaler().fit_transform(df.rank_CUST_NO)
###############
# BUSI_TYPE_CODE
###############
# one-hot
df = df.merge(jobinfo[['CUST_NO', 'BUSI_TYPE_CODE']], on='CUST_NO', how='left')
temp = pd.get_dummies(df.BUSI_TYPE_CODE, prefix='onehot_BUSI_TYPE_CODE', dummy_na=True)
df = pd.concat([df, temp], axis=1)
df.drop(['BUSI_TYPE_CODE'], axis=1, inplace=1)
del temp
###############
# URBAN_RURAL_FLAG
###############
# one-hot
df = df.merge(jobinfo[['CUST_NO', 'URBAN_RURAL_FLAG']], on='CUST_NO', how='left')
temp = pd.get_dummies(df.URBAN_RURAL_FLAG, prefix='onehot_URBAN_RURAL_FLAG', dummy_na=True)
df = pd.concat([df, temp], axis=1)
df.drop(['URBAN_RURAL_FLAG'], axis=1, inplace=1)
del temp
###############
# ORG_NO
###############
# ORG_NO 编码
# 12个一级编码，5位数字：33401，33402，.... ,33410， 33411， 33420 
# 75个二级编码，7位数字：前缀（33401，33402，.... ,33410， 33411）
# 96个三级编码，9位数字：前缀（33401，33402，.... ,33410， 33411）
# 1个四级编码，11位数字：33406400142
# add
df = df.merge(jobinfo[['CUST_NO', 'ORG_NO']], on='CUST_NO', how='left')
df['len_of_ORG_NO'] = df.ORG_NO.apply(lambda x:len(str(x)))
df.fillna(-1, inplace=1)
# 1.ratio
train = df[df.label != -1]
ratio = {}
for i in train.ORG_NO.unique():
    ratio[i] = len(train.loc[(train.ORG_NO == i) & (train.label == 1)]) / len(train.loc[train.ORG_NO == i])
df['ratio_ORG_NO'] = df.ORG_NO.map(ratio)
# 2.one-hot
temp = pd.get_dummies(df.len_of_ORG_NO, prefix='onehot_len_of_ORG_NO')
df = pd.concat([df, temp], axis=1)
# drop
df.drop(['ORG_NO', 'len_of_ORG_NO'], axis=1, inplace=1)
###############
# HANDLE_TIME
###############
# add
df = df.merge(jobinfo[['CUST_NO', 'HANDLE_TIME']], on='CUST_NO', how='left')
df['date'] = df['HANDLE_TIME'].apply(lambda x:pd.to_datetime(x.split()[0]))
df['time'] = df['HANDLE_TIME'].apply(lambda x:x.split()[1])
# month
# 1.label encoder
df['month'] = df['date'].apply(lambda x:x.month)
# day 一个月的第几天
# 1.label encoder
df['day'] = df.date.apply(lambda x:x.day)
# 2. 上旬，中旬，下旬
df['is_in_first_tendays'] = 0
df.loc[df.day.isin(range(1,11)), 'is_in_first_tendays'] = 1
df['is_in_middle_tendays'] = 0
df.loc[df.day.isin(range(11,21)), 'is_in_middle_tendays'] = 1
df['is_in_last_tendays'] = 0
df.loc[df.day.isin(range(21,32)), 'is_in_last_tendays'] = 1
# hour
# 1.label encoder
df['hour'] = df.time.apply(lambda x:int(x.split(':')[0]))
# drop
df.drop(['HANDLE_TIME', 'date', 'time'], axis=1, inplace=1)
###############
# ELEC_TYPE
###############
# add
df = df.merge(jobinfo[['CUST_NO', 'ELEC_TYPE']], on='CUST_NO', how='left')
df.fillna(0,inplace=1)
df['head_of_ELEC_TYPE'] = df.ELEC_TYPE.apply(lambda x:str(x)[0])
# 是否是空值
df['is_ELEC_TYPE_NaN'] = 0
df.loc[df.ELEC_TYPE == 0, 'is_ELEC_TYPE_NaN'] = 1
# 1.label encoder
df['label_encoder_ELEC_TYPE'] = LabelEncoder().fit_transform(df['ELEC_TYPE'])
# 2.ratio 
train = df[df.label != -1]
ratio = {}
for i in train.ELEC_TYPE.unique():
    ratio[i] = len(train.loc[(train.ELEC_TYPE == i) & (train.label == 1)]) / len(train.loc[train.ELEC_TYPE == i])
df['ratio_ELEC_TYPE'] = df.ELEC_TYPE.map(ratio)
df.fillna(0,inplace=1)
# 3.用电类别第一位数字one-hot
temp = pd.get_dummies(df.head_of_ELEC_TYPE, prefix='onehot_head_of_ELEC_TYPE')
df = pd.concat([df, temp], axis=1)
# drop
df.drop(['ELEC_TYPE', 'head_of_ELEC_TYPE'], axis=1, inplace=1)
###############
# CITY_ORG_NO
###############
# add
df = df.merge(jobinfo[['CUST_NO', 'CITY_ORG_NO']], on='CUST_NO', how='left')
# 1.ratio CITY_ORG_NO
train = df[df.label != -1]
ratio = {}
for i in train.CITY_ORG_NO.unique():
    ratio[i] = len(train.loc[(train.CITY_ORG_NO == i) & (train.label == 1)]) / len(train.loc[train.CITY_ORG_NO == i])
df['ratio_CITY_ORG_NO'] = df.CITY_ORG_NO.map(ratio)
# 2.one-hot
temp = pd.get_dummies(df.CITY_ORG_NO, prefix='onehot_CITY_ORG_NO')
df = pd.concat([df, temp], axis=1)
# drop
df.drop(['CITY_ORG_NO'], axis=1, inplace=1)
#####################################################################################################
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
#####################################################################################################
if not os.path.isdir('../myfeatures'):
    os.makedirs('../myfeatures')
    
print('统计特征搞定！')
pickle.dump(df, open('../myfeatures/statistical_features_1.pkl', 'wb'))
#####################################################################################################
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
jobinfo['content'] = jobinfo['contents'].apply(lambda x:hash_number(x))

jobinfo['len_of_contents'] = jobinfo.content.apply(lambda x:len(x.split()))
jobinfo['counts_of_words'] = jobinfo.content.apply(lambda x:len(set(x.split())))
text = df[['CUST_NO']].copy()
text = text.merge(jobinfo[['CUST_NO', 'len_of_contents', 'counts_of_words', 'content']], on='CUST_NO', how='left')
text = text.rename(columns={'content': 'contents'})

pickle.dump(text, open('../myfeatures/text_features_1.pkl', 'wb'))
print('done!')