# 1st Place Solution for 2016CCF StateGrid UserProfile

赛题链接：http://www.wid.org.cn/data/science/player/competition/detail/description/242

## 任务介绍

在复赛中，参赛者需要以电力用户的95598工单数据、电量电费营销数据等为基础，综合分析电费敏感客户特征，建立客户电费敏感度模型，对电费敏感用户的敏感程度进行量化评判，帮助供电企业快速、准确的识别电费敏感客户，从而对应的提供有针对性的电费、电量提醒等精细化用电服务。

## 数据下载


## 解决方案

详细解决方案戳这里



## 代码运行说明

按照95598工单记录次数对用户分为两类，分别构造特征和建模。

- [x] 将只有一条95598记录的用户定义为**低敏感度用户**，用A或者single指代
- [x] 将有多条95598记录的用户定义为**高敏感度用户**，用B或者multi指代


### 1.配置说明
程序依赖python3及以下程序包
> * anaconda3
> * xgboost
> * jieba

程序运行需要以下文件
```
/stopwords.txt  停用词表
```
请将原始数据放于下面目录中, 请确保都是utf-8编码格式
```
/rawdata/
    01_arc_s_95598_wkst_train.tsv
    01_arc_s_95598_wkst_test.tsv
    02_s_comm_rec.tsv
    09_arc_a_rcvbl_flow.tsv
    09_arc_a_rcvbl_flow_test.tsv
    train_label.csv
    test_to_predict.csv
```
其余目录作用
```
/code/  用于存放程序代码
/myfeatures/  用于存放程序运行生成的各种特征文件
/result/  用于存放最终的输出结果
```
### 2.运行
确认以上文件存在之后，依次运行：
```
python code/create_features_A.py    # 生成低敏感度用户的特征文件
python code/select_features_A.py    # 采用xgboost对低敏感度用户的文本特征进行筛选
python code/model_A.py              # 训练低敏感度用户的预测模型，及模型融合
python code/create_features_B.py    # 生成高敏感度用户的特征文件
python code/select_features_B.py    # 采用xgboost对高敏感度用户的文本特征进行筛选
python code/model_B.py              # 训练高敏感度用户的预测模型，及模型融合
```
### 3.输出文件说明
程序输出的结果包括`特征文件`和最终`预测结果`两部分：
```
myfeatures/
    statistical_features_1.pkl  低敏感度用户的统计特征
    text_features_1.pkl         低敏感度用户在表1中的ACCEPT_CONTENT文本信息
    single_select_words.pkl     低敏感度用户部分，采用xgboost选择的文本特征
    statistical_features_2.pkl  高敏感度用户的统计特征
    text_features_2.pkl         高敏感度用户在表1中的ACCEPT_CONTENT文本信息
    multi_select_words.pkl      高敏感度用户部分，采用xgboost选择的文本特征
    
result/                 
    A.csv               低敏感度用户中的电费敏感用户
    B.csv               高敏感度用户中的电费敏感用户
    result.csv          合并结果
```




