'''
训练低敏感度用户的模型并预测敏感用户

@Author：Fei peng
@E-mail：dlutfeipeng@gmail.com
'''
import pickle
import pandas as pd
import numpy as np
import os
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

def threshold(y, t):
    z = np.copy(y)
    z[z>=t] = 1
    z[z<t] = 0
    return z
print('training model...')
df = pickle.load(open('../myfeatures/statistical_features_1.pkl', 'rb'))
text = pickle.load(open('../myfeatures/text_features_1.pkl', 'rb'))
df = df.merge(text, on='CUST_NO', how='left')

train = df.loc[df.label != -1]
test = df.loc[df.label == -1]
print('训练集：',train.shape[0])
print('正样本:',train.loc[train.label == 1].shape[0])
print('负样本:',train.loc[train.label == 0].shape[0])
print('-----------------------')
print('测试集：',test.shape[0])
print('-----------------------')

#############################
x_data = train.copy()
x_val = test.copy()
x_data = x_data.sample(frac=1, random_state=1).reset_index(drop=True)
#################
#################
###   input   ###
#################
#################
delete_columns = ['CUST_NO', 'label', 'contents']

X_train_1 = csc_matrix(x_data.drop(delete_columns, axis=1).as_matrix())
X_val_1 = csc_matrix(x_val.drop(delete_columns, axis=1).as_matrix())

y_train = x_data.label.values
y_val = x_val.label.values
featurenames = list(x_data.drop(delete_columns, axis=1).columns)
print('tfidf...')
select_words = pickle.load(open('../myfeatures/single_select_words.pkl', 'rb'))
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, use_idf=False, smooth_idf=False, sublinear_tf=True, vocabulary=select_words)
tfidf.fit(x_data.contents)
word_names = tfidf.get_feature_names()
X_train_2 = tfidf.transform(x_data.contents)
X_val_2 = tfidf.transform(x_val.contents)
print('文本特征：{}维'.format(len(word_names)))
statistic_feature = featurenames.copy()
print('其他特征：{}维'.format(len(statistic_feature)))
featurenames.extend(word_names)
from scipy.sparse import hstack
X_train = hstack(((X_train_1), (X_train_2))).tocsc()
X_val = hstack(((X_val_1), (X_val_2))).tocsc()

print('特征数量',X_train.shape[1])
print('----------------------------------------------')
print('start 5 xgboost!')
bagging = []
for i in range(1,6):
    print('group:',i)
    ##############
    #  xgboost
    ##############
    print('training...')
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=featurenames)
    dval = xgb.DMatrix(X_val, feature_names=featurenames)

    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "error",
        "eta": 0.1,
        'max_depth':14,
        'subsample':0.8,
        'min_child_weight':2,
        'colsample_bytree':1,
        'gamma':0.2,
        "lambda":300,
        'silent':1,
        "seed":i,
    }
    watchlist = [(dtrain, 'train')]

    model = xgb.train(params, dtrain, 2000, evals=watchlist,
                    early_stopping_rounds=50, verbose_eval=100)

    print('predicting...')
    y_prob = model.predict(dval, ntree_limit=model.best_ntree_limit)
    bagging.append(y_prob)
    print('---------------------------------------------')
print('gl hf !')

print('voting...')
t = 0.5
pres = []
for i in bagging:
    pres.append(threshold(i, t))
    
# vote
pres = np.array(pres).T.astype('int64')
result = []
for line in pres:
    result.append(np.bincount(line).argmax())
    
myout = test[['CUST_NO']].copy()
myout['pre'] = result
print('output!')
if not os.path.isdir('../result'):
    os.makedirs('../result')
myout.loc[myout.pre == 1, 'CUST_NO'].to_csv('../result/A.csv', index=False)