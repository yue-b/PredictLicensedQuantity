#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

dir = '/Users/yue/Desktop/阿里天池大数据比赛/汽车上牌量预测/'
train = pd.read_table(dir + 'train_20171215.txt',engine='python')
test_A = pd.read_table(dir + 'test_A_20171225.txt',engine='python')
test_B = pd.read_table(dir + 'test_B_20171225.txt',engine='python')

actions1 = train.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})
actions1.to_csv('/Users/yue/Desktop/阿里天池大数据比赛/汽车上牌量预测/code/actions1.csv',columns=['date','day_of_week','count1'],index=False,header=True)


from sklearn import svm
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

df_train_target = actions1['count1'].values
df_train_data = actions1.drop(['count1'],axis = 1).values

# 切分数据（训练集和测试集）
train_X,test_X, train_y, test_y = train_test_split(df_train_data,df_train_target,test_size=0.2,random_state=0)
print "GradientBoostingRegressor"  
a = 0.16
gbdt = GradientBoostingRegressor(learning_rate=a).fit(train_X,train_y)
result1 = gbdt.predict(test_X)
print("learning_rate:",a,",",mean_squared_error(result1,test_y))


#预测数据
result_A = gbdt.predict(test_A)
result_B = gbdt.predict(test_B)
tmp_A = DataFrame(result_A,columns=['cnt'])
# print result_A.shape,tmp_A
result = pd.merge(test_A,tmp_A,left_index=True,right_index=True)
result['cnt'] = result['cnt'].apply(np.round)#四舍五入
result['cnt'] = result['cnt'].astype(int)#整数化
print result.shape
# result[result['cnt']<0].loc[0,'cnt']
print result.shape

action2 = result.groupby(['date'],as_index=False)['cnt'].agg({'cnt':np.sum})
action2.to_csv("/Users/yue/Desktop/阿里天池大数据比赛/汽车上牌量预测/result_A.txt",index=False,header=False,sep='	')

plt.plot(train_y,'r',result['cnt'],'b')
plt.show()



