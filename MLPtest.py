import os
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import sklearn.metrics as metrics

os.chdir("C:/Users/user/Desktop/water_level_pred")

def read_csv_by_dir(path, index_col=None):
    df_raw = pd.DataFrame()
    for files in os.listdir(path):
        if files.endswith('.csv'):
            df = pd.read_csv('/'.join([path,files]),
                            index_col=index_col)
        df_raw = pd.concat((df_raw,df),axis=0)
    return df_raw

path = 'C:/Users/user/Desktop/water_level_pred/competition_data'
_df_rf_raw = read_csv_by_dir('/'.join([path,'rf_data']),
                            index_col=0)

_df_water_raw = read_csv_by_dir('/'.join([path,'water_data']),
                               index_col=0)

_submission_raw = pd.read_csv('/'.join([path,'sample_submission.csv']),
                             index_col=0)

# raw_data 보존하기
df_rf=_df_rf_raw.copy()
df_rf.name = "rain_data"

df_water=_df_water_raw.copy()
df_water.name = "water_data"

submission=_submission_raw.copy()
submission.name = "submission"

# 일단 nan은 평균으로 대체한 뒤 성능 확인해보자
df_water.fillna(df_water.mean(),inplace=True)

data = pd.concat((df_water, df_rf), axis=1)

# MLP modeling
data = data[data.index<'2022-06-01']
submit_df = data[data.index>='2022-06-01']

k = int(data.shape[0] * 0.8)

train = data.iloc[:k,:]
test = data.iloc[k:,:]

trainX = train.iloc[:-1,[0,1,2,3,4,5,7,9,11,13,14,15,16]]
trainY = train.iloc[1:,[6,8,10,12]]
testX = test.iloc[:-1,[0,1,2,3,4,5,7,9,11,13,14,15,16]]
testY = test.iloc[1:,[6,8,10,12]]

input_num = 13
output_num = 4

layer1 = [16,32,64,128,256,512,1024,2048]
layer2 = [16,32,64,128,256,512,1024,2048]
layer3 = [16,32,64,128,256,512,1024,2048]
epoch_list = [5,10,15,20]
batchsize_list = [100, 6,6*24,6*24*7,6*24*7*4]

l1, l2, l3, e, b, t, m = [],[],[],[],[],[],[]

for bat in batchsize_list :
    for epoch in epoch_list :
        for layer_1 in layer1 :
            for layer_2 in layer2 :
                for layer_3 in layer3 :
                    if layer_1 < layer_2 :
                        pass
                    elif layer_2 < layer_3 :
                        pass
                    else :
                        for i in range(5) :
                            t.append(i)
                            MLP = keras.Sequential()
                            MLP.add(keras.layers.Input(shape = (input_num,)))
                            MLP.add(keras.layers.Dense(layer_1, activation = "selu"))
                            MLP.add(keras.layers.Dense(layer_2, activation = "selu"))
                            MLP.add(keras.layers.Dense(layer_3, activation = "selu"))
                            MLP.add(keras.layers.Dense(output_num, activation = "relu"))
                            
                            Adam = keras.optimizers.Adam(learning_rate = 0.001)
                            MLP.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])
                            
                            MLP.fit(trainX, trainY, epochs = epoch, batch_size = bat)
                            
                            mse = MLP.evaluate(testX, testY)[1]
                            m.append(mse)
                    
mlp_test = pd.DataFrame(list(zip(l1,l2,l3,e,b,t,m)), columns = ['layer1','layer2','layer3','epochs','batch_size','try','mse'])
mlp_test.to_csv("mlp_test.csv")
