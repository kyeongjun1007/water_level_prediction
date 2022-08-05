import os
import numpy as np
import pandas as pd
import missingno as msno
from tensorflow import keras
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader


os.chdir("C:/Users/Kyeongjun/Desktop/water_level_pred")

def read_csv_by_dir(path, index_col=None):
    df_raw = pd.DataFrame()
    for files in os.listdir(path):
        if files.endswith('.csv'):
            df = pd.read_csv('/'.join([path,files]),
                            index_col=index_col)
        df_raw = pd.concat((df_raw,df),axis=0)
    return df_raw

path = 'C:/Users/Kyeongjun/Desktop/water_level_pred/competition_data'
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

# 결측치 위치 확인
%matplotlib inline

msno.matrix(df_rf)
msno.matrix(df_water)

# 일단 nan은 평균으로 대체한 뒤 성능 확인해보자
df_water.fillna(df_water.mean(),inplace=True)

data = pd.concat((df_water, df_rf), axis=1)

del(df_rf, df_water)
# LSTM modeling

'''LSTM에 sliding window를 적용하려고 dataset을 수정하는데, 데이터 사이즈가 커서 메모리 이슈 발생
30000개씩 나누어 적용해봐도 numpy array를 합치는 과정에서 어차피 메모리 에러 발생 ㅠㅠ'''

# DataLoader를 이용하기 위해서 sliding window가 적용된 데이터셋 가져오는 class 생성!

data = torch.tensor(data.values)

class SlidingWindow(Dataset) :
    def __init__(self, data, window) :
        self.data = data
        self.window = window
        
    def __getitem__(self, index) :
        x = self.data[index:index+self.window]
        return x
    
    def __len__(self) :
        return len(self.data) - self.window
        
dataset = SlidingWindow(data=data, window=144)
dl = DataLoader(dataset=dataset,
           batch_size=1,
           shuffle=False)

for i, k in enumerate(dl) :
    print(k)
    if i==0 :
        break
'''
trainX = train.iloc[:-1, :]
trainY = train.iloc[1:,[6,8,10,12]]
testX = test.iloc[:-1, :]
testY = test.iloc[1:,[6,8,10,12]]

input_num = 17
output_num = 4

LSTM = keras.Sequential()
LSTM.add(keras.layers.Input(shape = (input_num,)))
LSTM.add(keras.layers.Dense(8, activation = "relu"))
LSTM.add(keras.layers.Dense(output_num, activation = "relu"))

Adam = keras.optimizers.Adam(learning_rate = 0.03)
LSTM.compile(optimizer = Adam, loss = 'mse', metrics = ['mse'])

LSTM.fit(trainX, trainY, epochs = 100, batch_size = 144)

LSTM.evaluate(testX, testY)
'''