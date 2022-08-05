import os
import numpy as np
import pandas as pd
import missingno as msno
from tensorflow import keras
from datetime import datetime

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

# 결측치 위치 확인
%matplotlib inline

msno.matrix(df_rf)
msno.matrix(df_water)

# 일단 nan은 평균으로 대체한 뒤 성능 확인해보자
df_water.fillna(df_water.mean(),inplace=True)

data = pd.concat((df_water, df_rf), axis=1)

# MLP modeling
train = timeseriesData[data.index<'2022-06-01']
test = timeseriesData[data.index>='2022-06-01']


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
