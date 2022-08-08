import os
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

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
msno.matrix(df_water)                          # fw_1018608은 버리자.

# line plot
plt.plot(range(len(df_water.iloc[:,:])),df_water.iloc[:,[6,8,10,12]]) # index 길어서 plotting 오래걸림. x축을 걍 range로 대체
plt.plot(range(len(df_water.iloc[:,:])),df_water.iloc[:,0:6])
plt.plot(range(len(df_water.iloc[:6*24*365,:])),df_rf.iloc[:6*24*365,:])

# # 일단 nan은 평균으로 대체한 뒤 eda
# df_water.fillna(df_water.mean(),inplace=True)

# data = pd.concat((df_water, df_rf), axis=1)

# del(df_rf, df_water)

# # data preprocessing
# del(data['fw_1018680'])

# data_submission = data.copy(deep=True)