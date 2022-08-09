import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

'''Setting------------------------------------------------------------------'''

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

del(df_rf, df_water)

# data preprocessing
del(data['fw_1018680'])

data_submission = data.copy(deep=True)

'''sliding window 없는 버전'''
#X = data.iloc[:-1,[0,1,2,3,4,5,7,10,12,13,14,15]]
#y = data.iloc[1:,[6,8,9,11]]
#
#ms = MinMaxScaler()
#ss = StandardScaler()
#
#X_ss = ss.fit_transform(X)
#y_ms = ms.fit_transform(y)
#
#k = len(data[data.index>='2022-06-01'])
#
#X_train = X_ss[:-k,:]
#X_test = X_ss[-k:,:]
#
#y_train = y_ms[:-k,:]
#y_test = y_ms[-k:,:]
#
#print('Training Shape', X_train.shape, y_train.shape)
#print('Testing Shape', X_test.shape, y_test.shape)
#
#X_train_tensors = Variable(torch.Tensor(X_train))
#X_test_tensors = Variable(torch.Tensor(X_test))
#
#y_train_tensors = Variable(torch.Tensor(y_train))
#y_test_tensors = Variable(torch.Tensor(y_test))
#
#X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0],
#                                                    1,X_train_tensors.shape[1]))
#X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0],
#                                                    1,X_test_tensors.shape[1]))

'''preprocessing------------------------------------------------------------'''

class SlidingWindow(Dataset) :
    def __init__(self, data, window) :
        self.data = data
        self.window = window
        
    def __getitem__(self, index) :
        x = self.data[index:index+self.window]
        return x
    
    def __len__(self) :
        return len(self.data) - self.window


# data setting
# submit 해야하는 기간 제외
k = len(data[data.index>='2022-06-01'])
data = data.iloc[:-k,:]

X = data.iloc[:,[0,1,2,3,4,5,7,10,12,13,14,15]]
y = data.iloc[:,[6,8,9,11]]

# train_test_split
k = int(len(data)*0.8)
train_X = X.iloc[:k,:]
test_X = X.iloc[k:,:]

train_y = y.iloc[:k,:]
test_y = y.iloc[k:,:]

# sliding window에 맞게 데이터 조정 (필요없는 앞, 뒤 잘라냄)
window_size = 144*2

train_y = train_y.iloc[window_size:,:]
test_y = test_y.iloc[window_size:,:]

# 각 y Variable 적용
train_y = np.array(train_y, dtype=float)
test_y = np.array(test_y, dtype=float)

y_train_tensors = Variable(torch.Tensor(train_y))
y_test_tensors = Variable(torch.Tensor(test_y))


# 각 X DataLoader 생성
train_X = torch.tensor(np.array(train_X, dtype=float))
test_X = torch.tensor(np.array(test_X, dtype=float))

train_x_dataset = SlidingWindow(data=train_X, window=window_size)
train_dl = DataLoader(dataset=train_x_dataset,
           batch_size=1,
           shuffle=False)

test_x_dataset = SlidingWindow(data=test_X, window=window_size)
test_dl = DataLoader(dataset=test_x_dataset,
           batch_size=1,
           shuffle=False)

'''Model_training-----------------------------------------------------------'''
class LSTM(nn.Module) :
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) :
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 128)
        self.fc = nn.Linear(128,num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x) :
        if torch.cuda.is_available() :
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        else :
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

# training settings
num_epochs = 20
learning_rate = 0.001

input_size = 12
hidden_size = 512
num_layers = 1

num_classes = 4
seq_length = window_size

model = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
if torch.cuda.is_available() :
    model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs) :
    for i, window in enumerate(train_dl):
        window = window.type(torch.float32)
        
        if torch.cuda.is_available() :
            window = Variable(window.cuda())
        else : 
            window = Variable(window)
        optimizer.zero_grad()
        outputs = model.forward(window)
        loss = criterion(outputs, y_train_tensors[i].view(1,4))
        if torch.cuda.is_available() :
            loss.cuda()
        
        loss.backward()
    
        optimizer.step()
        #print("window step : %d, loss : %1.5f" %(i, loss.item()))
    print("Epoch : %d, loss : %1.5f" %(epoch, loss.item()))


'''Model_validation---------------------------------------------------------'''

# validation
for i, window in enumerate(test_dl) :
    window = window.type(torch.float32)
    window = Variable(window)
    outputs = model.forward(window)
    loss = criterion(outputs, y_test_tensors[i].view(1,4))
    print("MSE of Validation : %1.5f" % loss.item())
    
print("MSE of Validation : %1.5f" % loss.item())

'''Extract result-----------------------------------------------------------'''

# DataLoader 생성
data_submission = data_submission[data_submission.index >= '2022-05-31']
submit_X = data_submission.iloc[:,[0,1,2,3,4,5,7,10,12,13,14,15]]

submit_X = torch.tensor(np.array(submit_X, dtype=float))
submit_x_dataset = SlidingWindow(data=submit_X, window=window_size)
submit_dl = DataLoader(dataset=submit_x_dataset,
           batch_size=1,
           shuffle=False)

# extract result
result = []

for i, window in enumerate(submit_dl) :
    window = window.type(torch.float32)
    window = Variable(window)
    outputs = model.forward(window)
    result.append([i.item() for i in outputs[0]])


# save result to csv file
result = np.array(result)

submission.iloc[:,:] = result

submission.to_csv("result_window.csv")
