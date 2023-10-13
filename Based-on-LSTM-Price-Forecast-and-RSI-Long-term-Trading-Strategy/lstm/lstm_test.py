import torch
import torch.nn as nn
import itertools
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots() 

import pandas as pd 
gold_data = pd.read_csv('./data/LBMA-GOLD-A.csv') 
gold_data.head()


all_data = gold_data['USD (PM)'].values.astype(float)
test_data_size = 1235

train_data = all_data[:-test_data_size] #size=60
test_data = all_data[-test_data_size:] #size=1205

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1,1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)




all_data_normalized = scaler.fit_transform(all_data.reshape(-1,1))
all_data_normalized = torch.FloatTensor(all_data_normalized).view(-1)

train_window=5

def create_inout_sequences(input_data,tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized,train_window)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# 模型的训练
epochs = 30

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                            torch.zeros(1,1,model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%5 ==1 :
        print(f'epoch:{i:3} loss:{single_loss.item():10.8f}')
print(f'epoch:{i:3} loss:{single_loss.item():10.8f}')


# 预测：
fut_pre = 1235

test_inputs = train_data_normalized[0:-1].tolist()

'''print(test_inputs)'''

model.eval()
for i in range(fut_pre):
    seq = torch.FloatTensor(all_data_normalized[i+25:train_window+25+i])
    with torch.no_grad():
        model.hidden = (torch.zeros(1,1,model.hidden_layer_size),
                            torch.zeros(1,1,model.hidden_layer_size))

        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[-test_data_size:] ).reshape(-1, 1))
'''print(actual_predictions)'''
A =    list(itertools.chain.from_iterable(actual_predictions)) - all_data[-test_data_size:]
x = np.arange(30, 1265, 1)
'''print(x)'''



ax1 = plt.subplot(2,1,1)

ax2 = plt.subplot(2,1,2)

plt.sca(ax1)
plt.title('Gold')
plt.ylabel('Real vs Predict')

plt.grid(True)
plt.autoscale(axis='x', tight=True)

l1, = plt.plot(gold_data['USD (PM)'][-test_data_size:])
l2, = plt.plot(x,actual_predictions)
plt.legend([l1,l2],["Real data","Predict data"],loc='upper left')

plt.sca(ax2)
plt.title('Difference')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(x,A)

plt.show()
