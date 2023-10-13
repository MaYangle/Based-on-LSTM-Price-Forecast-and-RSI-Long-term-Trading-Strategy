import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
fig, ax = plt.subplots()

import pandas as pd

bit_data = pd.read_csv('./data/BCHAIN-MKPRU.csv') 
bit_data.head()

Yb_data = bit_data['Value'].values.astype(float)
Xb_data = bit_data['Date'].values.astype(str)

gold_data = pd.read_csv('./data/LBMA-GOLD.csv') 
gold_data.head()

Yg_data = gold_data['USD (PM)'].values.astype(float)
Xg_data = gold_data['Date'].values.astype(str)


A=[0]*1826

for i in range(1265):
    for j in range(1826):
        if Xg_data[i] == Xb_data[j]:
            A[j]=Yg_data[i]
b=[Yb_data,A]
b=np.transpose(b)
np.savetxt(fname='./data/data.csv',X=b,fmt='%.2f',delimiter=' ',header='USD (PM)')
c = pd.read_csv('./data/data-A.csv') 

plt.title('GOLD AND BITCOIN')
plt.ylabel('Value')
plt.xlabel('Date')
plt.grid(True)
plt.plot(bit_data['Date'],bit_data['Value'])
fmt_half_year = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(fmt_half_year)
plt.plot(bit_data['Date'],c['USD (PM)'])

plt.show()



