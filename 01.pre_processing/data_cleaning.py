'''
2023-11-13
시계열 데이터 클린징 코드
튀는 값 제거용도
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm


Data_path = "D:\\blood_quality_prediction\\data\\data_origin\\JL_230829_ML6\\250_output"

data = pd.read_csv(os.path.join(Data_path, '1.csv'))
data = data.rename(columns={'Unnamed: 170': '170'})

if 'Time' in data.columns:
    data.set_index('Time', inplace=True)

mean_values = data.mean(axis=1)
cnt_1=0

cnt_2=0
# plt.plot(mean_values, label='Average')
#
plt.figure(figsize=(15, 8))
for column in data.columns:
    data_mean=data[column]-mean_values
    z_scores = data_mean / data[column].std()
    cnt_1+=1
    print(data_mean.max())
    if data_mean.abs().max()<0.03:
        plt.plot(data[column], label=column)
        cnt_2+=1
print(cnt_1,cnt_2)
plt.title('Time Series Data Visualization', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Value', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()