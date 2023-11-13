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


def data_cleansing(data, threshold):
    if 'Time' in data.columns:
        data.set_index('Time', inplace=True)

    '''
    정제전 데이터 볼려면 threshold 최대로
    1:정제 후 평균 출력
    2:정제 후 z-score 출력
    3:정제 후 데이터 출력
    '''
    mode = 3
    # threshold = 9999

    mean_values = data.mean(axis=1)
    cnt_all = 0
    cnt_sel = 0

    # plt.plot(mean_values, label='Average')

    plt.figure(figsize=(15, 8))
    for column in data.columns:
        data_mean = data[column] - mean_values
        z_scores = data_mean / data[column].std()
        cnt_all += 1
        print(data_mean.max())
        if data_mean.abs().max() < threshold:
            if mode == 1:
                plt.plot(data_mean, label=data_mean)
            elif mode == 2:
                plt.plot(z_scores, label=z_scores)
            elif mode == 3:
                plt.plot(data[column], label=data)
            cnt_sel += 1
    print(f"전체 데이터 갯수 : {cnt_all}")
    print(f"정제 후 데이터 갯수 : {cnt_sel}")
    plt.title('Time Series Data Visualization', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    Data_path = "D:\\blood_quality_prediction\\data\\data_origin\\JL_230829_ML6\\145_output"
    data = pd.read_csv(os.path.join(Data_path, '2.csv'))
    data = data.rename(columns={'Unnamed: 170': '170'})
    data_cleansing(data, 0.03)
