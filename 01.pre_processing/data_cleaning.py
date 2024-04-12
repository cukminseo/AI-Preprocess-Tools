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

import argparse  # commandline arguments parsing
import re

config = argparse.ArgumentParser()  # commandline arguments parsing
config.add_argument("--start", default=1.5, type=float)
config.add_argument("--end", default=2.1, type=float)
config.add_argument("--tol_limit", default=0.008, type=float)

config = config.parse_args()
config = vars(config)  # 딕셔너리 변환

# 변수 지정
start_time = config['start']
end_time = config['end']
tolerance_limit = config['tol_limit']


def data_cleansing(class_folder_path, threshold=0.01):

    # 각 클래스 폴더에 대한 빈 데이터프레임 생성
    df_class = pd.DataFrame()

    files = os.listdir(class_folder_path)
    cnt_all = 0
    cnt_sel = 0

    for file in tqdm(files, total = len(files), leave=False):
        class_file_path = os.path.join(class_folder_path, file)
        df_file = pd.read_csv(class_file_path, index_col='Time')

        # 'Unnamed: 170' 별도 처리 위한 구문
        df_file = df_file.rename(columns={'Unnamed: 170': '170'})

        # 가로로 붙이기
        df_class = pd.concat([df_class, df_file], axis=1)

    # 클래스 평균 구하기
    class_df_mean = df_class.mean(axis=1)


    for file_name in tqdm(files, total = len(files)):
        origin_file_path = os.path.join(class_folder_path, file_name)

        # CSV 파일에서 데이터 로드
        df = pd.read_csv(origin_file_path, index_col='Time')

        # 'Unnamed: 170' 별도 처리 위한 구문
        df = df.rename(columns={'Unnamed: 170': '170'})

        # 범위 한정
        df_lim = df.iloc[int(start_time * 500):int(end_time * 500), :]
        mean_values = class_df_mean.iloc[int(start_time * 500):int(end_time * 500)]

        for column in tqdm(df_lim.columns, desc=f"{file_name}", leave=False):
            cnt_all += 1
            # 평균과의 차이 확인
            data_mean = df_lim[column] - mean_values
            if data_mean.abs().max() < tolerance_limit:
                cnt_sel += 1
                plt.plot(df.index, df[column], label=column, alpha=0.3)

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
    class_path = "X:\\data\\data_origin\\JL_230829_ML6\\60_output"
    data_cleansing(class_path)
