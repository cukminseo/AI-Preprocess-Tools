import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
from tqdm import tqdm

classes = [
        "1mg/1L",
        "5mg/1L",
        "15mg/1L",
        " 1mg/50mL ",
        " 5mg/50mL ",
        " 15mg/50mL ",
    ]
folder_path = "./effi"
save_folder_path = "./effi_change"

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# file_list = os.listdir(folder_path)
file_list = [
    f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".pth")
]

for file in tqdm(file_list, desc="draw:"):
    cm = torch.load(f"{folder_path}/{file}", map_location="cpu")

    dpi_val = 68.84
    plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)

    # Seaborn 설정
    sn.set_context(font_scale=1)
    cm_numpy = cm.cpu().numpy()  # Torch tensor를 Numpy 배열로 변환
    df_cm = pd.DataFrame(
        cm_numpy / np.sum(cm_numpy, axis=1)[:, np.newaxis],
        index=classes,
        columns=classes,
    )
    # Heatmap 그리기

    cax = sn.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 35,"weight": "bold"}, cbar=True)  # heatmap의 숫자 글자 크기 조절

    # 컬러바 객체 얻기
    cbar = cax.collections[0].colorbar

    # 컬러바 눈금 라벨의 글자 크기 설정
    cbar.ax.tick_params(labelsize=30)
    for label in cbar.ax.get_yticklabels():
        label.set_weight('bold')

    # plt.xticks(fontsize=25, weight='bold', rotation=20)  # x축 틱 레이블의 글자 크기를 14로 설정
    # plt.yticks(fontsize=25, weight='bold', rotation=20)  # y축 틱 레이블의 글자 크기를 14로 설정
    plt.xticks(fontsize=18, weight='bold')  # x축 틱 레이블의 글자 크기를 14로 설정
    plt.yticks(fontsize=18, weight='bold')  # y축 틱 레이블의 글자 크기를 14로 설정

    # 특정 레이블의 글자 크기 변경
    # ax = plt.gca()  # 현재 축 가져오기
    # xlabels = ax.get_xticklabels()  # x축 레이블 리스트
    # ylabels = ax.get_yticklabels()  # y축 레이블 리스트
    #
    # z`# 첫 번째 레이블의 글자 크기 변경
    # xlabels[0].set_fontsize(20)
    # xlabels[1].set_fontsize(20)
    # xlabels[2].set_fontsize(20)
    # ylabels[0].set_fontsize(20)
    # ylabels[1].set_fontsize(20)
    # ylabels[2].set_fontsize(20)
    #
    # # 변경된 레이블 설정
    # ax.set_xticklabels(xlabels)
    # ax.set_yticklabels(ylabels)


    plt.xlabel("Predicted labels", labelpad=20, fontsize=35, weight='bold')
    plt.ylabel("True labels", labelpad=20, fontsize=35, weight='bold')
    # plt.title("Confusion Matrix", fontsize=16)   # 제목 글자 크기 조절
    plt.tight_layout()
    plt.subplots_adjust(right=1.02)

    save_path = os.path.join(save_folder_path, file.replace(".pth", ".png"))
    plt.savefig(save_path)

    # 그림을 닫아 리소스를 해제
    plt.close()

#
# cm = plot_confusion_matrix(c_mat.cpu().numpy())
#
# if not os.path.exists("./confusion_matrix_remake"):
#     os.makedirs("./confusion_matrix_remake")
#
# plt.savefig(
#     "./confusion_matrix/epoch{}_{}_fold{}_batch{}_lr{}_acc{:.2f}.png".format(
#         epoch,
#         config["model"],
#         i + 1,
#         config["batch_size"],
#         config["lr"],
#         eval_result["acc"],
#     )
# )
