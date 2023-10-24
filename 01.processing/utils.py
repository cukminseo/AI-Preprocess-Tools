import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os


def plot_confusion_matrix(cf_matrix):
    classes = [
        "0_output",
        "50_output",
        "60_output",
        "70_output",
        "75_output",
        "80_output",
        "85_output",
        "90_output",
        "95_output",
        "100_output",
        "105_output",
        "110_output",
        "115_output",
        "120_output",
        "125_output",
        "130_output",
        "135_output",
        "140_output",
        "145_output",
        "150_output",
        "160_output",
        "170_output",
        "180_output",
        "190_output",
        "200_output",
        "210_output",
        "220_output",
        "230_output",
        "240_output",
        "250_output",
    ]

    # classes = [
    #     "Control Group",
    #     "Hypoglycemia",
    #     "Normal",
    #     "Impaired fasting glucose",
    #     "Diabetes mellitus",
    # ]
    # classes=[]
    # data_path = "./data/data_specto/JL_230829_ML6"
    # clss = os.listdir(data_path)
    # for cls in clss:
    #     classes.append(cls)

    dpi_val = 68.84
    plt.figure(figsize=(1024 / dpi_val, 768 / dpi_val), dpi=dpi_val)

    sn.set_context(font_scale=1)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=classes, columns=classes
    )

    return sn.heatmap(df_cm, annot=False).get_figure()


def get_data_list(data_path):
    data_name_list = []
    y = []
    data_path_list = []
    # data_path = "./data/data_specto/JL_230829_ML6"
    clss = os.listdir(data_path)
    for cls in clss:
        print(f"class {clss.index(cls)} : {cls}")
        labels = os.listdir(f"{data_path}/{cls}")
        for label in labels:
            for data_name in os.listdir(f"{data_path}/{cls}/{label}"):
                if data_name == "Unnamed":
                    continue
                data_name_list.append(data_name)
                data_path_list.append(f"{data_path}/{cls}/{label}/{data_name}")
                y.append(clss.index(cls))
    print(f"find {len(data_name_list)} files")
    return [data_name_list, data_path_list, y]
