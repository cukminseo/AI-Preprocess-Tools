import os
import pandas as pd
from preprocess_fn import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm


# All_data_path = "D:\\water_quality_prediction\\data\\origin_Fe203"
# Data_path = "D:\\water_quality_prediction\\data\\origin_Fe203\\15mg_1L"
All_data_path = "D:\\blood_quality_prediction\\data\\data_origin\\JL_230829_ML6"
Data_path = "D:\\blood_quality_prediction\\data\\data_origin\\JL_230829_ML6\\250_output"
Graph_save_path = "D:\\AI-Preprocessing-and-Visualization-Tools\\01.pre_processing\\250_output"
set_folder(Graph_save_path)
allfiles = os.listdir(All_data_path)
files = os.listdir(Data_path)
select_csv = '1.csv'


def draw_individual_graphs():
    for file in files:
        if file.endswith('.csv'):
            filepath = os.path.join(Data_path, file)
            data = pd.read_csv(filepath)

            if 'Time' in data.columns:
                data.set_index('Time', inplace=True)

            plt.figure(figsize=(15, 8))
            for column in data.columns:
                plt.plot(data[column], label=column, alpha=0.1)

            plt.title(f'Time Series Data Visualization for {file}', fontsize=18)
            plt.xlabel('Time', fontsize=18)
            plt.ylabel('Value', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(os.path.join(Graph_save_path, f"{file.split('.')[0]}_graph.png"))
            plt.close()


def draw_combined_graph():
    combined_data = None
    for file in files:
        if file.endswith('.csv'):
            filepath = os.path.join(Data_path, file)
            data = pd.read_csv(filepath)

            if 'Time' in data.columns:
                data.set_index('Time', inplace=True)

            if combined_data is None:
                combined_data = data
            else:
                combined_data = combined_data.merge(data, left_index=True, right_index=True, how='outer',
                                                    suffixes=('', f'_{file}'))
    plt.figure(figsize=(15, 8))
    for column in combined_data.columns:
        plt.plot(combined_data[column], label=column, alpha=0.1)

    plt.title('Time Series Data Visualization', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_save_path, "combined_graph.png"))
    plt.close()



def draw_all_combined_graph():
    print(allfiles)
    all_combined_data = []
    for files in allfiles:
        combined_data = None
        print(files)
        for file in os.listdir(os.path.join(All_data_path,files)):
            if file.endswith('.csv'):
                filepath = os.path.join(Data_path, file)
                data = pd.read_csv(filepath)

                if 'Time' in data.columns:
                    data.set_index('Time', inplace=True)

                if combined_data is None:
                    combined_data = data
                else:
                    combined_data = combined_data.merge(data, left_index=True, right_index=True, how='outer',
                                                        suffixes=('', f'_{file}'))

        all_combined_data.append(combined_data)
    plt.figure(figsize=(15, 8))

    colors_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9edae5', '#dbdb8d',
        '#c7c7c7', '#c49c94', '#c5b0d5', '#f7b6d2', '#c7e9c0', '#bcbd22',
        '#dbdb8d', '#17becf', '#9edae5', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7e9c0', '#bcbd22', '#dbdb8d', '#17becf'
    ]

    for idx, df in enumerate(tqdm(all_combined_data, desc="Processing dataframes")):
        for column in df.columns:
            plt.plot(df[column], label=column, color=colors_list[idx % len(colors_list)], alpha=0.1)

    plt.title('Time Series Data Visualization', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_save_path, "all_combined_graph.png"))
    plt.close()



def draw_single_graph(select):
    data = pd.read_csv(os.path.join(Data_path, select))

    if 'Time' in data.columns:
        data.set_index('Time', inplace=True)

    plt.figure(figsize=(15, 8))
    for column in data.columns:
        plt.plot(data[column], label=column)

    plt.title('Time Series Data Visualization', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_save_path, "graph.png"))
    plt.close()


if __name__ == '__main__':
    # draw_single_graph(select_csv)
    draw_individual_graphs()
    draw_combined_graph()
    # draw_all_combined_graph()
