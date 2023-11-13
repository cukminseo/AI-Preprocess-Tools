import os

data_name_list = []
y = []
data_path = "./data/specto"
clss = os.listdir(data_path)
for cls in clss:
    print(cls)
    labels = os.listdir(f"./data/specto/{cls}")
    for label in labels:
        for data_name in os.listdir(f"./data/specto/{cls}/{label}"):
            if data_name == "Unnamed":
                continue
            data_name_list.append(data_name)
            y.append(labels.index(label))
print(len(data_name_list))
print(len(y))
