import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalar_from_event(event_path, tag):
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()  # Load the data
    # Extract the scalar of interest (e.g., "loss" or "accuracy")
    scalar_events = event_acc.Scalars(tag)
    values = [event.value for event in scalar_events]
    return values


# TensorBoard 로그 폴더들의 위치
log_path = "D:\\blood_quality_prediction\\fail\\no14\\logs"
# log_path = "D:\\water_quality_prediction\\output_efficentnet\\33_50ml\\logs"
log_folders = os.listdir(log_path)
for idx, log_folder in enumerate(log_folders):
    log_folders[idx] = os.path.join(log_path, log_folder)
all_acc_values = []
all_precision_values = []
all_recall_values = []
all_f1_values = []
all_loss_values = []

for folder in log_folders:
    all_acc_values.append(extract_scalar_from_event(folder, "acc/val"))
    all_precision_values.append(extract_scalar_from_event(folder, "precision/val"))
    all_recall_values.append(extract_scalar_from_event(folder, "recall/val"))
    all_f1_values.append(extract_scalar_from_event(folder, "f1-score/val"))
    all_loss_values.append(extract_scalar_from_event(folder, "loss/val"))

all_acc_values = [val[-1] if isinstance(val, (list, np.ndarray)) else val for val in all_acc_values]
all_precision_values = [val[-1] if isinstance(val, (list, np.ndarray)) else val for val in all_precision_values]
all_recall_values = [val[-1] if isinstance(val, (list, np.ndarray)) else val for val in all_recall_values]
all_f1_values = [val[-1] if isinstance(val, (list, np.ndarray)) else val for val in all_f1_values]
all_loss_values = [val[-1] if isinstance(val, (list, np.ndarray)) else val for val in all_loss_values]

# 각 메트릭의 평균 및 표준편차 계산
print(f"Accuracy: {np.mean(all_acc_values):.5f} {np.std(all_acc_values):.6f}")
print(f"Precision: {np.mean(all_precision_values):.5f} {np.std(all_precision_values):.6f}")
print(f"Recall: {np.mean(all_recall_values):.5f} {np.std(all_recall_values):.6f}")
print(f"F1: {np.mean(all_f1_values):.5f} {np.std(all_f1_values):.6f}")
print(f"loss: {np.mean(all_loss_values):.5f} {np.std(all_loss_values):.6f}")
