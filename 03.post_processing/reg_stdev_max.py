import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalar_from_event(event_path, tag):
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()  # 데이터 로드
    # 관심 있는 스칼라 추출
    scalar_events = event_acc.Scalars(tag)
    values = [event.value for event in scalar_events]
    steps = [event.step for event in scalar_events]  # 이 부분 추가하여 각 step (에폭) 정보도 가져옴
    return values, steps  # 값과 step 모두 반환


# TensorBoard 로그 폴더들의 위치
log_path = "I:/blood/6class/0/logs"
log_folders = [os.path.join(log_path, log_folder) for log_folder in os.listdir(log_path)]

# Initialize dictionaries to store the metrics
metrics = ["acc", "f1-score", "loss", "precision", "recall"]
train_metrics = {metric: [] for metric in metrics}
val_metrics = {metric: [] for metric in metrics}

# TensorBoard 로그에서 데이터 추출
for folder in log_folders:
    for metric in metrics:
        train_tag = f"{metric}/train"
        val_tag = f"{metric}/val"
        train_metrics[metric].append(extract_scalar_from_event(folder, train_tag)[0])  # 값만 가져옴
        val_metrics[metric].append(extract_scalar_from_event(folder, val_tag)[0])  # 값만 가져옴

# 각 메트릭에 대해 최대 값과 그에 해당하는 에폭 계산
for metric in metrics:
    # 각 실행의 최대 값과 해당 에폭 추출
    train_metrics[metric] = [np.nanmax(val) if val else np.nan for val in train_metrics[metric]]
    val_metrics[metric] = [np.nanmax(val) if val else np.nan for val in val_metrics[metric]]
    train_steps = [np.argmax(val) if val else np.nan for val in train_metrics[metric]]
    val_steps = [np.argmax(val) if val else np.nan for val in val_metrics[metric]]

    # 평균과 표준편차 계산
    train_mean, train_std = np.nanmean(train_metrics[metric]), np.nanstd(train_metrics[metric])
    val_mean, val_std = np.nanmean(val_metrics[metric]), np.nanstd(val_metrics[metric])

    print(f"{metric.capitalize()}/train: 최대값 = {train_mean:.5f}, 표준편차 = {train_std:.6f}, 에폭 = {train_steps}")
    print(f"{metric.capitalize()}/test : 최대값 = {val_mean:.5f}, 표준편차 = {val_std:.6f}, 에폭 = {val_steps}")

