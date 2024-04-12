import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_from_event(event_path, tag):
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()  # Load the data
    # Extract the scalar of interest
    scalar_events = event_acc.Scalars(tag)
    values = [event.value for event in scalar_events]
    return values
for i in range(3):
    # TensorBoard 로그 폴더들의 위치
    log_path = f"I:/blood/6class/{i}/logs"
    log_folders = [os.path.join(log_path, log_folder) for log_folder in os.listdir(log_path)]
    print(log_path)
    # Initialize dictionaries to store the metrics
    metrics = ["acc", "f1-score", "loss", "precision", "recall"]
    train_metrics = {metric: [] for metric in metrics}
    val_metrics = {metric: [] for metric in metrics}

    # Extract data from TensorBoard logs
    for folder in log_folders:
        for metric in metrics:
            train_tag = f"{metric}/train"
            val_tag = f"{metric}/val"
            val_metrics[metric].append(extract_scalar_from_event(folder, val_tag))
            train_metrics[metric].append(extract_scalar_from_event(folder, train_tag))

    # Compute the last values, mean, and standard deviation for each metric
    for metric in metrics:
        # Extract the last value of each run
        train_metrics[metric] = [val[-1] if val else np.nan for val in train_metrics[metric]]
        val_metrics[metric] = [val[-1] if val else np.nan for val in val_metrics[metric]]

        # Compute mean and standard deviation
        train_mean, train_std = np.nanmean(train_metrics[metric]), np.nanstd(train_metrics[metric])
        val_mean, val_std = np.nanmean(val_metrics[metric]), np.nanstd(val_metrics[metric])

        print(f"{metric.capitalize()}/train: Mean = {train_mean:.5f}, Std = {train_std:.6f}")
        print(f"{metric.capitalize()}/test : Mean = {val_mean:.5f}, Std = {val_std:.6f}")
