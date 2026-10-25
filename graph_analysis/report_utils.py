import os
import numpy as np
import re


def get_best_epoch_from_logs(logs_path):
    if not os.path.isdir(logs_path):
        raise FileNotFoundError(f"Logs path not found: {logs_path}")
    log_files = [
        filename
        for filename in os.listdir(logs_path)
        if filename.startswith("log_epoch_") and filename.endswith(".txt")
    ]
    log_files.sort(
        key=lambda filename: int(
            filename.removeprefix("log_epoch_").removesuffix(".txt")
        )
    )
    for filename in reversed(log_files):
        filepath = os.path.join(logs_path, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
        matches = re.findall(r"BEST EPOCH:\s*(\d+)", content)
        if matches:
            return int(matches[-1]) - 1
    raise ValueError(f"Could not find 'BEST EPOCH' in logs located at {logs_path}")
