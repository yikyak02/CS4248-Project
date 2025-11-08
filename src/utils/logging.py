import csv
import os
from typing import Dict, Any

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class CSVLogger:
    def __init__(self, filepath: str, fieldnames=None):
        self.filepath = filepath
        self.fieldnames = fieldnames or ["step", "epoch", "loss", "lr"]
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)