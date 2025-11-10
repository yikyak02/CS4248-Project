import csv
import os
from typing import Dict, Any

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

class CSVLogger:
    def __init__(self, filepath: str, fieldnames=None):
        """
        Simple CSV logger.
        
        Args:
            filepath: Path to CSV file
            fieldnames: List of column names. If None, defaults to common fields.
        """
        self.filepath = filepath
        if fieldnames is None:
            fieldnames = ["step", "epoch", "loss", "lr", "val_loss"]
        self.fieldnames = fieldnames
        
        if not os.path.exists(filepath):
            ensure_dir(os.path.dirname(filepath))
            with open(filepath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                w.writeheader()
        
        self.file = open(filepath, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction='ignore')
    
    def log(self, row: dict):
        """Write a row to the CSV file."""
        self.writer.writerow(row)
        self.file.flush()
    
    def __del__(self):
        """Close file on deletion."""
        if hasattr(self, 'file') and self.file:
            self.file.close()
