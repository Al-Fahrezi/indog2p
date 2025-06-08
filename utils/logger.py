# indog2p/utils/logger.py

"""
Custom logger IndoG2P:
- Logging ke file & terminal (dual output)
- Setup log file (auto append/timestamp)
- Fungsi log dengan level (info, warning, error)
- Print summary model (dipakai di train.py dsb)
"""

import logging
import os
import sys
from datetime import datetime

# -------- Logger Class --------
class IndoG2PLogger:
    """
    Logger IndoG2P. Output ke file dan terminal.
    """
    def __init__(self, log_file=None, verbose=True):
        self.log_file = log_file
        self.verbose = verbose
        self.logger = logging.getLogger("IndoG2P")
        self.logger.setLevel(logging.INFO)
        # Remove previous handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        # Terminal handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        # File handler (opsional)
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, msg):
        self.logger.info(msg)
    def warning(self, msg):
        self.logger.warning(msg)
    def error(self, msg):
        self.logger.error(msg)
    def close(self):
        logging.shutdown()

# -------- Fungsi Setup Global Logger --------
_global_logger = None

def setup_logger(log_file=None, verbose=True):
    """
    Setup global IndoG2P logger. Pakai di script utama.
    """
    global _global_logger
    _global_logger = IndoG2PLogger(log_file=log_file, verbose=verbose)
    return _global_logger

def log(msg, level="info"):
    """
    Logging via global IndoG2P logger. Safe dipakai di seluruh script.
    """
    global _global_logger
    if _global_logger is None:
        # Default: init logger tanpa file jika belum setup
        _global_logger = IndoG2PLogger()
    if level == "info":
        _global_logger.info(msg)
    elif level == "warning":
        _global_logger.warning(msg)
    elif level == "error":
        _global_logger.error(msg)

# -------- Print Model Summary --------
def print_model_summary(model):
    """
    Print ringkasan model (jumlah parameter, layer, dsb).
    Dipanggil di train.py/finetune.py.
    """
    n_params = sum(p.numel() for p in model.parameters())
    print("="*35)
    print(f"Model Summary: {model.__class__.__name__}")
    print(f"Total parameters: {n_params:,}")
    print("-"*35)
    for name, param in model.named_parameters():
        print(f"{name:40} {list(param.shape)}")
    print("="*35)

# -------- Fungsi Timestamp --------
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    # Demo/debug
    logger = setup_logger("logs/demo.log")
    log("Logger test: info")
    log("Logger test: warning", level="warning")
    log("Logger test: error", level="error")
    logger.close()
