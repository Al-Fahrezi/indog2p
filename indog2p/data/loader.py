# indog2p/data/loader.py

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class G2PDataset(Dataset):
    """
    Dataset PyTorch untuk G2P IndoG2P.
    File .pkl = list of (input_ids, label_ids)
    File .csv = dua kolom: 'input','label' (opsional)
    """
    def __init__(self, path, is_pickle=True):
        super().__init__()
        self.samples = []
        if is_pickle:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Tidak ditemukan: {path}")
            with open(path, "rb") as f:
                self.samples = pickle.load(f)
        else:
            df = pd.read_csv(path)
            if "input" in df.columns and "label" in df.columns:
                for _, row in df.iterrows():
                    self.samples.append((
                        eval(row["input"]),  # Pastikan simpan list/int, bukan str
                        eval(row["label"])
                    ))
            else:
                raise ValueError("CSV harus punya kolom 'input' dan 'label'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, label = self.samples[idx]
        inp = torch.tensor(inp, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return inp, label

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Utility DataLoader wrapper.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

class G2PInputDataset(Dataset):
    """
    Dataset khusus inference (tanpa label).
    Berisi list input_ids saja.
    """
    def __init__(self, path):
        super().__init__()
        self.samples = []
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.samples = [s[0] if isinstance(s, (tuple, list)) else s for s in data]
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            if "input" in df.columns:
                for _, row in df.iterrows():
                    self.samples.append(eval(row["input"]))
            else:
                raise ValueError("CSV harus punya kolom 'input' untuk inference.")
        else:
            raise ValueError("Ekstensi file harus .pkl atau .csv")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long)

def create_infer_dataloader(dataset, batch_size=32, num_workers=0):
    """
    Utility DataLoader khusus inference.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
