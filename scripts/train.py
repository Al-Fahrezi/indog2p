# scripts/train.py
import os
import sys
import time
import yaml
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

CONFIG_PATH = "config\config.yaml"
VOCAB_PATH = "config/vocab.json"
PROC_DIR = "data/processed/"
CHECKPOINT_DIR = "checkpoints/"
LOG_FILE = "logs/train.log"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Dataset class ===
class G2PDataset(Dataset):
    def __init__(self, encoded_pkl):
        with open(encoded_pkl, "rb") as f:
            self.samples = pickle.load(f)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx][0], dtype=torch.long),
            torch.tensor(self.samples[idx][1], dtype=torch.long),
        )

# === BERT G2P Model (character-level) ===
from transformers import BertConfig, BertForMaskedLM

def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        d = json.load(f)
    return d["token2id"], d["id2token"], d["vocab"]

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"{msg}\n")

def evaluate(model, dataloader, device, pad_token_id):
    model.eval()
    total_loss, total_chars, total_err = 0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    with torch.no_grad():
        for batch in dataloader:
            inp_ids, label_ids = [b.to(device) for b in batch]
            outputs = model(input_ids=inp_ids)
            logits = outputs.logits  # [B, L, V]
            loss = criterion(
                logits.view(-1, logits.size(-1)), label_ids.view(-1)
            )
            total_loss += loss.item() * inp_ids.size(0)

            # Calculate Phoneme Error Rate (PER)
            preds = logits.argmax(dim=-1)
            for pred, label in zip(preds, label_ids):
                pred_list = pred.cpu().numpy().tolist()
                label_list = label.cpu().numpy().tolist()
                # Hapus padding
                pred_str = [i for i in pred_list if i != pad_token_id]
                label_str = [i for i in label_list if i != pad_token_id]
                total_chars += len(label_str)
                total_err += edit_distance(pred_str, label_str)
    per = total_err / max(1, total_chars)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, per

def edit_distance(seq1, seq2):
    # Levenshtein
    dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    for i in range(len(seq1) + 1):
        dp[i][0] = i
    for j in range(len(seq2) + 1):
        dp[0][j] = j
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]
                )
    return dp[-1][-1]

def save_checkpoint(model, optimizer, epoch, best_val_loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }, path)

def main():
    config = load_config()
    token2id, id2token, vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = G2PDataset(os.path.join(PROC_DIR, "train_encoded.pkl"))
    val_set = G2PDataset(os.path.join(PROC_DIR, "val_encoded.pkl"))

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"], shuffle=False)

    # Model config
# Model config
    print("Vocab size (config):", config["model"]["vocab_size"])
    print("token2id['[PAD]']:", token2id["[PAD]"])
    print("token2id keys:", list(token2id.keys()))
    print("len(vocab):", len(vocab))


    model_cfg = BertConfig(
        vocab_size=len(vocab),   # <--- ini PASTI aman, index 0 .. len(vocab)-1
        hidden_size=config["model"]["embedding_dim"],
        num_attention_heads=config["model"]["num_heads"],
        num_hidden_layers=config["model"]["num_layers"],
        intermediate_size=config["model"]["feedforward_dim"],
        max_position_embeddings=config["model"]["max_len"],
        pad_token_id=token2id["[PAD]"]
    )
    model = BertForMaskedLM(model_cfg)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=token2id["[PAD]"])

    writer = SummaryWriter(log_dir="logs/tensorboard")
    log(f"Training started on {device}...")

    best_val_loss = float("inf")
    early_stopping_counter = 0
    patience = 8 if "patience" not in config["training"] else config["training"]["patience"]
    n_epochs = config["training"]["epochs"]
    pad_token_id = token2id["[PAD]"]

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0
        start_time = time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inp_ids, label_ids = [b.to(device) for b in batch]

            # Dynamic masking: pastikan mask hanya untuk 'e' (sudah di-preprocess)
            attention_mask = (inp_ids != pad_token_id).long()
            outputs = model(input_ids=inp_ids, attention_mask=attention_mask, labels=label_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inp_ids.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_per = evaluate(model, val_loader, device, pad_token_id)

        log(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PER: {val_per:.4f} | Time: {time.time() - start_time:.1f}s"
        )
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("PER/Val", val_per, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                os.path.join(CHECKPOINT_DIR, "bert_g2p_best.pt"),
            )
            log("  >> Model saved.")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            log(f"  >> No improvement ({early_stopping_counter} epochs).")
            if early_stopping_counter >= patience:
                log("Early stopping triggered.")
                break

    writer.close()
    log("Training selesai.")

if __name__ == "__main__":
    main()
