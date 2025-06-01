# scripts/evaluate.py
import os
import sys
import json
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM

CONFIG_PATH = "config/config.yaml"
VOCAB_PATH = "config/vocab.json"
PROC_DIR = "data/processed/"
CHECKPOINT_PATH = "checkpoints/bert_g2p_best.pt"
LOG_FILE = "logs/eval.log"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        d = json.load(f)
    return d["token2id"], d["id2token"], d["vocab"]

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

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

def edit_distance(seq1, seq2):
    # Levenshtein distance
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

def evaluate(model, dataloader, device, pad_token_id, id2token, sample_out=10):
    model.eval()
    total_loss, total_chars, total_err, total_acc = 0, 0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    samples = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            inp_ids, label_ids = [b.to(device) for b in batch]
            outputs = model(input_ids=inp_ids)
            logits = outputs.logits  # [B, L, V]
            loss = criterion(
                logits.view(-1, logits.size(-1)), label_ids.view(-1)
            )
            total_loss += loss.item() * inp_ids.size(0)

            preds = logits.argmax(dim=-1)
            for i in range(inp_ids.size(0)):
                pred_list = preds[i].cpu().numpy().tolist()
                label_list = label_ids[i].cpu().numpy().tolist()
                # Remove padding
                pred_seq = [x for x in pred_list if x != pad_token_id]
                label_seq = [x for x in label_list if x != pad_token_id]
                total_chars += len(label_seq)
                total_err += edit_distance(pred_seq, label_seq)
                correct = sum([1 for p, t in zip(pred_seq, label_seq) if p == t])
                total_acc += correct
                # Ambil beberapa sampel output
                if len(samples) < sample_out:
                    pred_str = "".join([id2token[str(idx)] for idx in pred_seq])
                    label_str = "".join([id2token[str(idx)] for idx in label_seq])
                    inp_str = "".join([id2token[str(idx)] for idx in inp_ids[i].cpu().numpy().tolist() if idx != pad_token_id])
                    samples.append((inp_str, pred_str, label_str))
    per = total_err / max(1, total_chars)
    acc = total_acc / max(1, total_chars)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, per, acc, samples

def main():
    # ========== Konfigurasi ==========
    config = load_config()
    token2id, id2token, vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pad_token_id = token2id["[PAD]"]
    # Ubah ke file test lain jika perlu
    eval_file = os.path.join(PROC_DIR, "test_encoded.pkl")
    if not os.path.exists(eval_file):
        log(f"Tidak ditemukan: {eval_file}")
        sys.exit(1)
    eval_set = G2PDataset(eval_file)
    eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False)

    model_cfg = BertConfig(
        vocab_size=len(vocab),      # <--- PASTIKAN AMBIL DARI JUMLAH TOKEN SEBENARNYA!
        hidden_size=config["model"]["embedding_dim"],
        num_attention_heads=config["model"]["num_heads"],
        num_hidden_layers=config["model"]["num_layers"],
        intermediate_size=config["model"]["feedforward_dim"],
        max_position_embeddings=config["model"]["max_len"],
        pad_token_id=pad_token_id
    )
    model = BertForMaskedLM(model_cfg)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    log(f"Model loaded from {CHECKPOINT_PATH}")

    # ========== Evaluasi ==========
    log("==== EVALUATION START ====")
    avg_loss, per, acc, samples = evaluate(
        model, eval_loader, device, pad_token_id, id2token, sample_out=20
    )
    log(f"Eval Loss  : {avg_loss:.4f}")
    log(f"PER (Phoneme Error Rate): {per:.4f}")
    log(f"Char Accuracy: {acc:.4f}")

    log("Sampel prediksi:")
    for inp, pred, label in samples:
        log(f"INP : {inp}")
        log(f"PRED: {pred}")
        log(f"REF : {label}")
        log("-" * 30)
    log("==== EVALUATION END ====")

if __name__ == "__main__":
    main()
