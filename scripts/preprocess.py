import os
import sys
import yaml
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from num2words import num2words
from tqdm import tqdm
import re

CONFIG_PATH = "config/config.yaml"
PHONEME_MAP_PATH = "config/phoneme_map.json"
DICT_PATH = "data/dictionary/id_word2phoneme.csv"
RAW_DATA_PATH = "data/dictionary/id_word2phoneme.csv"  # ganti sesuai nama aslinya
RAW_DIR = "data/raw/"
PROC_DIR = "data/processed/"
VOCAB_PATH = "config/vocab.json"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs("data/dictionary", exist_ok=True)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_phoneme_map():
    with open(PHONEME_MAP_PATH, "r") as f:
        return json.load(f)

def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', ' ', text)     # Hilangkan tanda baca kecuali huruf dan spasi
    text = re.sub(r'\s+', ' ', text)         # Hilangkan spasi berlebih
    text = normalize_number(text)
    return text.strip()

def normalize_number(text):
    def repl(m):
        n = int(m.group(0))
        return num2words(n, lang='id')
    return re.sub(r'\d+', repl, text)

def apply_phoneme_map(word, phoneme_map):
    w = word
    if w.endswith("k"):
        w = w[:-1] + "ʔ"
    for c in phoneme_map["consonants"]:
        w = re.sub(f"k{c}", f"ʔ{c}", w)
    for seq, rep in sorted(phoneme_map["grapheme_to_phoneme"].items(), key=lambda x: -len(x[0])):
        if seq != 'e':
            w = w.replace(seq, rep)
    return w

def build_vocab(data, extra_tokens=["[PAD]", "[UNK]", "[MASK]"]):
    charset = set()
    for inp, label in data:
        charset.update(list(inp))
        charset.update(list(label))
    vocab = sorted(list(charset))
    for token in extra_tokens:
        if token not in vocab:
            vocab.append(token)
    token2id = {c: i for i, c in enumerate(vocab)}
    id2token = {i: c for i, c in enumerate(vocab)}
    return vocab, token2id, id2token

def encode_seq(seq, token2id, maxlen):
    ids = [token2id.get(c, token2id["[UNK]"]) for c in seq]
    if len(ids) < maxlen:
        ids += [token2id["[PAD]"]] * (maxlen - len(ids))
    else:
        ids = ids[:maxlen]
    return ids

def encode_masked_input(word, token2id, maxlen):
    seq = []
    for c in word:
        if c == "e":
            seq.append("[MASK]")
        else:
            seq.append(c)
    return encode_seq(seq, token2id, maxlen)

def encode_label(label, token2id, maxlen):
    return encode_seq(label, token2id, maxlen)

def stratified_split(df, stratify_col, test_size=0.1, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state,
                            stratify=df[stratify_col])

def main():
    print("==== IndoG2P Preprocessing ====")
    config = load_config()
    phoneme_map = load_phoneme_map()
    maxlen = config["model"]["max_len"]

    # Load data CSV
    if os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH)
    elif os.path.exists(os.path.join(RAW_DIR, "id_word2phoneme.csv")):
        df = pd.read_csv(os.path.join(RAW_DIR, "id_word2phoneme.csv"))
    else:
        print("Dataset tidak ditemukan.")
        sys.exit(1)
    print(f"Loaded: {len(df)} data.")

    # Normalisasi
    df = df.dropna()
    df['kata'] = df['kata'].apply(clean_text)
    df['ipa'] = df['ipa'].apply(lambda s: re.sub(r'\s+', '', str(s)))
    df = df.drop_duplicates(subset=["kata"]).reset_index(drop=True)
    df['kata'] = df['kata'].str.lower()
    df['ipa'] = df['ipa'].str.lower()
    print("Cleaning done.")

    # Simpan dictionary IPA (untuk OOV lookup saat inferensi)
    df[["kata", "ipa"]].to_csv(DICT_PATH, index=False)

    # Tambahkan kolom jumlah huruf 'e' (untuk stratifikasi split)
    df["e_count"] = df['kata'].apply(lambda x: x.count('e'))

    # Split data (train/val/test) stratified by e_count
    train_df, temp_df = stratified_split(df, "e_count", test_size=0.2)
    val_df, test_df = stratified_split(temp_df, "e_count", test_size=0.5)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Simpan csv hasil split
    train_df.to_csv(os.path.join(RAW_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(RAW_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(RAW_DIR, "test.csv"), index=False)

    # Preprocess & build data pickle untuk model (train, val, test)
    datasets = {"train": train_df, "val": val_df, "test": test_df}
    all_pairs = []
    for name, data in datasets.items():
        pairs = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Prep {name}"):
            kata = row['kata']
            ipa = row['ipa']
            kata_mapped = apply_phoneme_map(kata, phoneme_map)
            pairs.append((kata_mapped, ipa))
            # Hanya train/val dimasukkan ke vocab
            if name != "test":
                all_pairs.append((kata_mapped, ipa))
        with open(os.path.join(PROC_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(pairs, f)

    # Build vocab dari seluruh data train+val
    vocab, token2id, id2token = build_vocab(all_pairs)
    print("Vocab size:", len(vocab))
    with open(VOCAB_PATH, "w") as f:
        json.dump({"vocab": vocab, "token2id": token2id, "id2token": id2token}, f, indent=2)

    # Encoding untuk model (train, val, test)
    for name in ["train", "val", "test"]:
        pkl_file = os.path.join(PROC_DIR, f"{name}.pkl")
        if not os.path.exists(pkl_file):
            continue
        with open(pkl_file, "rb") as f:
            pairs = pickle.load(f)
        encoded = []
        for inp, label in tqdm(pairs, desc=f"Encoding {name}"):
            inp_ids = encode_masked_input(inp, token2id, maxlen)
            label_ids = encode_label(label, token2id, maxlen)
            encoded.append((inp_ids, label_ids))
        with open(os.path.join(PROC_DIR, f"{name}_encoded.pkl"), "wb") as f:
            pickle.dump(encoded, f)

    print("Preprocessing Selesai.")
    print(f"File dihasilkan di {PROC_DIR}, vocab di {VOCAB_PATH}.")

if __name__ == "__main__":
    main()
