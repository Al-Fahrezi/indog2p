# scripts/predict.py
import os
import sys
import argparse
import yaml
import json
import pickle
import re
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertConfig, BertForMaskedLM
from num2words import num2words

CONFIG_PATH = "config/config.yaml"
VOCAB_PATH = "config/vocab.json"
PHONEME_MAP_PATH = "config/phoneme_map.json"
DICT_PATH = "data/dictionary/ind_phoneme_dict.csv"
CHECKPOINT_PATH = "checkpoints/bert_g2p_best.pt"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_phoneme_map():
    with open(PHONEME_MAP_PATH, "r") as f:
        return json.load(f)

def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        d = json.load(f)
    return d["token2id"], d["id2token"], d["vocab"]

def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', ' ', text)     # Hilangkan tanda baca, kecuali huruf dan spasi
    text = re.sub(r'\s+', ' ', text)
    text = normalize_number(text)
    return text.strip()

def normalize_number(text):
    # Angka jadi kata
    def repl(m):
        n = int(m.group(0))
        return num2words(n, lang='id')
    return re.sub(r'\d+', repl, text)

def apply_phoneme_map(word, phoneme_map):
    w = word
    # Glottal stop rules
    if w.endswith("k"):
        w = w[:-1] + "ʔ"
    for c in phoneme_map["consonants"]:
        w = re.sub(f"k{c}", f"ʔ{c}", w)
    # grapheme to phoneme (skip 'e')
    for seq, rep in sorted(phoneme_map["grapheme_to_phoneme"].items(), key=lambda x: -len(x[0])):
        if seq != 'e':
            w = w.replace(seq, rep)
    return w

def tokenize(text):
    return text.strip().split()

def encode_masked_input(word, token2id, maxlen):
    seq = []
    for c in word:
        if c == "e":
            seq.append("[MASK]")
        else:
            seq.append(c)
    ids = [token2id.get(c, token2id["[UNK]"]) for c in seq]
    if len(ids) < maxlen:
        ids += [token2id["[PAD]"]] * (maxlen - len(ids))
    else:
        ids = ids[:maxlen]
    return ids, seq

def load_lexicon():
    lexicon = {}
    if not os.path.exists(DICT_PATH):
        return lexicon
    df = pd.read_csv(DICT_PATH)
    for _, row in df.iterrows():
        lexicon[str(row['kata']).strip()] = str(row['ipa']).strip()
    return lexicon

def predict_word(word, lexicon, model, token2id, id2token, maxlen, phoneme_map, device):
    # 1. Kamus: jika ada di kamus, return fonem kamus
    if word in lexicon:
        return lexicon[word]
    # 2. Aturan fonem (tanpa mapping 'e'), masking 'e'
    mapped = apply_phoneme_map(word, phoneme_map)
    inp_ids, inp_seq = encode_masked_input(mapped, token2id, maxlen)
    inp_tensor = torch.tensor([inp_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids=inp_tensor).logits
    pred_ids = inp_ids.copy()
    for i, tok in enumerate(inp_seq):
        if tok == "[MASK]":
            pred_idx = int(torch.argmax(logits[0, i]))
            pred_tok = id2token[str(pred_idx)]
            pred_ids[i] = pred_idx
    # Hasil akhir: semua token, kecuali pad, digabungkan jadi string fonem
    out_fonem = ''.join([id2token[str(idx)] for idx in pred_ids if id2token[str(idx)] not in ["[PAD]", "[MASK]"]])
    return out_fonem

def predict_line(line, lexicon, model, token2id, id2token, maxlen, phoneme_map, device):
    # Clean, tokenize
    words = tokenize(clean_text(line))
    result = []
    for word in words:
        fonem = predict_word(word, lexicon, model, token2id, id2token, maxlen, phoneme_map, device)
        result.append(fonem)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Teks yang akan diubah ke fonem")
    parser.add_argument("--file", type=str, help="Path file berisi satu kalimat per baris")
    parser.add_argument("--output", type=str, help="Output file (opsional)")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Berikan input dengan --text atau --file")
        sys.exit(1)

    config = load_config()
    phoneme_map = load_phoneme_map()
    token2id, id2token, vocab = load_vocab()
    lexicon = load_lexicon()
    maxlen = config["model"]["max_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = BertConfig(
        vocab_size=len(vocab),                      # <-- INI BENAR!
        hidden_size=config["model"]["embedding_dim"],
        num_attention_heads=config["model"]["num_heads"],
        num_hidden_layers=config["model"]["num_layers"],
        intermediate_size=config["model"]["feedforward_dim"],
        max_position_embeddings=maxlen,
        pad_token_id=token2id["[PAD]"],
    )

    model = BertForMaskedLM(model_cfg)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.text:
        hasil = predict_line(args.text, lexicon, model, token2id, id2token, maxlen, phoneme_map, device)
        print(f"Teks      : {args.text}")
        print("Fonem IPA : " + " | ".join(hasil))
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Tidak ditemukan: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        out_lines = []
        for line in lines:
            line = line.strip()
            if not line: continue
            fonem_seq = predict_line(line, lexicon, model, token2id, id2token, maxlen, phoneme_map, device)
            out_str = f"{line}\t{' | '.join(fonem_seq)}"
            out_lines.append(out_str)
            print(out_str)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for out in out_lines:
                    f.write(out + "\n")
            print(f"Hasil fonem disimpan di: {args.output}")

if __name__ == "__main__":
    main()
