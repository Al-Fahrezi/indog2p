# scripts/predict.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # <<< Penting agar Python menemukan utils/

import argparse
import yaml
import json
import re
import torch
import pandas as pd
from transformers import BertConfig, BertForMaskedLM
from num2words import num2words

# ====================================================
# Tambahkan import TextProcessor dan Stanza
# ====================================================
from utils.text_processor import TextProcessor
import stanza
import logging

# ====================================================
# Matikan logging berlebihan dari Stanza
# ====================================================
logging.getLogger("stanza").setLevel(logging.WARNING)

# ====================================================
# Inisialisasi TextProcessor
# ====================================================
tp = TextProcessor()

# ====================================================
# Inisialisasi pipeline Stanza untuk Bahasa Indonesia
# ====================================================
nlp = stanza.Pipeline(
    lang="id",
    processors="tokenize,mwt,pos",
    tokenize_no_ssplit=True,
    use_gpu=False
)

# ====================================================
# Konstanta path default; sesuaikan jika diperlukan
# ====================================================
CONFIG_PATH       = "/content/git/config/config.yaml"
VOCAB_PATH        = "/content/git/config/vocab.json"
PHONEME_MAP_PATH  = "/content/git/config/phoneme_map.json"
DICT_PATH         = "/content/git/data/dictionary/ind_phoneme_dict.csv"
CHECKPOINT_PATH   = "/content/git/checkpoints/bert_g2p_best.pt"

# ====================================================
# Fungsi‐fungsi bantu
# ====================================================

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_phoneme_map():
    with open(PHONEME_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["token2id"], d["id2token"], d["vocab"]

def clean_text(text: str) -> str:
    # Kalau masih ingin membersihkan karakter tersisa:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def apply_phoneme_map(word: str, phoneme_map: dict) -> str:
    w = word
    # Aturan glottal stop: jika akhir 'k' ganti 'ʔ'
    if w.endswith("k"):
        w = w[:-1] + "ʔ"
    for c in phoneme_map.get("consonants", {}):
        w = re.sub(f"k{c}", f"ʔ{c}", w)
    # G2P basic (skip 'e'—akan di-mask kemudian)
    for seq, rep in sorted(phoneme_map.get("grapheme_to_phoneme", {}).items(), key=lambda x: -len(x[0])):
        if seq != "e":
            w = w.replace(seq, rep)
    return w

def tokenize(text: str) -> list:
    return text.strip().split()

def encode_masked_input(word: str, token2id: dict, maxlen: int):
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

def load_lexicon() -> dict:
    lexicon = {}
    if not os.path.exists(DICT_PATH):
        return lexicon
    df = pd.read_csv(DICT_PATH)
    for _, row in df.iterrows():
        ipa = str(row["ipa"]).strip().rstrip(",")
        lexicon[str(row["kata"]).strip()] = ipa
    return lexicon

# ====================================================
# Fungsi utama prediksi per kata (tanpa disambiguator)
# ====================================================
def predict_word(word: str,
                 pos_tag: str,
                 lexicon: dict,
                 model: BertForMaskedLM,
                 token2id: dict,
                 id2token: dict,
                 maxlen: int,
                 phoneme_map: dict,
                 device: torch.device) -> str:

    # =======================================================
    # 1. Cek lexicon (kamus kata→IPA)
    # =======================================================
    if word in lexicon:
        return lexicon[word]

    # =======================================================
    # 2. Jika kata tertentu memerlukan override berdasarkan POS
    #    Contoh: "apel" sebagai homograf
    # =======================================================
    if word.lower() == "apel":
        if pos_tag == "NOUN":
            return "apəl"   # fonem untuk “apel” (buah)
        elif pos_tag == "VERB":
            return "apel"   # fonem untuk “apel” (upacara)

    # =======================================================
    # 3. Aturan fonem dasar (skip 'e')
    # =======================================================
    mapped = apply_phoneme_map(word, phoneme_map)

    # =======================================================
    # 4. Masking huruf 'e' → gunakan BERT
    # =======================================================
    inp_ids, inp_seq = encode_masked_input(mapped, token2id, maxlen)
    inp_tensor = torch.tensor([inp_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        attention_mask = (inp_tensor != token2id["[PAD]"]).long()
        logits = model(input_ids=inp_tensor, attention_mask=attention_mask).logits

    pred_ids = inp_ids.copy()
    for i, tok in enumerate(inp_seq):
        if tok == "[MASK]":
            pred_idx = int(torch.argmax(logits[0, i]))
            pred_ids[i] = pred_idx

    # =======================================================
    # 5. Gabungkan output → string fonem
    # =======================================================
    out_phon = "".join([
        id2token[str(idx)]
        for idx in pred_ids
        if id2token[str(idx)] not in ["[PAD]", "[MASK]"]
    ])
    return out_phon

# ====================================================
# Fungsi prediksi satu baris teks
# ====================================================
def predict_line(line: str,
                 lexicon: dict,
                 model: BertForMaskedLM,
                 token2id: dict,
                 id2token: dict,
                 maxlen: int,
                 phoneme_map: dict,
                 device: torch.device) -> list:
    # ----------------------------------------------------
    # 1. Langsung normalisasi penuh dengan TextProcessor
    # ----------------------------------------------------
    normalized = tp.normalize(line)

    # ----------------------------------------------------
    # 2. (Opsional) Bersihkan sisa karakter non‐alfanumerik
    # ----------------------------------------------------
    normalized = clean_text(normalized)

    # ----------------------------------------------------
    # 3. Jalankan Stanza untuk mendapatkan token & POS
    # ----------------------------------------------------
    doc = nlp(normalized)
    stanza_words = []
    print("\n[DEBUG] POS tagging result:")
    for sent in doc.sentences:
        for w in sent.words:
            print(f" - {w.text:15} → {w.upos}")  # ← tampilkan kata dan tag-nya
            stanza_words.append((w.text, w.upos))

    # ----------------------------------------------------
    # 4. Tokenisasi sederhana (split spasi)
    # ----------------------------------------------------
    words = tokenize(normalized)

    result = []
    for idx, w in enumerate(words):
        pos_tag = None
        # Cocokkan tokenisasi Stanza dan split spasi:
        if idx < len(stanza_words) and stanza_words[idx][0] == w:
            pos_tag = stanza_words[idx][1]
        else:
            # Jika mismatch, cari di stanza_words
            for tw, tp_pos in stanza_words:
                if tw == w:
                    pos_tag = tp_pos
                    break

        # Panggil prediksi fonem per kata
        fonem = predict_word(
            w,
            pos_tag,
            lexicon,
            model,
            token2id,
            id2token,
            maxlen,
            phoneme_map,
            device
        )
        result.append(fonem)

    return result

# ====================================================
# Fungsi utama: proses --text atau --file
# ====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str,
                        help="Teks (satu kalimat) yang akan diprediksi fonemnya")
    parser.add_argument("--file", type=str,
                        help="Path ke file teks input, satu kalimat per baris")
    parser.add_argument("--output", type=str,
                        help="Output file (opsional); hanya menyimpan string fonem saja")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Harap berikan input dengan --text atau --file")
        sys.exit(1)

    # -------------------------
    # 1. Load semua asset
    # -------------------------
    config      = load_config()
    phoneme_map = load_phoneme_map()
    token2id, id2token, vocab = load_vocab()
    lexicon     = load_lexicon()
    maxlen      = config["model"]["max_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Membangun dan memuat model BERT‐for‐MaskedLM
    model_cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=config["model"]["embedding_dim"],
        num_attention_heads=config["model"]["num_heads"],
        num_hidden_layers=config["model"]["num_layers"],
        intermediate_size=config["model"]["feedforward_dim"],
        max_position_embeddings=maxlen,
        pad_token_id=token2id["[PAD]"],
    )
    model = BertForMaskedLM(model_cfg)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint model tidak ditemukan: {CHECKPOINT_PATH}")
        sys.exit(1)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    out_lines = []

    # -------------------------
    # 2. Jika ada --text
    # -------------------------
    if args.text:
        kal       = args.text.strip()
        fonem_seq = predict_line(kal, lexicon, model,
                                 token2id, id2token, maxlen,
                                 phoneme_map, device)
        # Cetak hanya rangkaian fonem (dipisah spasi)
        out_str = " ".join(fonem_seq)
        print(out_str)
        out_lines.append(out_str)

    # -------------------------
    # 3. Jika ada --file
    # -------------------------
    if args.file:
        if not os.path.exists(args.file):
            print(f"File input tidak ditemukan: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for raw in lines:
            kal = raw.strip()
            if not kal:
                continue
            fonem_seq = predict_line(kal, lexicon, model,
                                     token2id, id2token, maxlen,
                                     phoneme_map, device)
            out_str = " ".join(fonem_seq)
            print(out_str)
            out_lines.append(out_str)

    # -------------------------
    # 4. Jika ada --output, simpan hasil ke file
    # -------------------------
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for out in out_lines:
                f.write(out + "\n")
        print(f"Hasil fonem disimpan di: {args.output}")

if __name__ == "__main__":
    main()
