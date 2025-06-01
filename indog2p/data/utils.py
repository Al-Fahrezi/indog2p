# indog2p/data/utils.py

import os
import json
import pickle

def build_vocab_from_pairs(pairs, extra_tokens=["[PAD]", "[UNK]", "[MASK]"]):
    """
    Buat vocab karakter dari semua pasangan (input, label).
    Tambah token khusus: [PAD], [UNK], [MASK].
    Return: vocab list, token2id dict, id2token dict.
    """
    charset = set()
    for inp, label in pairs:
        charset.update(list(inp))
        charset.update(list(label))
    vocab = sorted(list(charset))
    for token in extra_tokens:
        if token not in vocab:
            vocab.append(token)
    token2id = {c: i for i, c in enumerate(vocab)}
    id2token = {str(i): c for i, c in enumerate(vocab)}
    return vocab, token2id, id2token

def save_vocab_json(vocab, token2id, id2token, path):
    with open(path, "w") as f:
        json.dump({"vocab": vocab, "token2id": token2id, "id2token": id2token}, f, indent=2)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def encode_seq(seq, token2id, maxlen):
    """
    Encode list/str seq ke id, pad/truncate ke maxlen.
    """
    ids = [token2id.get(c, token2id["[UNK]"]) for c in seq]
    if len(ids) < maxlen:
        ids += [token2id["[PAD]"]] * (maxlen - len(ids))
    else:
        ids = ids[:maxlen]
    return ids

def encode_masked_input(word, token2id, maxlen, mask_char='e'):
    """
    Huruf mask_char diganti [MASK], lain di-encode biasa.
    """
    seq = []
    for c in word:
        if c == mask_char:
            seq.append("[MASK]")
        else:
            seq.append(c)
    return encode_seq(seq, token2id, maxlen)

def encode_label(label, token2id, maxlen):
    """
    Encode label (IPA string) ke id, pad/truncate ke maxlen.
    """
    return encode_seq(label, token2id, maxlen)

def decode_ids(ids, id2token, stop_token="[PAD]"):
    """
    Balik id ke string karakter, stop di stop_token kalau ada.
    """
    out = []
    for idx in ids:
        tok = id2token.get(str(idx), "?")
        if tok == stop_token:
            break
        out.append(tok)
    return "".join(out)

def pad_seq(seq, maxlen, pad_val):
    """
    Pad/truncate list seq ke panjang maxlen.
    """
    if len(seq) < maxlen:
        seq = seq + [pad_val] * (maxlen - len(seq))
    else:
        seq = seq[:maxlen]
    return seq

def batch_encode_words(words, token2id, maxlen, mask_char='e'):
    """
    Encode banyak kata (list of string), return list of input_ids (masking mask_char)
    """
    all_ids = []
    for word in words:
        ids = encode_masked_input(word, token2id, maxlen, mask_char=mask_char)
        all_ids.append(ids)
    return all_ids

def batch_encode_labels(labels, token2id, maxlen):
    """
    Encode banyak label IPA ke id
    """
    all_ids = []
    for label in labels:
        ids = encode_label(label, token2id, maxlen)
        all_ids.append(ids)
    return all_ids

def onehot_encode(ids, vocab_size):
    """
    One-hot encoding untuk batch of id sequence (tidak selalu dipakai, hanya untuk analisa)
    """
    import numpy as np
    batch = []
    for seq in ids:
        arr = np.zeros((len(seq), vocab_size), dtype='float32')
        for i, idx in enumerate(seq):
            arr[i, idx] = 1.0
        batch.append(arr)
    return batch

def flatten_batch(batch):
    """
    Flatten nested batch (list of tuple) jadi 2 list.
    """
    inps, labels = [], []
    for inp, label in batch:
        inps.append(inp)
        labels.append(label)
    return inps, labels

# Jika ingin ditambah fungsi tokenisasi kata/teks yang lebih advanced, tambahkan di sini.

