# tests/test_data.py

import os
import sys
import tempfile
import pickle
import json

sys.path.append(os.path.abspath(".."))  # agar modul indog2p bisa diimport dari root project

import pytest

from indog2p.data.loader import G2PDataset, create_dataloader
from indog2p.data.utils import (
    build_vocab_from_pairs,
    encode_seq,
    encode_masked_input,
    encode_label,
    decode_ids,
    save_pickle,
    load_pickle
)

def test_build_vocab_and_encode_decode():
    pairs = [("samping", "sampiŋ"), ("tayangan", "tajaŋan"), ("bepercikan", "bəpərtʃikan")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    assert "[PAD]" in vocab and "[MASK]" in vocab
    assert token2id["[PAD]"] != token2id["[UNK]"]
    # Test encode
    maxlen = 10
    seq = "samping"
    ids = encode_seq(seq, token2id, maxlen)
    assert len(ids) == maxlen
    # Test decode
    seq_out = decode_ids(ids, id2token)
    assert isinstance(seq_out, str)
    # OOV handling
    seq_oov = "xyz"
    ids_oov = encode_seq(seq_oov, token2id, maxlen)
    unk_id = token2id["[UNK]"]
    assert all([x == unk_id or x in token2id.values() for x in ids_oov[:3]])

def test_encode_masked_input():
    pairs = [("deduktif", "deduʔtif")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    word = "deduktif"
    maxlen = 12
    ids = encode_masked_input(word, token2id, maxlen, mask_char='e')
    # Huruf 'e' jadi [MASK]
    mask_idx = token2id["[MASK]"]
    assert ids[0] == mask_idx

def test_loader_and_pickle(tmp_path):
    # Simulasi dataset (input_ids, label_ids)
    data = [
        ([1,2,3,4,0,0], [5,6,7,0,0,0]),
        ([1,1,1,1,1,1], [2,2,2,2,2,2])
    ]
    pkl_path = tmp_path / "test.pkl"
    save_pickle(data, pkl_path)
    loaded = load_pickle(pkl_path)
    assert len(loaded) == 2
    # Test loader
    dataset = G2PDataset(pkl_path)
    assert len(dataset) == 2
    inp, label = dataset[0]
    assert inp.shape[0] == 6 and label.shape[0] == 6

def test_dataloader_batch(tmp_path):
    # Simulasi dataset
    data = [([1,2,3], [4,5,6]), ([7,8,9], [1,2,3]), ([1,1,1], [2,2,2])]
    pkl_path = tmp_path / "mini.pkl"
    save_pickle(data, pkl_path)
    dataset = G2PDataset(pkl_path)
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2  # 2 batch (2 + 1)
    b0_inp, b0_label = batches[0]
    assert b0_inp.shape == (2, 3)
    assert b0_label.shape == (2, 3)

def test_encode_label_and_padding():
    pairs = [("nganga", "ŋaŋa")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    label = "ŋaŋa"
    maxlen = 7
    ids = encode_label(label, token2id, maxlen)
    pad_id = token2id["[PAD]"]
    assert len(ids) == maxlen
    assert ids[-1] == pad_id

if __name__ == "__main__":
    # Jalankan manual tanpa pytest
    test_build_vocab_and_encode_decode()
    test_encode_masked_input()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_loader_and_pickle(tmpdir)
        test_dataloader_batch(tmpdir)
    test_encode_label_and_padding()
    print("All data tests PASSED.")
