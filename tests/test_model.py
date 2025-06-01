# tests/test_model.py

import os
import sys
import tempfile
import torch

sys.path.append(os.path.abspath(".."))  # Akses root project

import pytest

from indog2p.models.bert_g2p import BertG2P, BertG2PConfig, build_bert_g2p_from_yaml_config
from indog2p.data.utils import build_vocab_from_pairs, encode_masked_input, encode_label
from indog2p.utils.logger import print_model_summary

def test_build_model_from_config():
    # Simulasi config YAML dan vocab
    config_dict = {
        "model": {
            "vocab_size": 32,
            "embedding_dim": 16,
            "num_heads": 2,
            "num_layers": 2,
            "feedforward_dim": 32,
            "max_len": 12
        }
    }
    pairs = [("deduktif", "deduʔtif"), ("bepercikan", "bəpərtʃikan")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    pad_token_id = token2id["[PAD]"]
    model = BertG2P.from_project_config(config_dict, pad_token_id=pad_token_id)
    print_model_summary(model)
    assert isinstance(model, BertG2P)
    assert model.config.vocab_size == len(vocab)
    # Test forward
    inp = torch.tensor([encode_masked_input("deduktif", token2id, 12)], dtype=torch.long)
    label = torch.tensor([encode_label("deduʔtif", token2id, 12)], dtype=torch.long)
    out = model(input_ids=inp, labels=label)
    assert "loss" in out and "logits" in out
    assert out["logits"].shape == (1, 12, len(vocab))
    assert out["loss"].item() >= 0

def test_model_inference_and_masking():
    # Simulasi vocab dan model
    pairs = [("bepercikan", "bəpərtʃikan")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    config = BertG2PConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        max_position_embeddings=12,
        pad_token_id=token2id["[PAD]"],
    )
    model = BertG2P(config)
    # Test input dengan huruf 'e' (harus masking)
    inp = torch.tensor([encode_masked_input("bepercikan", token2id, 12, mask_char='e')], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=inp)
    assert "logits" in out
    logits = out["logits"]
    assert logits.shape == (1, 12, len(vocab))

def test_freeze_backbone_layers():
    # Test freeze backbone untuk fine-tuning head
    pairs = [("nyanyi", "ɲaɲi")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    config = BertG2PConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        max_position_embeddings=8,
        pad_token_id=token2id["[PAD]"],
    )
    model = BertG2P(config)
    # Freeze backbone, only head (mlm_head) trainable
    for name, param in model.named_parameters():
        if not name.startswith("mlm_head"):
            param.requires_grad = False
    n_frozen = sum(not p.requires_grad for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    assert n_frozen > 0 and n_frozen < n_total

def test_edgecase_short_and_long_input():
    # Test input < maxlen dan > maxlen
    pairs = [("aku", "aku"), ("berkebun", "bərkəbun")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    config = BertG2PConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=16,
        max_position_embeddings=6,
        pad_token_id=token2id["[PAD]"],
    )
    model = BertG2P(config)
    inp_short = torch.tensor([encode_masked_input("aku", token2id, 6)], dtype=torch.long)
    inp_long = torch.tensor([encode_masked_input("berkebun", token2id, 6)], dtype=torch.long)
    # Model tidak error walau input < atau > maxlen (akan dipad/truncate)
    with torch.no_grad():
        out_short = model(input_ids=inp_short)
        out_long = model(input_ids=inp_long)
    assert out_short["logits"].shape == (1, 6, len(vocab))
    assert out_long["logits"].shape == (1, 6, len(vocab))

def test_model_save_and_load(tmp_path):
    # Simulasi save dan load model state dict
    pairs = [("eksplorasi", "ekspLorasi")]
    vocab, token2id, id2token = build_vocab_from_pairs(pairs)
    config = BertG2PConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=16,
        max_position_embeddings=10,
        pad_token_id=token2id["[PAD]"],
    )
    model = BertG2P(config)
    # Save
    save_path = tmp_path / "bert_g2p_test.pt"
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    # Load baru dan compare param
    model2 = BertG2P(config)
    state = torch.load(save_path, map_location="cpu")["model_state_dict"]
    model2.load_state_dict(state)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

if __name__ == "__main__":
    # Jalankan semua test manual jika tidak pakai pytest
    test_build_model_from_config()
    test_model_inference_and_masking()
    test_freeze_backbone_layers()
    test_edgecase_short_and_long_input()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_model_save_and_load(tmpdir)
    print("All model tests PASSED.")
