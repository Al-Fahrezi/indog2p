# tests/test_infer.py

import os
import sys
import tempfile

sys.path.append(os.path.abspath(".."))  # Agar indog2p bisa diimport

import pytest
import torch

from indog2p.infer.g2p_infer import (
    IndoG2PPipeline,
    g2p_infer,
    batch_g2p_predict,
    load_infer_pipeline
)
from indog2p.utils.text_cleaner import clean_text
from indog2p.utils.phoneme_utils import apply_phoneme_map

# Simulasi asset minimal (tanpa model besar)
def build_dummy_assets(tmpdir):
    # Vocab minimal
    vocab = ["b", "e", "p", "r", "c", "i", "k", "a", "n", "ʔ", "[PAD]", "[UNK]", "[MASK]"]
    token2id = {c: i for i, c in enumerate(vocab)}
    id2token = {str(i): c for i, c in enumerate(vocab)}
    with open(os.path.join(tmpdir, "vocab.json"), "w") as f:
        import json
        json.dump({"vocab": vocab, "token2id": token2id, "id2token": id2token}, f)
    # Config
    config = {
        "model": {
            "vocab_size": len(vocab),
            "embedding_dim": 8,
            "num_heads": 2,
            "num_layers": 2,
            "feedforward_dim": 8,
            "max_len": 10
        }
    }
    import yaml
    with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    # Phoneme map
    phoneme_map = {
        "grapheme_to_phoneme": {
            "ny": "ɲ",
            "ng": "ŋ",
            "c": "tʃ",
            "'": "ʔ",
            "j": "dʒ",
            "y": "j",
            "q": "k"
        },
        "consonants": list("bcdfghjklmnpqrstvwxyzɲ")
    }
    with open(os.path.join(tmpdir, "phoneme_map.json"), "w") as f:
        json.dump(phoneme_map, f)
    # Lexicon CSV
    with open(os.path.join(tmpdir, "ind_phoneme_dict.csv"), "w") as f:
        f.write("kata,ipa\nbepercikan,bəpərtʃikan\ndeduktif,deduʔtif\n")
    # Dummy model: save state dict kecil
    from indog2p.models.bert_g2p import BertG2P, BertG2PConfig
    config_obj = BertG2PConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=8,
        max_position_embeddings=10,
        pad_token_id=token2id["[PAD]"]
    )
    model = BertG2P(config_obj)
    ckpt = {"model_state_dict": model.state_dict()}
    torch.save(ckpt, os.path.join(tmpdir, "bert_g2p_best.pt"))

def test_single_word_pipeline(tmp_path):
    build_dummy_assets(tmp_path)
    pipe = IndoG2PPipeline(
        config_path=os.path.join(tmp_path, "config.yaml"),
        vocab_path=os.path.join(tmp_path, "vocab.json"),
        phoneme_map_path=os.path.join(tmp_path, "phoneme_map.json"),
        lexicon_path=os.path.join(tmp_path, "ind_phoneme_dict.csv"),
        checkpoint_path=os.path.join(tmp_path, "bert_g2p_best.pt"),
        device="cpu"
    )
    # Test kamus hit
    out = pipe.g2p_word("bepercikan")
    assert out == "bəpərtʃikan"
    # Test fallback (rule + model) untuk kata tidak ada di kamus
    test_word = "deduktif"
    out2 = pipe.g2p_word(test_word)
    assert isinstance(out2, str)
    assert len(out2) > 0
    # Test cleaning dan mapping
    raw = "   Deduktif "
    cleaned = clean_text(raw)
    mapped = apply_phoneme_map(cleaned, pipe.phoneme_map)
    assert isinstance(mapped, str)

def test_g2p_batch_and_sentence(tmp_path):
    build_dummy_assets(tmp_path)
    pipe = IndoG2PPipeline(
        config_path=os.path.join(tmp_path, "config.yaml"),
        vocab_path=os.path.join(tmp_path, "vocab.json"),
        phoneme_map_path=os.path.join(tmp_path, "phoneme_map.json"),
        lexicon_path=os.path.join(tmp_path, "ind_phoneme_dict.csv"),
        checkpoint_path=os.path.join(tmp_path, "bert_g2p_best.pt"),
        device="cpu"
    )
    kalimat = "bepercikan deduktif"
    out = pipe.g2p(kalimat)
    assert isinstance(out, list)
    assert len(out) == 2
    batch = ["bepercikan deduktif", "bepercikan"]
    batch_out = pipe.g2p_batch(batch)
    assert len(batch_out) == 2
    assert all(isinstance(x, list) for x in batch_out)

def test_batch_g2p_predict_cli(tmp_path):
    build_dummy_assets(tmp_path)
    pipe = load_infer_pipeline(
        config_path=os.path.join(tmp_path, "config.yaml"),
        vocab_path=os.path.join(tmp_path, "vocab.json"),
        phoneme_map_path=os.path.join(tmp_path, "phoneme_map.json"),
        lexicon_path=os.path.join(tmp_path, "ind_phoneme_dict.csv"),
        checkpoint_path=os.path.join(tmp_path, "bert_g2p_best.pt"),
        device="cpu"
    )
    outs = batch_g2p_predict(["bepercikan deduktif", "bepercikan"])
    assert len(outs) == 2
    for o in outs:
        assert isinstance(o, list)
        assert all(isinstance(x, str) for x in o)

def test_error_handling_missing_files(tmp_path):
    # Jika asset model tidak ada, harus raise error
    try:
        _ = IndoG2PPipeline(
            config_path=os.path.join(tmp_path, "NO_config.yaml"),
            vocab_path=os.path.join(tmp_path, "NO_vocab.json"),
            phoneme_map_path=os.path.join(tmp_path, "NO_phoneme_map.json"),
            lexicon_path=os.path.join(tmp_path, "NO_dict.csv"),
            checkpoint_path=os.path.join(tmp_path, "NO_ckpt.pt"),
            device="cpu"
        )
    except FileNotFoundError:
        assert True
    else:
        assert False, "Should raise FileNotFoundError for missing files"

if __name__ == "__main__":
    # Manual run tanpa pytest
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = tmpdir
        test_single_word_pipeline(tmp_path)
        test_g2p_batch_and_sentence(tmp_path)
        test_batch_g2p_predict_cli(tmp_path)
        test_error_handling_missing_files(tmp_path)
    print("All inference pipeline tests PASSED.")
