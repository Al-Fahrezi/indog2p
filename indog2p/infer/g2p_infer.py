# indog2p/indog2p/infer/g2p_infer.py

import os
import sys
import json
import yaml
import torch
import pandas as pd

from transformers import BertConfig, BertForMaskedLM

from indog2p.utils.text_cleaner import clean_text
from indog2p.utils.phoneme_utils import load_phoneme_map, apply_phoneme_map
from indog2p.data.utils import encode_masked_input, decode_ids
from indog2p.utils.logger import log

# ==== Kelas/Pipeline G2P End-to-End ====

# Import baru: disambiguator
from indog2p.disambiguator import disambiguate

class IndoG2PPipeline:
    """
    Pipeline inferensi G2P IndoG2P (BERT + rule-based + kamus OOV).
    """
    def __init__(self,
                 config_path="config/config.yaml",
                 vocab_path="config/vocab.json",
                 phoneme_map_path="config/phoneme_map.json",
                 lexicon_path="data/dictionary/ind_phoneme_dict.csv",
                 checkpoint_path="checkpoints/bert_g2p_best.pt",
                 device=None):
        # Load config dan asset
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab tidak ditemukan: {vocab_path}")
        if not os.path.exists(phoneme_map_path):
            raise FileNotFoundError(f"Phoneme map tidak ditemukan: {phoneme_map_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint tidak ditemukan: {checkpoint_path}")

        self.config = self._load_yaml(config_path)
        self.token2id, self.id2token, self.vocab = self._load_vocab(vocab_path)
        self.phoneme_map = load_phoneme_map(phoneme_map_path)
        self.maxlen = self.config["model"]["max_len"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        self.lexicon = self._load_lexicon(lexicon_path)
        self.model.eval()
        log(f"IndoG2PPipeline siap di device: {self.device}")

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_vocab(self, path):
        with open(path, "r") as f:
            d = json.load(f)
        return d["token2id"], d["id2token"], d["vocab"]

    def _load_model(self, checkpoint_path):
        model_cfg = BertConfig(
            vocab_size=self.config["model"]["vocab_size"],
            hidden_size=self.config["model"]["embedding_dim"],
            num_attention_heads=self.config["model"]["num_heads"],
            num_hidden_layers=self.config["model"]["num_layers"],
            intermediate_size=self.config["model"]["feedforward_dim"],
            max_position_embeddings=self.maxlen,
            pad_token_id=self.token2id["[PAD]"],
        )
        model = BertForMaskedLM(model_cfg)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model

    def _load_lexicon(self, path):
        lexicon = {}
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                lexicon[str(row['kata']).strip()] = str(row['ipa']).strip()
        return lexicon

    def _predict_word(self, word):
        # 1. Kamus fonem (lexicon)
        if word in self.lexicon:
            return self.lexicon[word]

        # 2. Rule-based mapping fonem kecuali 'e'
        mapped = apply_phoneme_map(word, self.phoneme_map)

        # 3. Masking huruf 'e' (jadi [MASK])
        inp_ids, inp_seq = encode_masked_input(mapped, self.token2id, self.maxlen), []
        for c in mapped:
            if c == 'e':
                inp_seq.append("[MASK]")
            else:
                inp_seq.append(c)
        inp_tensor = torch.tensor([inp_ids], dtype=torch.long).to(self.device)

        # 4. Model prediksi fonem 'e'
        with torch.no_grad():
            logits = self.model(input_ids=inp_tensor).logits
        pred_ids = inp_ids.copy()
        for i, tok in enumerate(inp_seq):
            if tok == "[MASK]":
                pred_idx = int(torch.argmax(logits[0, i]))
                pred_tok = self.id2token[str(pred_idx)]
                pred_ids[i] = pred_idx

        # 5. Gabungkan hasil fonem, skip PAD/MASK
        out_fonem = ''.join([
            self.id2token[str(idx)]
            for idx in pred_ids
            if self.id2token[str(idx)] not in ["[PAD]", "[MASK]"]
        ])
        return out_fonem

    def g2p(self, text):
        """
        Prediksi fonem (IPA) untuk satu kalimat/teks (bukan batch).
        Return: list fonem per kata (["fonem_kata1", ...])
        """
        text = clean_text(text)
        words = text.strip().split()
        fonem_seq = [self.g2p_word(w) for w in words]
        return fonem_seq

    def g2p_word(self, word):
        """
        Prediksi fonem untuk satu kata (string).
        Pertama, cek rule-based disambiguation; jika tidak ada, fallback ke model.
        """
        word_clean = clean_text(word)

        #  Integrasi disambiguator
        phon = disambiguate(word_clean, [word_clean])
        if phon is not None:
            return phon

        # Kalau tidak ada rules yang cocok, gunakan model
        return self._predict_word(word_clean)

    def g2p_batch(self, texts, return_align=False):
        """
        Batch prediksi G2P untuk list kalimat (atau list kata).
        Return: list of list fonem.
        """
        results = []
        for t in texts:
            out = self.g2p(t)
            if return_align:
                pairs = list(zip(t.strip().split(), out))
                results.append(pairs)
            else:
                results.append(out)
        return results

# === Fungsi pipeline convenience untuk script/CLI ===

_pipeline = None

def load_infer_pipeline(
    config_path="config/config.yaml",
    vocab_path="config/vocab.json",
    phoneme_map_path="config/phoneme_map.json",
    lexicon_path="data/dictionary/ind_phoneme_dict.csv",
    checkpoint_path="checkpoints/bert_g2p_best.pt",
    device=None
):
    global _pipeline
    _pipeline = IndoG2PPipeline(
        config_path, vocab_path, phoneme_map_path, lexicon_path, checkpoint_path, device
    )
    return _pipeline

def g2p_infer(text):
    """
    Pipeline G2P sekali panggil (pastikan sudah load pipeline).
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = load_infer_pipeline()
    return _pipeline.g2p(text)

def batch_g2p_predict(texts, return_align=False):
    """
    Batch pipeline G2P untuk list kalimat/kata.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = load_infer_pipeline()
    return _pipeline.g2p_batch(texts, return_align=return_align)

# ===== Demo/debug =====

if __name__ == "__main__":
    # Demo: prediksi kalimat
    pipeline = load_infer_pipeline()
    kalimat = "bepercikan deduktif tayangan sampingnya 100% ASLI Rp25000"
    fonems = pipeline.g2p(kalimat)
    for w, f in zip(kalimat.strip().split(), fonems):
        print(f"{w}\t{f}")
    print("Prediksi satu kata:", pipeline.g2p_word("bepercikan"))
    print("Prediksi batch:", batch_g2p_predict(["bepercikan deduktif", "tayangan"]))
