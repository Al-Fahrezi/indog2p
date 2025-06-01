# 🇮🇩 IndoG2P — Grapheme-to-Phoneme Bahasa Indonesia (BERT + IPA)

> G2P (Grapheme-to-Phoneme) Bahasa Indonesia modern:  
> **BERT-based, IPA output, integrasi dengan pipeline TTS, voice cloning, dan evaluasi riset.**

---

## 🚀 Fitur Utama
- **Model G2P berbasis BERT** (karakter-level, custom vocab)
- **Konversi ejaan Bahasa Indonesia ke fonem IPA** (sudah support /ə, ŋ, ɲ, tʃ, dʒ, ʔ, dsb)
- **Pipeline training, evaluasi, prediksi, ONNX export**
- **Modular:** Bisa dikembangkan ke REST API, batch prediction, TTS, dsb.
- **Data-driven:** Dukung training dari data sendiri
- **Testing lengkap:** Tersedia unit test (`tests/`), EDA notebook, coverage metrik (PER, char acc)

---

## 📂 Struktur Project
indog2p/
│
├── models/ # Model BERT G2P (bert_g2p.py, init.py)
├── data/ # Loader, encoding, utils data (utils.py, loader.py, init.py)
├── utils/ # Utility project: phoneme map, metrik, logger, cleaner
├── infer/ # Pipeline inferensi (g2p_infer.py, init.py)
│
scripts/
│ ├── preprocess.py # Preprocessing data mentah
│ ├── train.py # Training G2P model
│ ├── evaluate.py # Evaluasi model (PER, char acc)
│ ├── predict.py # CLI prediksi G2P
│ ├── finetune.py # Fine-tune model dari checkpoint
│ └── export_onnx.py # Export model ke ONNX
│
config/
│ ├── config.yaml # Konfigurasi model & pipeline
│ └── phoneme_map.json # Mapping grafem ke fonem
│ └── vocab.json # Vocab karakter model
│
data/
│ ├── raw/ # Dataset mentah (csv/txt)
│ ├── processed/ # Hasil preprocessing (pkl)
│ └── dictionary/ # Kamus fonem (csv)
│
checkpoints/ # Model hasil training
logs/ # File log training, tensorboard
notebooks/
│ └── EDA_and_Experiment.ipynb
tests/ # Unit test
requirements.txt
setup.py
README.md
---

## ⚡️ Quickstart

### 1. **Persiapan Environment**
```bash
git clone https://github.com/username/indog2p.git
cd indog2p
pip install -r requirements.txt
