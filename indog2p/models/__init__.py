# indog2p/models/__init__.py

"""
indog2p.models
==============

Modul untuk semua model G2P yang digunakan dalam proyek IndoG2P.

- BertG2P : model utama (berbasis BERT karakter-level)
- [Tambahkan model lain di sini jika perlu]
"""

from transformers import BertForMaskedLM

# Alias BertG2P ke BertForMaskedLM, supaya kompatibel untuk import dari script lain
BertG2P = BertForMaskedLM

# Contoh placeholder untuk model lain (bisa diisi nanti)
# from .lstm_g2p import LSTMG2P

__all__ = [
    "BertG2P",
    # "LSTMG2P",
]

