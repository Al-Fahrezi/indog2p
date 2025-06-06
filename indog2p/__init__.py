"""
indog2p: Grapheme-to-Phoneme Bahasa Indonesia berbasis BERT.

API Publik:
- BertG2P: Model BERT karakter-level
- G2PDataset: Dataset utility
- apply_phoneme_map: Aturan mapping fonem
- clean_text: Normalisasi & cleaning
- phoneme_error_rate: Evaluasi fonem
"""

# from .infer.g2p_infer import G2P  # ← ini tetap di-comment/hapus

from .models.bert_g2p import BertG2P
from .data.loader import G2PDataset
from .utils.phoneme_utils import apply_phoneme_map
from .utils.text_cleaner import clean_text
from .utils.metrics import phoneme_error_rate

__all__ = [
    "BertG2P",
    "G2PDataset",
    "apply_phoneme_map",
    "clean_text",
    "phoneme_error_rate"
]
