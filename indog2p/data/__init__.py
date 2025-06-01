# indog2p/data/__init__.py

"""
indog2p.data
============

Submodul untuk semua fungsi pemrosesan data di IndoG2P:
- Loader dataset (pickle/csv)
- Tokenizer dan vocab tools
- Preprocessing pipeline
- Dataset class siap pakai PyTorch
"""

from .loader import G2PDataset, create_dataloader
from .utils import (
    build_vocab_from_pairs,
    encode_masked_input,
    encode_label,
    encode_seq,
    load_pickle,
    save_pickle
)

__all__ = [
    "G2PDataset",
    "create_dataloader",
    "build_vocab_from_pairs",
    "encode_masked_input",
    "encode_label",
    "encode_seq",
    "load_pickle",
    "save_pickle",
]
