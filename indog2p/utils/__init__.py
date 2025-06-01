# indog2p/utils/__init__.py

"""
indog2p.utils
=============

Submodul utilitas utama IndoG2P:
- Fungsi manipulasi fonem, mapping, dan normalisasi teks
- Perhitungan metrik (PER, akurasi karakter, dsb)
- Logging project, print summary
- Cleaning dan normalisasi teks (angka, simbol, dsb)

Dapat diimport langsung dari package utils:
    from indog2p.utils import (
        phoneme_utils, metrics, logger, text_cleaner,
        apply_phoneme_map, clean_text, normalize_number,
        per, char_accuracy, print_model_summary
    )
"""

from .phoneme_utils import (
    apply_phoneme_map,
    load_phoneme_map,
    PHONEME_SYMBOLS
)

from .metrics import (
    per,
    char_accuracy,
    edit_distance
)

from .logger import (
    setup_logger,
    log,
    print_model_summary
)

from .text_cleaner import (
    clean_text,
    normalize_number
)

__all__ = [
    # phoneme utils
    "apply_phoneme_map",
    "load_phoneme_map",
    "PHONEME_SYMBOLS",
    # metrics
    "per",
    "char_accuracy",
    "edit_distance",
    # logger
    "setup_logger",
    "log",
    "print_model_summary",
    # text cleaning
    "clean_text",
    "normalize_number"
]
