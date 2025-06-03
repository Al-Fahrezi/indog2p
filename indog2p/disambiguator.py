# indog2p/indog2p/disambiguator.py

from typing import List, Optional

DISAMBIG_RULES = {
    "apel": {
        "buah": "apəl",
        "makan": "apəl",
        "pohon": "apəl",
        "upacara": "apel",
        "bendera": "apel",
        "pagi": "apel",
    },
    "mental": {
        "gangguan": "mɛntal",
        "psikolog": "mɛntal",
        "kesehatan": "mɛntal",
        "semangat": "mɛntal",
        "jatuh": "mɛntal",
        "drop": "mɛntal",
    },
    "serang": {
        "kota": "sɛraŋ",
        "provinsi": "sɛraŋ",
        "banten": "sɛraŋ",
        "diserang": "səraŋ",
        "menyerang": "səraŋ",
        "serangan": "səraŋ",
    },
}

def disambiguate(word: str, sentence_tokens: List[str]) -> Optional[str]:
    """
    Mengembalikan fonem yang tepat untuk kata homograf (apel, mental, serang)
    berdasarkan konteks (list token dalam sebuah kalimat). 
    Jika tidak ditemukan aturan yang cocok, kembalikan None.
    """
    lw = word.lower()
    if lw in DISAMBIG_RULES:
        rules = DISAMBIG_RULES[lw]
        for tok in sentence_tokens:
            key = tok.lower()
            if key in rules:
                return rules[key]
    return None
