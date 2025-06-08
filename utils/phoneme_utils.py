# indog2p/utils/phoneme_utils.py

import json
import os
import re

PHONEME_SYMBOLS = [
    # Vokal Bahasa Indonesia
    "a", "i", "u", "e", "o", "ə",
    # Fonem konsonan standar dan extended IPA
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n",
    "p", "q", "r", "s", "t", "v", "w", "x", "y", "z",
    "ɲ", "ŋ", "ʃ", "tʃ", "dʒ", "ʔ"
]

def load_phoneme_map(path="config/phoneme_map.json"):
    """
    Load mapping grafem ke fonem dari JSON project.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Phoneme map tidak ditemukan: {path}")
    with open(path, "r") as f:
        return json.load(f)

def apply_phoneme_map(word, phoneme_map=None):
    """
    Terapkan aturan konversi grafem-ke-fonem Bahasa Indonesia.
    - Diaplikasikan ke kata sebelum masking 'e' (untuk BERT).
    - phoneme_map: dict hasil load_phoneme_map().
    Return: string hasil mapping.
    """
    if phoneme_map is None:
        phoneme_map = load_phoneme_map()

    w = word
    # Aturan glottal stop (contoh: kata berakhir 'k' diubah jadi ʔ)
    if w.endswith("k"):
        w = w[:-1] + "ʔ"
    # Aturan: 'k' di tengah sebelum konsonan jadi ʔ juga
    for c in phoneme_map.get("consonants", []):
        w = re.sub(f"k{c}", f"ʔ{c}", w)
    # Urutan mapping: dari yang paling panjang
    mapping = phoneme_map.get("grapheme_to_phoneme", {})
    for seq, rep in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if seq != 'e':  # 'e' dimasking oleh model, BERT akan prediksi
            w = w.replace(seq, rep)
    return w

def is_valid_phoneme_seq(seq, phoneme_set=None):
    """
    Cek apakah sequence string terdiri hanya dari fonem yang valid.
    """
    if phoneme_set is None:
        phoneme_set = set(PHONEME_SYMBOLS)
    for c in seq:
        if c not in phoneme_set:
            return False
    return True

def count_phoneme_coverage(dataset, phoneme_set=None):
    """
    Hitung distribusi fonem pada dataset (list of IPA label).
    """
    if phoneme_set is None:
        phoneme_set = set(PHONEME_SYMBOLS)
    from collections import Counter
    counter = Counter()
    for seq in dataset:
        for c in seq:
            if c in phoneme_set:
                counter[c] += 1
    return dict(counter)

def split_word_phonemes(word, phoneme_map=None):
    """
    Pisahkan kata jadi list fonem (bukan sekadar split karakter, 
    tapi urutkan fonem multi-char seperti 'tʃ', 'dʒ', dsb).
    """
    if phoneme_map is None:
        phoneme_map = load_phoneme_map()
    mapping = phoneme_map.get("grapheme_to_phoneme", {})
    multi_phonemes = sorted(mapping.values(), key=lambda x: -len(x))
    res = []
    i = 0
    while i < len(word):
        found = False
        for ph in multi_phonemes:
            if word[i:i+len(ph)] == ph:
                res.append(ph)
                i += len(ph)
                found = True
                break
        if not found:
            res.append(word[i])
            i += 1
    return res

def ipa_to_text(seq, reverse_map=None):
    """
    (Opsional) Untuk kebutuhan debugging: fonem → grafem/teks kasar.
    """
    if reverse_map is None:
        return seq
    for ipa, g in sorted(reverse_map.items(), key=lambda x: -len(x[0])):
        seq = seq.replace(ipa, g)
    return seq

if __name__ == "__main__":
    # Demo/debug manual
    phoneme_map = load_phoneme_map()
    print("Mapping test:")
    print(apply_phoneme_map("bepercikan", phoneme_map))
    print(apply_phoneme_map("deduktif", phoneme_map))
    print(split_word_phonemes("bəpərtʃikan", phoneme_map))
