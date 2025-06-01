# indog2p/utils/metrics.py

"""
Metrik evaluasi untuk pipeline IndoG2P:
- Phoneme Error Rate (PER)
- Character-level accuracy
- Edit distance (Levenshtein)
- (Opsi) Word Error Rate
"""

import numpy as np

def edit_distance(seq1, seq2):
    """
    Hitung Levenshtein distance antara dua sequence (bisa string/list id).
    Return: int, jumlah edit
    """
    len1, len2 = len(seq1), len(seq2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # delete
                    dp[i][j - 1],    # insert
                    dp[i - 1][j - 1] # replace
                )
    return dp[len1][len2]

def per(predictions, references, pad_token_id=None):
    """
    Phoneme Error Rate (PER): Levenshtein / jumlah target
    Input:
        predictions : list of list (id atau string fonem)
        references  : list of list (id atau string fonem)
        pad_token_id: int/str, untuk masking (optional)
    Output: float (0...1)
    """
    total_dist = 0
    total_len = 0
    for pred, ref in zip(predictions, references):
        # Hapus padding jika ada
        if pad_token_id is not None:
            pred = [x for x in pred if x != pad_token_id]
            ref = [x for x in ref if x != pad_token_id]
        total_dist += edit_distance(pred, ref)
        total_len += len(ref)
    if total_len == 0:
        return 0.0
    return total_dist / total_len

def char_accuracy(predictions, references, pad_token_id=None):
    """
    Akurasi karakter level untuk fonem (mirip token-level acc).
    Input:
        predictions, references: list of list (id atau char)
    Output: float (0...1)
    """
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        # Masking pad token
        if pad_token_id is not None:
            ref_nopad = [x for x in ref if x != pad_token_id]
            pred_nopad = [x for i, x in enumerate(pred[:len(ref_nopad)])]
        else:
            ref_nopad = ref
            pred_nopad = pred[:len(ref_nopad)]
        for p, r in zip(pred_nopad, ref_nopad):
            if p == r:
                correct += 1
        total += len(ref_nopad)
    if total == 0:
        return 0.0
    return correct / total

def wer(predictions, references):
    """
    Word Error Rate (WER) — umumnya untuk evaluasi kalimat, jarang dipakai G2P per kata.
    Input: list of list of token (biasanya word-level, bisa untuk analisa kalimat)
    Output: float (0...1)
    """
    total_dist = 0
    total_len = 0
    for pred, ref in zip(predictions, references):
        total_dist += edit_distance(pred, ref)
        total_len += len(ref)
    if total_len == 0:
        return 0.0
    return total_dist / total_len

if __name__ == "__main__":
    # Unit test/debug
    pred = ["sampiŋɲa", "tajaŋan", "bəpərtʃikan"]
    ref  = ["sampiŋɲa", "tajaŋan", "bəpərtʃikan"]
    print("PER perfect:", per(pred, ref))
    print("Char acc perfect:", char_accuracy(pred, ref))

    # Error test
    pred2 = ["sampiŋa", "tajanan", "bəpərtʃika"]
    print("PER error:", per(pred2, ref))
    print("Char acc error:", char_accuracy(pred2, ref))
    print("Edit distance:", edit_distance(pred2[0], ref[0]))
