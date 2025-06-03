# indog2p/tests/test_disambiguator.py

import pytest
from indog2p.disambiguator import disambiguate

def test_apel_buah():
    tokens = "Saya makan apel segar".split()
    assert disambiguate("apel", tokens) == "apəl"

def test_apel_upacara():
    tokens = "Kami berkumpul untuk apel pagi".split()
    assert disambiguate("apel", tokens) == "apel"

def test_mental_gangguan():
    tokens = "Dia mengalami gangguan mental".split()
    assert disambiguate("mental", tokens) == "mɛntal"

def test_mental_semangat():
    tokens = "Tim menunjukkan mental juara".split()
    assert disambiguate("mental", tokens) == "mɛntal"

def test_serang_kota():
    tokens = "Saya tinggal di Serang Banten".split()
    assert disambiguate("serang", tokens) == "sɛraŋ"

def test_serang_aksi():
    tokens = "Pasukan menyerang musuh di pagi hari".split()
    # Kata “menyerang” juga termasuk rule untuk “serang”
    assert disambiguate("menyerang", tokens) == "səraŋ"
