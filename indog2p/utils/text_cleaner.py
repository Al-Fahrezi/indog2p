# indog2p/utils/text_cleaner.py

"""
Text cleaning dan normalisasi input untuk pipeline IndoG2P:
- Lowercase, hilangkan simbol asing, normalisasi spasi
- Ekspansi angka ke kata (pakai num2words Bahasa Indonesia)
- Penanganan ekspresi khusus (mata uang, persen, dsb)
- Bisa di-extend untuk domain TTS
"""

import re
try:
    from num2words import num2words
except ImportError:
    def num2words(x, lang="id"):
        return str(x)  # fallback (tidak ekspansi)

def normalize_number(text):
    """
    Ubah angka ke kata dengan num2words (Bahasa Indonesia).
    """
    def replacer(match):
        n = int(match.group(0))
        try:
            return num2words(n, lang="id")
        except Exception:
            return str(n)
    # Ganti semua angka (tanpa titik/koma ribuan)
    return re.sub(r'\d+', replacer, text)

def expand_special_symbols(text):
    """
    Ekspansi simbol mata uang, persen, dst ke bentuk teks.
    """
    # Rp... → "rupiah ..."
    text = re.sub(r"rp\s?(\d+)", lambda m: f"rupiah {num2words(int(m.group(1)), lang='id')}", text, flags=re.IGNORECASE)
    # % → "persen"
    text = re.sub(r"(\d+)\s?%", lambda m: f"{num2words(int(m.group(1)), lang='id')} persen", text)
    # $... → "dolar ..."
    text = re.sub(r"\$\s?(\d+)", lambda m: f"dolar {num2words(int(m.group(1)), lang='id')}", text)
    # Angka dengan . (juta, miliar) → bisa dihandle tambahan
    return text

def clean_text(text):
    """
    Cleaning dan normalisasi lengkap untuk input IndoG2P.
    """
    # Lowercase
    text = str(text).strip().lower()
    # Hilangkan simbol asing (kecuali huruf/angka/spasi)
    text = re.sub(r"[^a-z0-9\s%$.,\-\/]", " ", text)
    # Normalisasi simbol ke bentuk kata
    text = expand_special_symbols(text)
    # Ubah angka ke kata
    text = normalize_number(text)
    # Hilangkan tanda baca sisa, kecuali spasi dan strip
    text = re.sub(r"[.,\/\-]", " ", text)
    # Hilangkan spasi lebih
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def full_clean_text(text):
    """
    Untuk cleaning lebih ketat (misal dataset TTS):  
    - Lowercase  
    - Ejaan baku  
    - Semua simbol dihapus  
    """
    text = clean_text(text)
    # Hapus semua kecuali huruf dan spasi
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

if __name__ == "__main__":
    # Demo
    data = [
        "Rp25000 itu harga kopi.",
        "100% ASLI!",
        "Gaji saya $1000/bulan.",
        "Dia menulis 2 buku & 1 jurnal.",
        "Tanggal 17/08/1945 adalah hari kemerdekaan."
    ]
    for t in data:
        print("ORIG:", t)
        print("CLEAN:", clean_text(t))
        print()
