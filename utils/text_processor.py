# utils/text_processor.py

import re
import os
import pkg_resources
from num2words import num2words

class TextProcessor:
    """
    TextProcessor for Bahasa Indonesia:
    - Menormalisasi angka, mata uang, satuan, dan timezone.
    - Menghapus URL, tanda baca tidak perlu, dsb.
    """

    def __init__(self):
        # Path ke file TSV (diasumsikan ditempatkan di folder utils/)
        base = os.path.dirname(os.path.abspath(__file__))
        self.measurements_path = os.path.join(base, "measurements.tsv")
        self.currency_path     = os.path.join(base, "currency.tsv")
        self.timezones_path    = os.path.join(base, "timezones.tsv")

        # Muat data TSV jika diperlukan
        self.measurements = self._load_tsv(self.measurements_path)
        self.currency    = self._load_tsv(self.currency_path)
        self.timezones   = self._load_tsv(self.timezones_path)

    def _load_tsv(self, path):
        """
        Baca file TSV dan kembalikan dict:
        misal: { 'kg': 'kilogram', 'm': 'meter', ... }
        """
        d = {}
        if not os.path.exists(path):
            return d
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    key, val = parts[0].strip(), parts[1].strip()
                    d[key.lower()] = val.lower()
        return d

    def normalize(self, text: str) -> str:
        """
        Lakukan normalisasi menyeluruh:
        1. Ubah mata uang: 'Rp20000' → 'dua puluh ribu rupiah'
        2. Ubah satuan berat: '3,5kg' → 'tiga koma lima kilogram'
        3. Tangani decimalsemikolon (desimal dengan koma)
        4. Ubah tanggal/waktu/timezone (jika ada di TSV)
        5. Ubah angka lain: '123' → 'seratus dua puluh tiga'
        6. Hapus URL, tanda baca tidak perlu, dan huruf kapital
        """
        txt = text.strip().lower()

        # 1. Hapus URL
        txt = re.sub(r"https?://\S+|www\.\S+", "", txt)

        # 2. Normalisasi mata uang (contoh: rp20000, rp 20000, rp2.000,00)
        #    Tangani format dengan titik dan koma:
        #    - hapus titik ribuan, ganti koma desimal dengan '.' untuk num2words
        def repl_rupiah(m):
            amt_text = m.group(1)
            # Hapus titik ribuan: '20.000' → '20000'
            amt_clean = amt_text.replace(".", "")
            # Ganti koma desimal ke titik: '2,5' → '2.5'
            amt_clean = amt_clean.replace(",", ".")
            try:
                # num2words tidak langsung menerima float penuh jadi pisah integer/desimal
                if "." in amt_clean:
                    integer_part, decimal_part = amt_clean.split(".")
                    integer_part = int(integer_part)
                    decimal_part = decimal_part.lstrip("0")
                    words_int = num2words(integer_part, lang="id")
                    words_dec = " ".join(num2words(int(d), lang="id") for d in decimal_part)
                    return f"{words_int} koma {words_dec} rupiah"
                else:
                    integer_part = int(amt_clean)
                    words_int = num2words(integer_part, lang="id")
                    return f"{words_int} rupiah"
            except:
                return m.group(0)  # fallback

        txt = re.sub(
            r"\brp\s?([\d\.\,]+)\b",
            repl_rupiah,
            txt
        )

        # 3. Normalisasi satuan (contoh: 3,5kg, 10kg, 2,75 m)
        #    Tangani desimal dan satuan yang melekat atau dengan spasi
        def repl_measure(m):
            num_text = m.group(1).replace(".", "").replace(",", ".")
            unit     = m.group(2).lower()
            # Ubah ke bentuk kata
            try:
                if "." in num_text:
                    integer_part, decimal_part = num_text.split(".")
                    integer_part = int(integer_part)
                    decimal_part = decimal_part.lstrip("0")
                    words_int = num2words(integer_part, lang="id")
                    words_dec = " ".join(num2words(int(d), lang="id") for d in decimal_part)
                    full_num = f"{words_int} koma {words_dec}"
                else:
                    full_num = num2words(int(num_text), lang="id")
            except:
                full_num = m.group(1)

            # Cari nama satuan di TSV; jika tidak ada, pakai teks unit
            unit_name = self.measurements.get(unit, unit)
            return f"{full_num} {unit_name}"

        txt = re.sub(
            r"\b([\d\.\,]+)\s*([a-zA-Z]{1,4})\b",
            repl_measure,
            txt
        )

        # 4. Normalisasi timezone (contoh: +07:00, UTC+7, WIB)
        def repl_tz(m):
            tz = m.group(0).lower()
            return self.timezones.get(tz, tz)
        txt = re.sub(
            r"\b(?:[a-z]{2,3}[+\-]\d{1,2}:\d{2}|utc[+\-]?\d{1,2}|[a-z]{2,3})\b",
            repl_tz,
            txt
        )

        # 5. Normalisasi angka biasa (contoh: '123' → 'seratus dua puluh tiga')
        def repl_num(m):
            try:
                n = int(m.group(0))
                return num2words(n, lang="id")
            except:
                return m.group(0)

        txt = re.sub(r"\b\d+\b", repl_num, txt)

        # 6. Bersihkan sisa tanda baca dan spasi ganda
        txt = re.sub(r"[^\w\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()

        return txt


# Jika ingin langsung mencoba:
if __name__ == "__main__":
    tp = TextProcessor()

    contoh = [
        "Saya membeli 3,5kg beras Rp20000 di pasar.",
        "Angka 12345 harus jadi kata.",
        "Jam 17:30 WITA siap berangkat.",
        "Harga: Rp 1.000.000,50 saja!",
    ]
    for c in contoh:
        print(c, "→", tp.normalize(c))
