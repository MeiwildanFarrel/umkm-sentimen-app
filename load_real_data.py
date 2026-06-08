# -*- coding: utf-8 -*-
"""
=============================================================================
  SIMANTAP UMKM - Dinas Koperasi dan UKM Kabupaten Banyumas
  ---------------------------------------------------------------------------
  load_real_data.py : Memuat data ulasan NYATA hasil pengumpulan manual
                      dari Google Maps ke dalam sistem SIMANTAP UMKM.

  Dijalankan SETELAH train.py, sebagai PENGGANTI generate_umkm.py.
  Membaca  : data_ulasan_umkm.csv  (dikumpulkan manual dari Google Maps)
  Menghasilkan:
    - umkm_data.json          (dipakai oleh app.py)
    - umkm_reviews_detail.csv (transparansi data ulasan per baris)
    - umkm_summary.xlsx       (ringkasan per UMKM, 2 sheet)

  Pipeline : Baca CSV -> Preprocessing -> SVM -> Hitung Skor -> Simpan JSON
=============================================================================
"""

import json
import re
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ─────────────────────────────────────────────────────────────────────────────
# 1. KAMUS KEYWORD ASPEK & ATURAN REKOMENDASI
#    (identik dengan generate_umkm.py agar konsisten)
# ─────────────────────────────────────────────────────────────────────────────
ASPEK_KEYWORDS = {
    "Kualitas": [
        "kualitas", "bagus", "jelek", "rusak", "cacat", "sesuai", "tidak sesuai",
        "bahan", "material", "awet", "tahan", "rapuh", "original", "asli", "palsu",
        "mantap", "memuaskan", "kecewa", "mengecewakan", "sempurna", "buruk",
        "rasa", "tekstur", "warna", "bentuk", "ukuran", "fungsi", "berfungsi",
        "mati", "error", "lumayan", "oke", "produk", "barang",
    ],
    "Packaging": [
        "packing", "packaging", "kemasan", "bungkus", "kotak", "dos", "kardus",
        "bubble", "wrap", "aman", "selamat", "pecah", "retak", "penyok",
        "lecet", "rapi", "berantakan", "terbuka", "bocor",
    ],
    "Harga": [
        "harga", "mahal", "murah", "worth", "terjangkau", "overpriced",
        "ekonomis", "hemat", "bersaing", "sesuai harga", "nilai", "bayar",
        "ongkir", "gratis ongkir", "promo", "diskon", "biaya",
    ],
    "Pengiriman": [
        "pengiriman", "kirim", "ekspedisi", "kurir", "datang", "tiba",
        "sampai", "cepat", "lambat", "lama", "tepat waktu", "telat",
        "delay", "tracking", "resi", "j&t", "jne", "sicepat", "shopee express",
    ],
}

REKOMENDASI_RULES = {
    ("Packaging",   "negatif"): ["Ikutkan dalam pelatihan packaging Diskop",
                                  "Konsultasi desain kemasan dengan mentor"],
    ("Kualitas",    "negatif"): ["Kunjungan lapangan untuk evaluasi proses produksi",
                                  "Pendampingan quality control"],
    ("Harga",       "negatif"): ["Analisis struktur biaya dengan konsultan Diskop",
                                  "Evaluasi strategi penetapan harga"],
    ("Pengiriman",  "negatif"): ["Rekomendasi mitra ekspedisi yang lebih handal",
                                  "Pelatihan manajemen pengiriman"],
}

REKOMENDASI_BAIK = [
    "Pertahankan kualitas produk dan layanan saat ini",
    "Berpotensi dijadikan UMKM percontohan binaan Diskop",
]


def deteksi_aspek(teks_asli: str) -> dict:
    teks = str(teks_asli).lower()
    return {aspek: any(kw in teks for kw in keywords)
            for aspek, keywords in ASPEK_KEYWORDS.items()}


def hitung_skor(reviews_hasil: list) -> int:
    """Skor 0-100: 60% rasio positif + 40% rata-rata confidence positif."""
    if not reviews_hasil:
        return 50
    positif = [r for r in reviews_hasil if r["sentimen"] == "Positive"]
    rasio_positif = len(positif) / len(reviews_hasil)
    rata_confidence = float(np.mean([r["confidence_positif"] for r in reviews_hasil]))
    return int(round((rasio_positif * 0.6 + rata_confidence * 0.4) * 100))


def tentukan_status(skor: int) -> str:
    if skor >= 75:   return "Baik"
    elif skor >= 55: return "Pantau"
    elif skor >= 40: return "Perlu Perhatian"
    else:            return "Kritis"


def top_kata(processed_list: list, n: int = 5) -> list:
    counter = Counter()
    for teks in processed_list:
        for kata in teks.split():
            if len(kata) > 2:
                counter[kata] += 1
    return [kata for kata, _ in counter.most_common(n)]


def potong(teks: str, n: int = 280) -> str:
    teks = str(teks).strip().replace("\n", " ")
    return teks if len(teks) <= n else teks[:n].rsplit(" ", 1)[0] + "..."


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL & VECTORIZER
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 66)
print("  SIMANTAP UMKM - LOAD DATA REAL DARI GOOGLE MAPS")
print("=" * 66)
print()
print("[1/4] Memuat model SVM, vectorizer & stopword ...")

try:
    vectorizer  = joblib.load("vectorizer.pkl")
    svm_model   = joblib.load("svm_model.pkl")
    STOPWORDS_ID = joblib.load("stopwords_set.pkl")
except FileNotFoundError as e:
    print("  [ERROR] File tidak ditemukan: " + str(e.filename))
    print("  Jalankan terlebih dahulu: python train.py")
    raise SystemExit(1)

stemmer  = StemmerFactory().create_stemmer()
classes  = list(svm_model.classes_)
POS_IDX  = classes.index("Positive")
print("      [OK] Model & vectorizer dimuat.")


def preprocess(text: str) -> str:
    """Pipeline preprocessing — IDENTIK dengan train.py."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text.replace("_", " ")
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BACA & BERSIHKAN CSV
# ─────────────────────────────────────────────────────────────────────────────
CSV_INPUT = "data_ulasan_umkm_clean.csv"
print()
print("[2/4] Membaca " + CSV_INPUT + " ...")

try:
    df_raw = pd.read_csv(CSV_INPUT, encoding="cp1252", sep=";", skiprows=1, header=0)
except FileNotFoundError:
    print("  [ERROR] " + CSV_INPUT + " tidak ditemukan di folder ini.")
    raise SystemExit(1)

# Ambil hanya 7 kolom utama
kolom = ["nama_umkm", "kategori", "lokasi", "ulasan", "rating", "label_sentimen", "produk_utama"]
df_raw = df_raw[kolom]

# Buang baris contoh/header duplikat dan baris kosong
df_raw = df_raw[
    df_raw["ulasan"].notna() &
    (df_raw["ulasan"].str.strip() != "") &
    (df_raw["ulasan"] != "Copy-paste teks ulasan lengkap dari GMaps")
]
df_raw = df_raw[
    df_raw["nama_umkm"].notna() &
    (df_raw["nama_umkm"].str.strip() != "") &
    (df_raw["nama_umkm"] != "Nama usaha sesuai Google Maps")
].reset_index(drop=True)

# Normalisasi rating: "4,5" / "4.5" → dibulatkan ke int
def parse_rating(r):
    try:
        return int(round(float(str(r).replace(",", "."))))
    except:
        return None

df_raw["rating"] = df_raw["rating"].apply(parse_rating)

# Isi produk_utama yang kosong
df_raw["produk_utama"] = df_raw["produk_utama"].fillna("-")

total_baris = len(df_raw)
total_umkm  = df_raw["nama_umkm"].nunique()
print(f"      [OK] {total_baris} ulasan dari {total_umkm} UMKM dimuat.")
print("      Distribusi label:")
for label, cnt in df_raw["label_sentimen"].value_counts().items():
    print(f"        {label}: {cnt}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROSES SETIAP UMKM
# ─────────────────────────────────────────────────────────────────────────────
print()
print("[3/4] Menganalisis ulasan per UMKM ...")

hasil_umkm = []
semua_reviews_detail = []
aspek_keluhan_global = {a: 0 for a in ASPEK_KEYWORDS}

for idx, (nama_umkm, grup) in enumerate(df_raw.groupby("nama_umkm", sort=False), start=1):
    umkm_id    = f"R{idx:03d}"
    kategori   = grup["kategori"].iloc[0]
    lokasi     = grup["lokasi"].iloc[0]
    produk     = grup["produk_utama"].iloc[0]

    # Buang baris dengan label kosong dan duplikat ulasan
    grup = grup[grup["label_sentimen"].notna() & (grup["label_sentimen"].str.strip() != "")]
    grup = grup.drop_duplicates(subset=["ulasan"])

    ulasan_list  = grup["ulasan"].astype(str).tolist()
    label_list   = grup["label_sentimen"].astype(str).tolist()

    # a. Preprocessing
    pasangan = [(raw, preprocess(raw), lbl) for raw, lbl in zip(ulasan_list, label_list)]
    pasangan = [(raw, pro, lbl) for raw, pro, lbl in pasangan if pro.strip()]
    if not pasangan:
        print(f"      {umkm_id} {nama_umkm} — SKIP (semua ulasan kosong setelah preprocessing)")
        continue

    raws  = [p[0] for p in pasangan]
    procs = [p[1] for p in pasangan]
    lbls  = [p[2] for p in pasangan]

    # b. SVM untuk confidence score (bukan untuk label)
    vecs   = vectorizer.transform(procs)
    probas = svm_model.predict_proba(vecs)

    def label_to_sentimen(lbl: str) -> str:
        """Konversi label manual CSV → format internal sistem (Positif→Positive, lainnya→Negative)."""
        return "Positive" if str(lbl).strip().lower() == "positif" else "Negative"

    reviews_hasil = []
    for raw, pro, pr, lbl in zip(raws, procs, probas, lbls):
        p_pos = float(pr[POS_IDX])
        reviews_hasil.append({
            "review":            raw,
            "processed":         pro,
            "sentimen":          label_to_sentimen(lbl),   # ← dari label manual
            "confidence_positif": p_pos,                   # ← dari SVM
            "aspek":             deteksi_aspek(raw),
        })

    # c. Kumpulkan detail ulasan untuk CSV ekspor
    for nomor, r in enumerate(reviews_hasil, start=1):
        semua_reviews_detail.append({
            "id_umkm":            umkm_id,
            "nama_umkm":          nama_umkm,
            "kategori_umkm":      kategori,
            "lokasi":             lokasi,
            "produk_utama":       produk,
            "sumber_data":        "Google Maps (manual)",
            "no_ulasan":          nomor,
            "ulasan_asli":        r["review"],
            "ulasan_preprocessed": r["processed"],
            "prediksi_sentimen":  r["sentimen"],
            "confidence_positif": round(r["confidence_positif"], 4),
            "confidence_negatif": round(1 - r["confidence_positif"], 4),
            "aspek_kualitas":     r["aspek"]["Kualitas"],
            "aspek_packaging":    r["aspek"]["Packaging"],
            "aspek_harga":        r["aspek"]["Harga"],
            "aspek_pengiriman":   r["aspek"]["Pengiriman"],
        })

    total        = len(reviews_hasil)
    review_pos   = sum(1 for r in reviews_hasil if r["sentimen"] == "Positive")
    review_neg   = total - review_pos

    # d. Skor & status
    skor   = hitung_skor(reviews_hasil)
    status = tentukan_status(skor)

    # e. Analisis per-aspek
    aspek_detail = {}
    for nama_aspek in ASPEK_KEYWORDS:
        disebut = [r for r in reviews_hasil if r["aspek"][nama_aspek]]
        if disebut:
            pos      = sum(1 for r in disebut if r["sentimen"] == "Positive")
            skor_pos = int(round(pos / len(disebut) * 100))
            terdeteksi = True
        else:
            skor_pos   = 50
            terdeteksi = False
        keluhan = sum(1 for r in disebut if r["sentimen"] == "Negative")
        aspek_keluhan_global[nama_aspek] += keluhan
        aspek_detail[nama_aspek] = {
            "skor_positif":  skor_pos,
            "terdeteksi":    terdeteksi,
            "keluhan_count": keluhan,
            "positif_count": pos if disebut else 0,
            "total_mention": len(disebut),
        }

    # f. Kata kunci dominan
    teks_pos = [r["processed"] for r in reviews_hasil if r["sentimen"] == "Positive"]
    teks_neg = [r["processed"] for r in reviews_hasil if r["sentimen"] == "Negative"]
    kata_pos = top_kata(teks_pos, 5)
    kata_neg = top_kata(teks_neg, 5)

    # g. Masalah utama & rekomendasi
    terdeteksi_list = [a for a, d in aspek_detail.items() if d["terdeteksi"]]
    total_keluhan   = sum(d["keluhan_count"] for d in aspek_detail.values())

    if total_keluhan == 0 or not terdeteksi_list:
        masalah_utama = "Tidak ada"
        rekomendasi   = list(REKOMENDASI_BAIK)
    else:
        urut          = sorted(terdeteksi_list, key=lambda a: aspek_detail[a]["skor_positif"])
        masalah_utama = urut[0]
        rekomendasi   = list(REKOMENDASI_RULES.get((masalah_utama, "negatif"), []))
        if len(urut) > 1 and aspek_detail[urut[1]]["skor_positif"] < 55:
            extra = REKOMENDASI_RULES.get((urut[1], "negatif"), [])
            if extra:
                rekomendasi.append(extra[0])
        rekomendasi = rekomendasi[:3]

    # h. Contoh ulasan terbaik/terburuk
    pos_sorted  = sorted([r for r in reviews_hasil if r["sentimen"] == "Positive"],
                         key=lambda r: r["confidence_positif"], reverse=True)
    neg_sorted  = sorted([r for r in reviews_hasil if r["sentimen"] == "Negative"],
                         key=lambda r: r["confidence_positif"])
    contoh_pos  = potong(pos_sorted[0]["review"]) if pos_sorted else ""
    contoh_neg  = potong(neg_sorted[0]["review"]) if neg_sorted else ""

    hasil_umkm.append({
        "id":                    umkm_id,
        "nama":                  nama_umkm,
        "pemilik":               "-",          # tidak dicatat dari GMaps
        "kategori":              kategori,
        "lokasi":                lokasi,
        "produk_utama":          produk,
        "skor":                  skor,
        "status":                status,
        "total_review":          total,
        "review_positif":        review_pos,
        "review_negatif":        review_neg,
        "aspek":                 aspek_detail,
        "kata_kunci_positif":    kata_pos,
        "kata_kunci_negatif":    kata_neg,
        "masalah_utama":         masalah_utama,
        "rekomendasi":           rekomendasi,
        "contoh_review_positif": contoh_pos,
        "contoh_review_negatif": contoh_neg,
    })
    print(f"      {umkm_id} {nama_umkm:<40} ulasan={total:<3} skor={skor:<3} {status}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SUSUN RINGKASAN GLOBAL & SIMPAN JSON
# ─────────────────────────────────────────────────────────────────────────────
print()
print("[4/4] Menyimpan output ...")

ringkasan = {
    "kritis":           sum(1 for u in hasil_umkm if u["status"] == "Kritis"),
    "perlu_perhatian":  sum(1 for u in hasil_umkm if u["status"] == "Perlu Perhatian"),
    "pantau":           sum(1 for u in hasil_umkm if u["status"] == "Pantau"),
    "baik":             sum(1 for u in hasil_umkm if u["status"] == "Baik"),
}

output = {
    "generated_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "sumber_data":           "Google Maps (pengumpulan manual)",
    "total_umkm":            len(hasil_umkm),
    "ringkasan":             ringkasan,
    "aspek_keluhan_global":  dict(
        sorted(aspek_keluhan_global.items(), key=lambda x: x[1], reverse=True)
    ),
    "umkm":                  hasil_umkm,
}

with open("umkm_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("      [OK] umkm_data.json tersimpan (" + str(len(hasil_umkm)) + " UMKM).")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EKSPOR CSV DETAIL
# ─────────────────────────────────────────────────────────────────────────────
CSV_OUT  = "umkm_reviews_detail.csv"
XLSX_OUT = "umkm_summary.xlsx"

df_detail = pd.DataFrame(semua_reviews_detail)
df_detail.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print("      [OK] " + CSV_OUT + " tersimpan (" + str(len(df_detail)) + " baris).")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EKSPOR EXCEL RINGKASAN
# ─────────────────────────────────────────────────────────────────────────────
baris_summary = []
for u in hasil_umkm:
    total_u   = u["total_review"]
    persen_pos = round(u["review_positif"] / total_u * 100, 1) if total_u else 0
    rek = u["rekomendasi"]
    baris_summary.append({
        "ID UMKM":               u["id"],
        "Nama UMKM":             u["nama"],
        "Kategori":              u["kategori"],
        "Lokasi":                u["lokasi"],
        "Produk Utama":          u["produk_utama"],
        "Total Ulasan":          total_u,
        "Ulasan Positif":        u["review_positif"],
        "Ulasan Negatif":        u["review_negatif"],
        "% Positif":             persen_pos,
        "Skor Kesehatan":        u["skor"],
        "Status":                u["status"],
        "Kualitas (% pos)":      u["aspek"]["Kualitas"]["skor_positif"],
        "Packaging (% pos)":     u["aspek"]["Packaging"]["skor_positif"],
        "Harga (% pos)":         u["aspek"]["Harga"]["skor_positif"],
        "Pengiriman (% pos)":    u["aspek"]["Pengiriman"]["skor_positif"],
        "Masalah Utama":         u["masalah_utama"],
        "Rekomendasi 1":         rek[0] if len(rek) > 0 else "",
        "Rekomendasi 2":         rek[1] if len(rek) > 1 else "",
        "Rekomendasi 3":         rek[2] if len(rek) > 2 else "",
        "Contoh Ulasan Positif": u["contoh_review_positif"],
        "Contoh Ulasan Negatif": u["contoh_review_negatif"],
    })

df_summary = pd.DataFrame(baris_summary)

try:
    with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Ringkasan UMKM", index=False)
        ws1 = writer.sheets["Ringkasan UMKM"]
        for col in ws1.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col)
            ws1.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

        df_detail.to_excel(writer, sheet_name="Detail Ulasan", index=False)
        ws2 = writer.sheets["Detail Ulasan"]
        for col in ws2.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col)
            ws2.column_dimensions[col[0].column_letter].width = min(max_len + 4, 80)

    print("      [OK] " + XLSX_OUT + " tersimpan (2 sheet).")
except ImportError:
    print("      [SKIP] openpyxl tidak terinstal. Install: pip install openpyxl")

print()
print("=" * 66)
print("  RINGKASAN STATUS UMKM")
print("-" * 66)
print(f"  Kritis           : {ringkasan['kritis']}")
print(f"  Perlu Perhatian  : {ringkasan['perlu_perhatian']}")
print(f"  Pantau           : {ringkasan['pantau']}")
print(f"  Baik             : {ringkasan['baik']}")
print("=" * 66)
print()
print("  [SELESAI] Jalankan berikutnya : streamlit run app.py")
print("=" * 66)
