# -*- coding: utf-8 -*-
"""
=============================================================================
  SIMANTAP UMKM - Dinas Koperasi dan UKM Kabupaten Banyumas
  ---------------------------------------------------------------------------
  generate_umkm.py : Membuat data 40 UMKM binaan Banyumas.

  Dijalankan SETELAH train.py. Untuk setiap UMKM:
    1. Mengambil sampel ulasan dari kategori PRDECT-ID yang sesuai
    2. Menjalankan model SVM untuk klasifikasi sentimen tiap ulasan
    3. Mendeteksi aspek (Kualitas / Packaging / Harga / Pengiriman)
    4. Menghitung skor kesehatan UMKM (0-100) & status
    5. Menyusun rekomendasi tindakan untuk Diskop

  Output : umkm_data.json
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
# 1. DATA MASTER 40 UMKM BANYUMAS
# ─────────────────────────────────────────────────────────────────────────────
UMKM_MASTER = [
    # ── KULINER ──────────────────────────────────────────────────────────
    {"id": "U001", "nama": "Mendoan Pak Karjo",         "pemilik": "Karjo Sutrisno",    "kategori": "Kuliner",    "lokasi": "Purwokerto Timur", "produk_utama": "Mendoan & gorengan khas Banyumas", "prdect_category": "Food and Drink"},
    {"id": "U002", "nama": "Keripik Tempe Bu Sari",     "pemilik": "Sari Wahyuni",      "kategori": "Kuliner",    "lokasi": "Karanglewas",      "produk_utama": "Keripik tempe aneka rasa",         "prdect_category": "Food and Drink"},
    {"id": "U003", "nama": "Getuk Goreng Pak Tohir",    "pemilik": "M. Tohir Arifin",   "kategori": "Kuliner",    "lokasi": "Sokaraja",         "produk_utama": "Getuk goreng original & modern",  "prdect_category": "Food and Drink"},
    {"id": "U004", "nama": "Jamu Herbal Bu Tini",       "pemilik": "Martini Rahayu",    "kategori": "Kuliner",    "lokasi": "Banyumas",         "produk_utama": "Jamu tradisional & wedang rempah", "prdect_category": "Food and Drink"},
    {"id": "U005", "nama": "Carang Gesing Mbak Wati",   "pemilik": "Dwi Lestari",       "kategori": "Kuliner",    "lokasi": "Sokaraja",         "produk_utama": "Carang gesing & jenang Banyumas",  "prdect_category": "Food and Drink"},
    {"id": "U006", "nama": "Kopi Robusta Pak Slamet",   "pemilik": "Slamet Riyadi",     "kategori": "Kuliner",    "lokasi": "Lumbir",           "produk_utama": "Kopi robusta bubuk & biji",        "prdect_category": "Food and Drink"},
    {"id": "U007", "nama": "Tempe Mendoan Bu Parmi",    "pemilik": "Suparmi Ningsih",   "kategori": "Kuliner",    "lokasi": "Kembaran",         "produk_utama": "Tempe mendoan & kedelai olahan",   "prdect_category": "Food and Drink"},
    {"id": "U008", "nama": "Gula Kelapa Pak Wahyu",     "pemilik": "Wahyu Santoso",     "kategori": "Kuliner",    "lokasi": "Cilongok",         "produk_utama": "Gula kelapa cetak & cair",         "prdect_category": "Food and Drink"},
    {"id": "U009", "nama": "Snack Singkong Bu Lastri",  "pemilik": "Lastri Handayani",  "kategori": "Kuliner",    "lokasi": "Ajibarang",        "produk_utama": "Keripik singkong & olahan ubi",    "prdect_category": "Food and Drink"},
    {"id": "U010", "nama": "Onde-Onde Bu Ningsih",      "pemilik": "Sri Ningsih",       "kategori": "Kuliner",    "lokasi": "Purwokerto Barat", "produk_utama": "Onde-onde & kue tradisional",      "prdect_category": "Food and Drink"},
    {"id": "U011", "nama": "Rengginang Bu Yati",        "pemilik": "Suryati Prawiro",   "kategori": "Kuliner",    "lokasi": "Rawalo",           "produk_utama": "Rengginang ketan & singkong",      "prdect_category": "Food and Drink"},
    {"id": "U012", "nama": "Herbal Pak Budi Rawalo",    "pemilik": "Budi Prasetyo",     "kategori": "Kuliner",    "lokasi": "Rawalo",           "produk_utama": "Minuman herbal & jahe instan",     "prdect_category": "Food and Drink"},
    # ── FASHION & BATIK ──────────────────────────────────────────────────
    {"id": "U013", "nama": "Batik Sekar Banyumas",      "pemilik": "Hendra Prasetya",   "kategori": "Fashion",    "lokasi": "Purwokerto",       "produk_utama": "Batik tulis & cap motif Banyumasan", "prdect_category": "Women's Fashion"},
    {"id": "U014", "nama": "Tenun Lurik Mbak Dewi",     "pemilik": "Dewi Ratnasari",    "kategori": "Fashion",    "lokasi": "Banyumas",         "produk_utama": "Kain lurik tenun tangan",          "prdect_category": "Women's Fashion"},
    {"id": "U015", "nama": "Batik Banyumasan Pak Agus", "pemilik": "Agus Wibowo",       "kategori": "Fashion",    "lokasi": "Sokaraja",         "produk_utama": "Batik kombinasi modern-tradisional", "prdect_category": "Muslim Fashion"},
    {"id": "U016", "nama": "Konveksi Berkah Bu Rini",   "pemilik": "Rini Sulistyowati", "kategori": "Fashion",    "lokasi": "Purwokerto Utara", "produk_utama": "Seragam & pakaian kerja bordir",   "prdect_category": "Men's Fashion"},
    {"id": "U017", "nama": "Busana Muslim Mbak Nisa",   "pemilik": "Anisa Pratiwi",     "kategori": "Fashion",    "lokasi": "Purwokerto Selatan", "produk_utama": "Gamis & mukena bordir khas daerah", "prdect_category": "Muslim Fashion"},
    {"id": "U018", "nama": "Kaos Sablon Pak Joni",      "pemilik": "Joni Kristanto",    "kategori": "Fashion",    "lokasi": "Purwokerto Timur", "produk_utama": "Kaos sablon custom & souvenir",    "prdect_category": "Men's Fashion"},
    # ── KERAJINAN TANGAN ─────────────────────────────────────────────────
    {"id": "U019", "nama": "Kerajinan Bambu Wangon",    "pemilik": "Supriyono",         "kategori": "Kerajinan",  "lokasi": "Wangon",           "produk_utama": "Anyaman bambu & furnitur rotan",   "prdect_category": "Carpentry"},
    {"id": "U020", "nama": "Anyaman Pandan Bu Suryani", "pemilik": "Suryani Utami",     "kategori": "Kerajinan",  "lokasi": "Ajibarang",        "produk_utama": "Tas & topi anyaman pandan",        "prdect_category": "Party Supplies and Craft"},
    {"id": "U021", "nama": "Gerabah Rawalo Pak Muji",   "pemilik": "Muji Santosa",      "kategori": "Kerajinan",  "lokasi": "Rawalo",           "produk_utama": "Gerabah & keramik tradisional",    "prdect_category": "Carpentry"},
    {"id": "U022", "nama": "Ukiran Kayu Pak Santoso",   "pemilik": "Santoso Harjono",   "kategori": "Kerajinan",  "lokasi": "Lumbir",           "produk_utama": "Ukiran kayu jati & souvenir",      "prdect_category": "Carpentry"},
    {"id": "U023", "nama": "Tas Anyaman Mbak Citra",    "pemilik": "Citra Paramita",    "kategori": "Kerajinan",  "lokasi": "Cilongok",         "produk_utama": "Tas anyaman & aksesori wanita",    "prdect_category": "Party Supplies and Craft"},
    {"id": "U024", "nama": "Souvenir Banyumas Pak Dedi", "pemilik": "Dedi Kurniawan",   "kategori": "Kerajinan",  "lokasi": "Purwokerto",       "produk_utama": "Souvenir khas Banyumas & oleh-oleh", "prdect_category": "Party Supplies and Craft"},
    {"id": "U025", "nama": "Batako & Bata Bu Aminah",   "pemilik": "Aminah Saputri",    "kategori": "Kerajinan",  "lokasi": "Sumpiuh",          "produk_utama": "Batako press & bata merah",        "prdect_category": "Household"},
    # ── PERTANIAN & OLAHAN ───────────────────────────────────────────────
    {"id": "U026", "nama": "Minyak Kayu Putih Bu Tono", "pemilik": "Sutono Warsito",    "kategori": "Pertanian",  "lokasi": "Banyumas",         "produk_utama": "Minyak kayu putih & minyak herbal", "prdect_category": "Health"},
    {"id": "U027", "nama": "Bibit Tanaman Pak Hadi",    "pemilik": "Hadi Susilo",       "kategori": "Pertanian",  "lokasi": "Cilongok",         "produk_utama": "Bibit sayur & tanaman hias",       "prdect_category": "Health"},
    {"id": "U028", "nama": "Sabun Herbal Mbak Ani",     "pemilik": "Ani Setyaningsih",  "kategori": "Pertanian",  "lokasi": "Sokaraja",         "produk_utama": "Sabun herbal aloe vera & zaitun",  "prdect_category": "Body Care"},
    {"id": "U029", "nama": "Pupuk Organik Pak Warno",   "pemilik": "Warno Haryanto",    "kategori": "Pertanian",  "lokasi": "Kalibagor",        "produk_utama": "Pupuk kompos & bio-organik",       "prdect_category": "Health"},
    {"id": "U030", "nama": "Rempah Segar Pak Joko",     "pemilik": "Joko Supriyadi",    "kategori": "Pertanian",  "lokasi": "Sokaraja",         "produk_utama": "Rempah-rempah segar & kering",     "prdect_category": "Food and Drink"},
    # ── TEKNOLOGI & JASA ─────────────────────────────────────────────────
    {"id": "U031", "nama": "Service Elektronik Pak Rio", "pemilik": "Rio Firmansyah",   "kategori": "Teknologi",  "lokasi": "Purwokerto",       "produk_utama": "Servis HP, laptop & elektronik",   "prdect_category": "Electronics"},
    {"id": "U032", "nama": "Print & Sablon Mbak Dian",  "pemilik": "Dian Permatasari",  "kategori": "Teknologi",  "lokasi": "Purwokerto Utara", "produk_utama": "Cetak digital & sablon kustomisasi", "prdect_category": "Office & Stationery"},
    {"id": "U033", "nama": "Aksesoris HP Bu Yuni",      "pemilik": "Yuni Hartati",      "kategori": "Teknologi",  "lokasi": "Sokaraja",         "produk_utama": "Aksesoris HP & casing custom",     "prdect_category": "Phones and Tablets"},
    # ── PERALATAN & RUMAH TANGGA ─────────────────────────────────────────
    {"id": "U034", "nama": "Perabot Dapur Pak Suryo",   "pemilik": "Suryo Atmojo",      "kategori": "Rumah Tangga", "lokasi": "Purwokerto",     "produk_utama": "Peralatan masak & dapur",          "prdect_category": "Kitchen"},
    {"id": "U035", "nama": "Mebel Rotan Pak Tarno",     "pemilik": "Sutarno Hadi",      "kategori": "Rumah Tangga", "lokasi": "Rawalo",         "produk_utama": "Mebel rotan & kayu jati",          "prdect_category": "Household"},
    {"id": "U036", "nama": "Alat Pertanian Bu Kasih",   "pemilik": "Kasih Rahayu",      "kategori": "Rumah Tangga", "lokasi": "Ajibarang",      "produk_utama": "Peralatan pertanian tangan",       "prdect_category": "Household"},
    # ── KESEHATAN & KECANTIKAN ───────────────────────────────────────────
    {"id": "U037", "nama": "Apotek Herbal Mbak Fitri",  "pemilik": "Fitri Andriani",    "kategori": "Kesehatan",  "lokasi": "Purwokerto",       "produk_utama": "Produk herbal & suplemen tradisional", "prdect_category": "Health"},
    {"id": "U038", "nama": "Kosmetik Lokal Bu Endang",  "pemilik": "Endang Kusumawati", "kategori": "Kesehatan",  "lokasi": "Purwokerto Selatan", "produk_utama": "Kosmetik natural bahan lokal",    "prdect_category": "Beauty"},
    {"id": "U039", "nama": "Perawatan Rambut Pak Dono", "pemilik": "Dono Prayitno",     "kategori": "Kesehatan",  "lokasi": "Banyumas",         "produk_utama": "Produk perawatan rambut herbal",   "prdect_category": "Body Care"},
    {"id": "U040", "nama": "Alat Olahraga Bu Mira",     "pemilik": "Mira Yuniarti",     "kategori": "Kesehatan",  "lokasi": "Purwokerto Timur", "produk_utama": "Alat olahraga rumahan & fitness",  "prdect_category": "Sport"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. KAMUS KEYWORD ASPEK & ATURAN REKOMENDASI
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
    ("Packaging", "negatif"):  ["Ikutkan dalam pelatihan packaging Diskop",
                                "Konsultasi desain kemasan dengan mentor"],
    ("Kualitas", "negatif"):   ["Kunjungan lapangan untuk evaluasi proses produksi",
                                "Pendampingan quality control"],
    ("Harga", "negatif"):      ["Analisis struktur biaya dengan konsultan Diskop",
                                "Evaluasi strategi penetapan harga"],
    ("Pengiriman", "negatif"): ["Rekomendasi mitra ekspedisi yang lebih handal",
                                "Pelatihan manajemen pengiriman"],
}

REKOMENDASI_BAIK = [
    "Pertahankan kualitas produk dan layanan saat ini",
    "Berpotensi dijadikan UMKM percontohan binaan Diskop",
]


def deteksi_aspek(teks_asli: str) -> dict:
    """Deteksi aspek berbasis kata kunci pada teks ulasan asli."""
    teks = str(teks_asli).lower()
    return {aspek: any(kw in teks for kw in keywords)
            for aspek, keywords in ASPEK_KEYWORDS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PERHITUNGAN SKOR & STATUS
# ─────────────────────────────────────────────────────────────────────────────
def hitung_skor(reviews_hasil: list) -> int:
    """Skor 0-100: 60% rasio review positif + 40% rata-rata confidence positif."""
    if not reviews_hasil:
        return 50
    positif = [r for r in reviews_hasil if r["sentimen"] == "Positive"]
    rasio_positif = len(positif) / len(reviews_hasil)
    rata_confidence = float(np.mean([r["confidence_positif"] for r in reviews_hasil]))
    skor = (rasio_positif * 0.6 + rata_confidence * 0.4) * 100
    return int(round(skor))


def tentukan_status(skor: int) -> str:
    if skor >= 75:
        return "Baik"
    elif skor >= 55:
        return "Pantau"
    elif skor >= 40:
        return "Perlu Perhatian"
    else:
        return "Kritis"


def top_kata(processed_list: list, n: int = 5) -> list:
    """Ambil n kata paling sering dari daftar teks yang sudah dipreprocess."""
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
# 4. LOAD MODEL & DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 66)
print("  SIMANTAP UMKM - GENERATE DATA 40 UMKM BANYUMAS")
print("=" * 66)
print()
print("[1/3] Memuat model, vectorizer, stopword & dataset ...")

try:
    vectorizer = joblib.load("vectorizer.pkl")
    svm_model = joblib.load("svm_model.pkl")
    STOPWORDS_ID = joblib.load("stopwords_set.pkl")
except FileNotFoundError as e:
    print("  [ERROR] File tidak ditemukan: " + str(e.filename))
    print("  Jalankan terlebih dahulu: python train.py")
    raise SystemExit(1)

stemmer = StemmerFactory().create_stemmer()
df_raw = pd.read_csv("PRDECT-ID Dataset.csv")
print("      [OK] Semua sumber data dimuat.")


def preprocess(text: str) -> str:
    """Pipeline preprocessing — IDENTIK dengan train.py."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


classes = list(svm_model.classes_)
POS_IDX = classes.index("Positive")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PROSES SETIAP UMKM
# ─────────────────────────────────────────────────────────────────────────────
print()
print("[2/3] Menganalisis ulasan untuk 40 UMKM ...")

hasil_umkm = []
aspek_keluhan_global = {a: 0 for a in ASPEK_KEYWORDS}

for umkm in UMKM_MASTER:
    seed = int(umkm["id"][1:])           # "U001" -> 1 (reproducible)
    rng = np.random.RandomState(seed)

    # a. Ambil ulasan dari kategori PRDECT-ID yang cocok
    subset = df_raw[df_raw["Category"] == umkm["prdect_category"]]
    semua_review = subset["Customer Review"].dropna().astype(str).tolist()

    # b. Sampel 15-40 ulasan
    n_sampel = int(rng.randint(15, 41))
    if len(semua_review) > n_sampel:
        idx = rng.choice(len(semua_review), size=n_sampel, replace=False)
        sampel = [semua_review[i] for i in idx]
    else:
        sampel = semua_review

    # c. Preprocess + buang yang kosong
    pasangan = [(r, preprocess(r)) for r in sampel]
    pasangan = [(raw, pro) for raw, pro in pasangan if pro.strip()]
    if not pasangan:
        continue
    raws = [p[0] for p in pasangan]
    procs = [p[1] for p in pasangan]

    # d. Klasifikasi sentimen dengan model SVM
    vecs = vectorizer.transform(procs)
    probas = svm_model.predict_proba(vecs)

    reviews_hasil = []
    for raw, pro, pr in zip(raws, procs, probas):
        p_pos = float(pr[POS_IDX])
        reviews_hasil.append({
            "review": raw,
            "processed": pro,
            "sentimen": "Positive" if p_pos >= 0.5 else "Negative",
            "confidence_positif": p_pos,
            "aspek": deteksi_aspek(raw),
        })

    total = len(reviews_hasil)
    review_positif = sum(1 for r in reviews_hasil if r["sentimen"] == "Positive")
    review_negatif = total - review_positif

    # e. Skor & status UMKM
    skor = hitung_skor(reviews_hasil)
    status = tentukan_status(skor)

    # f. Analisis per-aspek
    aspek_detail = {}
    for nama_aspek in ASPEK_KEYWORDS:
        disebut = [r for r in reviews_hasil if r["aspek"][nama_aspek]]
        if disebut:
            pos = sum(1 for r in disebut if r["sentimen"] == "Positive")
            skor_pos = int(round(pos / len(disebut) * 100))
            terdeteksi = True
        else:
            skor_pos = 50
            terdeteksi = False
        keluhan = sum(1 for r in disebut if r["sentimen"] == "Negative")
        aspek_keluhan_global[nama_aspek] += keluhan
        aspek_detail[nama_aspek] = {
            "skor_positif": skor_pos,
            "terdeteksi": terdeteksi,
            "keluhan_count": keluhan,
        }

    # g. Kata kunci dominan
    teks_pos = [r["processed"] for r in reviews_hasil if r["sentimen"] == "Positive"]
    teks_neg = [r["processed"] for r in reviews_hasil if r["sentimen"] == "Negative"]
    kata_pos = top_kata(teks_pos, 5)
    kata_neg = top_kata(teks_neg, 5)

    # h. Masalah utama + rekomendasi
    terdeteksi_list = [a for a, d in aspek_detail.items() if d["terdeteksi"]]
    total_keluhan = sum(d["keluhan_count"] for d in aspek_detail.values())

    if total_keluhan == 0 or not terdeteksi_list:
        masalah_utama = "Tidak ada"
        rekomendasi = list(REKOMENDASI_BAIK)
    else:
        # Urutkan aspek bermasalah dari skor_positif terendah
        urut = sorted(terdeteksi_list, key=lambda a: aspek_detail[a]["skor_positif"])
        masalah_utama = urut[0]
        rekomendasi = list(REKOMENDASI_RULES.get((masalah_utama, "negatif"), []))
        # Tambah 1 rekomendasi dari aspek terburuk kedua bila masih bermasalah
        if len(urut) > 1 and aspek_detail[urut[1]]["skor_positif"] < 55:
            extra = REKOMENDASI_RULES.get((urut[1], "negatif"), [])
            if extra:
                rekomendasi.append(extra[0])
        rekomendasi = rekomendasi[:3]

    # i. Contoh ulasan
    pos_sorted = sorted(
        [r for r in reviews_hasil if r["sentimen"] == "Positive"],
        key=lambda r: r["confidence_positif"], reverse=True,
    )
    neg_sorted = sorted(
        [r for r in reviews_hasil if r["sentimen"] == "Negative"],
        key=lambda r: r["confidence_positif"],
    )
    contoh_pos = potong(pos_sorted[0]["review"]) if pos_sorted else ""
    contoh_neg = potong(neg_sorted[0]["review"]) if neg_sorted else ""

    hasil_umkm.append({
        "id": umkm["id"],
        "nama": umkm["nama"],
        "pemilik": umkm["pemilik"],
        "kategori": umkm["kategori"],
        "lokasi": umkm["lokasi"],
        "produk_utama": umkm["produk_utama"],
        "skor": skor,
        "status": status,
        "total_review": total,
        "review_positif": review_positif,
        "review_negatif": review_negatif,
        "aspek": aspek_detail,
        "kata_kunci_positif": kata_pos,
        "kata_kunci_negatif": kata_neg,
        "masalah_utama": masalah_utama,
        "rekomendasi": rekomendasi,
        "contoh_review_positif": contoh_pos,
        "contoh_review_negatif": contoh_neg,
    })
    print(f"      {umkm['id']} {umkm['nama']:<32} skor={skor:<3} {status}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. RINGKASAN GLOBAL & SIMPAN JSON
# ─────────────────────────────────────────────────────────────────────────────
print()
print("[3/3] Menyusun ringkasan & menyimpan umkm_data.json ...")

ringkasan = {
    "kritis": sum(1 for u in hasil_umkm if u["status"] == "Kritis"),
    "perlu_perhatian": sum(1 for u in hasil_umkm if u["status"] == "Perlu Perhatian"),
    "pantau": sum(1 for u in hasil_umkm if u["status"] == "Pantau"),
    "baik": sum(1 for u in hasil_umkm if u["status"] == "Baik"),
}

output = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_umkm": len(hasil_umkm),
    "ringkasan": ringkasan,
    "aspek_keluhan_global": dict(
        sorted(aspek_keluhan_global.items(), key=lambda x: x[1], reverse=True)
    ),
    "umkm": hasil_umkm,
}

with open("umkm_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("      [OK] umkm_data.json tersimpan (" + str(len(hasil_umkm)) + " UMKM).")
print()
print("=" * 66)
print("  RINGKASAN STATUS UMKM")
print("-" * 66)
print(f"  Kritis           : {ringkasan['kritis']}")
print(f"  Perlu Perhatian  : {ringkasan['perlu_perhatian']}")
print(f"  Pantau           : {ringkasan['pantau']}")
print(f"  Baik             : {ringkasan['baik']}")
print("=" * 66)
print("  [SELESAI] Langkah berikutnya : streamlit run app.py")
print("=" * 66)
