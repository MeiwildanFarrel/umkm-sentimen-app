# PROMPT UNTUK CLAUDE OPUS — ROMBAK TOTAL PROYEK
## Sistem Analisis Sentimen UMKM Banyumas — Dashboard Diskop

---

## IDENTITAS TUGAS

Kamu diminta merombak total sebuah folder proyek Python. Folder ini berisi sistem analisis sentimen untuk Dinas Koperasi dan UKM (Diskop) Kabupaten Banyumas. Kamu harus menulis ulang semua file dari nol, menghapus file yang tidak diperlukan, dan menghasilkan proyek yang bersih, fungsional, dan siap di-demo.

**Bahasa kode:** Python  
**Framework UI:** Streamlit  
**ML Library:** scikit-learn  
**NLP:** PySastrawi + NLTK stopwords  
**Dataset:** PRDECT-ID Dataset.csv (sudah ada di folder, jangan hapus)

---

## ISI FOLDER SAAT INI (yang perlu kamu tangani)

```
umkm-sentimen-app/
├── PRDECT-ID Dataset.csv   ← JANGAN HAPUS, ini dataset utama
├── app.py                  ← HAPUS, tulis ulang dari nol
├── train.py                ← HAPUS, tulis ulang dari nol
├── requirements.txt        ← UPDATE
├── nb_model.pkl            ← HAPUS (akan di-generate ulang oleh train.py)
├── svm_model.pkl           ← HAPUS (akan di-generate ulang)
├── rf_model.pkl            ← HAPUS (akan di-generate ulang)
├── vectorizer.pkl          ← HAPUS (akan di-generate ulang)
```

---

## STRUKTUR FOLDER SETELAH SELESAI

```
umkm-sentimen-app/
├── PRDECT-ID Dataset.csv        ← tetap ada
├── train.py                     ← BARU (tulis ulang)
├── generate_umkm.py             ← BARU (file baru, buat dari nol)
├── app.py                       ← BARU (tulis ulang total, multi-page)
├── requirements.txt             ← UPDATE
│
│   (file-file berikut di-generate saat script dijalankan, jangan buat manual)
├── vectorizer.pkl               (output train.py)
├── nb_model.pkl                 (output train.py)
├── svm_model.pkl                (output train.py)
├── rf_model.pkl                 (output train.py)
├── evaluation_results.json      (output train.py)
└── umkm_data.json               (output generate_umkm.py)
```

**Urutan menjalankan:**
```
1. python train.py           → hasilkan model .pkl + evaluation_results.json
2. python generate_umkm.py   → hasilkan umkm_data.json
3. streamlit run app.py      → buka dashboard
```

---

## DESIGN SYSTEM — WAJIB DIIKUTI

Ini adalah aturan visual yang harus konsisten di seluruh app.py. Diskop adalah lembaga pemerintah, jadi desain harus **bersih, profesional, tidak berlebihan**.

### Palet Warna

```python
# Warna utama (CSS variables / hex)
MERAH_PRIMER    = "#B91C1C"   # Merah koperasi (dark, profesional)
MERAH_MUDA      = "#FEE2E2"   # Merah pastel (background status kritis)
HIJAU_PRIMER    = "#15803D"   # Hijau koperasi
HIJAU_MUDA      = "#DCFCE7"   # Hijau pastel (background status baik)
KUNING_MUDA     = "#FEF3C7"   # Kuning pastel (status perlu perhatian)
ABU_MUDA        = "#F3F4F6"   # Abu pastel (status pantau)

BG_HALAMAN      = "#F8F9FA"   # Background halaman utama
BG_SIDEBAR      = "#FFFFFF"   # Sidebar putih bersih
BG_CARD         = "#FFFFFF"   # Card putih
BORDER          = "#E5E7EB"   # Border semua card dan input
TEKS_UTAMA      = "#111827"   # Teks hitam utama
TEKS_SEKUNDER   = "#6B7280"   # Teks abu-abu label/subtitle
TEKS_SANGAT_ABU = "#9CA3AF"   # Teks hint / placeholder
```

### Status UMKM — Warna & Logika

| Skor | Status | Bg | Teks | Border |
|---|---|---|---|---|
| 75–100 | Baik | #DCFCE7 | #15803D | #BBF7D0 |
| 55–74 | Pantau | #F3F4F6 | #6B7280 | #E5E7EB |
| 40–54 | Perlu Perhatian | #FEF3C7 | #B45309 | #FDE68A |
| 0–39 | Kritis | #FEE2E2 | #B91C1C | #FECACA |

### Tipografi & Layout

- Font: bawaan Streamlit (sans-serif system font) — tidak perlu import Google Fonts
- Sidebar lebar: 260px
- Padding halaman: 1.5rem
- Card border-radius: 8px (sedikit rounded, tidak terlalu bulat)
- Shadow card: `0 1px 3px rgba(0,0,0,0.08)`

### Aturan Ikon

- **Boleh** menggunakan emoji/ikon tapi HEMAT — hanya untuk signifier navigasi dan status kritis
- **Dilarang**: emoji dekoratif yang tidak informatif, terlalu banyak ikon di satu halaman
- Contoh penggunaan yang BENAR: ikon pada item menu sidebar, badge status UMKM
- Contoh yang SALAH: setiap judul section pakai emoji, setiap card punya ikon

### CSS Global yang Wajib Diterapkan di app.py

```python
GLOBAL_CSS = """
<style>
/* Sembunyikan toolbar default Streamlit */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Background halaman */
.stApp { background-color: #F8F9FA; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    width: 260px !important;
}
section[data-testid="stSidebar"] * { color: #374151 !important; }

/* Hapus padding default Streamlit */
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Tombol primer */
.stButton > button {
    background-color: #B91C1C !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
}
.stButton > button:hover {
    background-color: #991B1B !important;
}

/* Input fields */
.stTextInput > div > input, .stSelectbox > div > div {
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    background: #FFFFFF !important;
}

/* Tabel */
.stDataFrame { border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden; }

/* Divider */
hr { border: none; border-top: 1px solid #E5E7EB; margin: 1rem 0; }
</style>
"""
```

---

## FILE 1: `requirements.txt`

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
tqdm>=4.65.0
PySastrawi>=1.2.0
nltk>=3.8.0
plotly>=5.18.0
```

---

## FILE 2: `train.py`

Script ini dijalankan SEKALI untuk melatih model. Tidak ada interaksi dengan Streamlit.

### Yang harus dilakukan:

**1. Load & Filter Dataset**
```python
df = pd.read_csv("PRDECT-ID Dataset.csv")
df = df[["Category", "Customer Review", "Sentiment"]].dropna()
df.columns = ["category", "review", "label"]
df = df[df["label"].isin(["Positive", "Negative"])].reset_index(drop=True)
```

**2. Stopwords Indonesia**
Definisikan sebagai konstanta di bagian atas file. Gunakan NLTK Indonesian stopwords PLUS tambahan manual:
```python
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS_ID = set(stopwords.words('indonesian'))
# Tambah kata-kata yang tidak relevan untuk sentimen produk
STOPWORDS_ID.update([
    'yg', 'dgn', 'utk', 'jg', 'sy', 'aja', 'udah', 'udh', 'gak', 'ga',
    'gk', 'bgt', 'bngt', 'banget', 'sama', 'jadi', 'bisa', 'ada', 'ini',
    'itu', 'yang', 'dan', 'di', 'ke', 'dari', 'nya', 'ku', 'mu', 'sih',
    'dong', 'deh', 'loh', 'lho', 'kok', 'kan', 'ya', 'dg', 'br', 'tp',
    'tpi', 'krna', 'krn', 'karna', 'makasih', 'terimakasih', 'terima',
    'kasih', 'hatur', 'nuhun', 'maturnuwun', 'matur'
])
```

**3. Preprocessing Pipeline**
```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text: str) -> str:
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
```

Terapkan dengan progress bar tqdm.

**4. TF-IDF & Split**
```python
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Split 80/20 stratified, random_state=42
```

**5. Training 3 Model**

PENTING: SVM harus di-wrap `CalibratedClassifierCV` agar bisa `predict_proba`:
```python
from sklearn.calibration import CalibratedClassifierCV

nb_model  = MultinomialNB(alpha=0.5)
svm_base  = LinearSVC(C=1.0, max_iter=3000, random_state=42)
svm_model = CalibratedClassifierCV(svm_base, cv=3)  # ← PENTING
rf_model  = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
```

**6. Cross-Validation**
Untuk setiap model, jalankan StratifiedKFold 5-fold dan catat mean ± std F1-weighted.

**7. Simpan evaluation_results.json**
```json
{
  "Naive Bayes": {
    "accuracy": 0.89,
    "f1_weighted": 0.89,
    "f1_positive": 0.88,
    "f1_negative": 0.90,
    "precision_positive": 0.87, "recall_positive": 0.89,
    "precision_negative": 0.91, "recall_negative": 0.90,
    "confusion_matrix": [[TN, FP], [FN, TP]],
    "cv_f1_mean": 0.88, "cv_f1_std": 0.012
  },
  "SVM": { ... },
  "Random Forest": { ... },
  "best_model": "SVM",
  "dataset_info": {
    "total_samples": 5400,
    "train_samples": 4320,
    "test_samples": 1080,
    "positive_count": 2579,
    "negative_count": 2821,
    "n_features": 10000
  }
}
```

**8. Simpan file pkl**
```
vectorizer.pkl, nb_model.pkl, svm_model.pkl, rf_model.pkl
```

Juga simpan `stopwords_set.pkl` (set STOPWORDS_ID) agar app.py bisa load dan gunakan versi yang identik.

---

## FILE 3: `generate_umkm.py`

Script ini dijalankan SETELAH train.py. Fungsinya: membuat data UMKM Banyumas fiktif-realistis, assign review dari PRDECT-ID ke masing-masing UMKM, jalankan model untuk klasifikasi, hitung skor, simpan ke `umkm_data.json`.

### DAFTAR 40 UMKM BANYUMAS — HARDCODE INI PERSIS

```python
UMKM_MASTER = [
    # ── KULINER (pakai reviews dari: Food and Drink) ──────────────────────
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

    # ── FASHION & BATIK (pakai reviews dari: Women's Fashion, Muslim Fashion, Men's Fashion) ──
    {"id": "U013", "nama": "Batik Sekar Banyumas",      "pemilik": "Hendra Prasetya",   "kategori": "Fashion",    "lokasi": "Purwokerto",       "produk_utama": "Batik tulis & cap motif Banyumasan","prdect_category": "Women's Fashion"},
    {"id": "U014", "nama": "Tenun Lurik Mbak Dewi",     "pemilik": "Dewi Ratnasari",    "kategori": "Fashion",    "lokasi": "Banyumas",         "produk_utama": "Kain lurik tenun tangan",          "prdect_category": "Women's Fashion"},
    {"id": "U015", "nama": "Batik Banyumasan Pak Agus", "pemilik": "Agus Wibowo",       "kategori": "Fashion",    "lokasi": "Sokaraja",         "produk_utama": "Batik kombinasi modern-tradisional","prdect_category": "Muslim Fashion"},
    {"id": "U016", "nama": "Konveksi Berkah Bu Rini",   "pemilik": "Rini Sulistyowati", "kategori": "Fashion",    "lokasi": "Purwokerto Utara", "produk_utama": "Seragam & pakaian kerja bordir",    "prdect_category": "Men's Fashion"},
    {"id": "U017", "nama": "Busana Muslim Mbak Nisa",   "pemilik": "Anisa Pratiwi",     "kategori": "Fashion",    "lokasi": "Purwokerto Selatan","produk_utama": "Gamis & mukena bordir khas daerah", "prdect_category": "Muslim Fashion"},
    {"id": "U018", "nama": "Kaos Sablon Pak Joni",      "pemilik": "Joni Kristanto",    "kategori": "Fashion",    "lokasi": "Purwokerto Timur", "produk_utama": "Kaos sablon custom & souvenir",    "prdect_category": "Men's Fashion"},

    # ── KERAJINAN TANGAN (pakai reviews dari: Party Supplies and Craft, Carpentry, Household) ──
    {"id": "U019", "nama": "Kerajinan Bambu Wangon",    "pemilik": "Supriyono",         "kategori": "Kerajinan",  "lokasi": "Wangon",           "produk_utama": "Anyaman bambu & furnitur rotan",   "prdect_category": "Carpentry"},
    {"id": "U020", "nama": "Anyaman Pandan Bu Suryani", "pemilik": "Suryani Utami",     "kategori": "Kerajinan",  "lokasi": "Ajibarang",        "produk_utama": "Tas & topi anyaman pandan",        "prdect_category": "Party Supplies and Craft"},
    {"id": "U021", "nama": "Gerabah Rawalo Pak Muji",   "pemilik": "Muji Santosa",      "kategori": "Kerajinan",  "lokasi": "Rawalo",           "produk_utama": "Gerabah & keramik tradisional",    "prdect_category": "Carpentry"},
    {"id": "U022", "nama": "Ukiran Kayu Pak Santoso",   "pemilik": "Santoso Harjono",   "kategori": "Kerajinan",  "lokasi": "Lumbir",           "produk_utama": "Ukiran kayu jati & souvenir",      "prdect_category": "Carpentry"},
    {"id": "U023", "nama": "Tas Anyaman Mbak Citra",    "pemilik": "Citra Paramita",    "kategori": "Kerajinan",  "lokasi": "Cilongok",         "produk_utama": "Tas anyaman & aksesori wanita",    "prdect_category": "Party Supplies and Craft"},
    {"id": "U024", "nama": "Souvenir Banyumas Pak Dedi","pemilik": "Dedi Kurniawan",    "kategori": "Kerajinan",  "lokasi": "Purwokerto",       "produk_utama": "Souvenir khas Banyumas & oleh-oleh","prdect_category": "Party Supplies and Craft"},
    {"id": "U025", "nama": "Batako & Bata Bu Aminah",   "pemilik": "Aminah Saputri",    "kategori": "Kerajinan",  "lokasi": "Sumpiuh",          "produk_utama": "Batako press & bata merah",        "prdect_category": "Household"},

    # ── PERTANIAN & OLAHAN (pakai reviews dari: Health, Body Care, Food and Drink) ──
    {"id": "U026", "nama": "Minyak Kayu Putih Bu Tono", "pemilik": "Sutono Warsito",    "kategori": "Pertanian",  "lokasi": "Banyumas",         "produk_utama": "Minyak kayu putih & minyak herbal", "prdect_category": "Health"},
    {"id": "U027", "nama": "Bibit Tanaman Pak Hadi",    "pemilik": "Hadi Susilo",       "kategori": "Pertanian",  "lokasi": "Cilongok",         "produk_utama": "Bibit sayur & tanaman hias",       "prdect_category": "Health"},
    {"id": "U028", "nama": "Sabun Herbal Mbak Ani",     "pemilik": "Ani Setyaningsih",  "kategori": "Pertanian",  "lokasi": "Sokaraja",         "produk_utama": "Sabun herbal aloe vera & zaitun",  "prdect_category": "Body Care"},
    {"id": "U029", "nama": "Pupuk Organik Pak Warno",   "pemilik": "Warno Haryanto",    "kategori": "Pertanian",  "lokasi": "Kalibagor",        "produk_utama": "Pupuk kompos & bio-organik",       "prdect_category": "Health"},
    {"id": "U030", "nama": "Rempah Segar Pak Joko",     "pemilik": "Joko Supriyadi",    "kategori": "Pertanian",  "lokasi": "Sokaraja",         "produk_utama": "Rempah-rempah segar & kering",    "prdect_category": "Food and Drink"},

    # ── TEKNOLOGI & JASA (pakai reviews dari: Electronics, Computers and Laptops) ──
    {"id": "U031", "nama": "Service Elektronik Pak Rio","pemilik": "Rio Firmansyah",    "kategori": "Teknologi",  "lokasi": "Purwokerto",       "produk_utama": "Servis HP, laptop & elektronik",   "prdect_category": "Electronics"},
    {"id": "U032", "nama": "Print & Sablon Mbak Dian",  "pemilik": "Dian Permatasari",  "kategori": "Teknologi",  "lokasi": "Purwokerto Utara", "produk_utama": "Cetak digital & sablon kustomisasi","prdect_category": "Office & Stationery"},
    {"id": "U033", "nama": "Aksesoris HP Bu Yuni",      "pemilik": "Yuni Hartati",      "kategori": "Teknologi",  "lokasi": "Sokaraja",         "produk_utama": "Aksesoris HP & casing custom",     "prdect_category": "Phones and Tablets"},

    # ── PERALATAN & RUMAH TANGGA ──
    {"id": "U034", "nama": "Perabot Dapur Pak Suryo",   "pemilik": "Suryo Atmojo",      "kategori": "Rumah Tangga","lokasi": "Purwokerto",      "produk_utama": "Peralatan masak & dapur",          "prdect_category": "Kitchen"},
    {"id": "U035", "nama": "Mebel Rotan Pak Tarno",     "pemilik": "Sutarno Hadi",      "kategori": "Rumah Tangga","lokasi": "Rawalo",          "produk_utama": "Mebel rotan & kayu jati",          "prdect_category": "Household"},
    {"id": "U036", "nama": "Alat Pertanian Bu Kasih",   "pemilik": "Kasih Rahayu",      "kategori": "Rumah Tangga","lokasi": "Ajibarang",       "produk_utama": "Peralatan pertanian tangan",       "prdect_category": "Household"},

    # ── KESEHATAN & KECANTIKAN ──
    {"id": "U037", "nama": "Apotek Herbal Mbak Fitri",  "pemilik": "Fitri Andriani",    "kategori": "Kesehatan",  "lokasi": "Purwokerto",       "produk_utama": "Produk herbal & suplemen tradisional","prdect_category": "Health"},
    {"id": "U038", "nama": "Kosmetik Lokal Bu Endang",  "pemilik": "Endang Kusumawati", "kategori": "Kesehatan",  "lokasi": "Purwokerto Selatan","produk_utama": "Kosmetik natural bahan lokal",     "prdect_category": "Beauty"},
    {"id": "U039", "nama": "Perawatan Rambut Pak Dono", "pemilik": "Dono Prayitno",     "kategori": "Kesehatan",  "lokasi": "Banyumas",         "produk_utama": "Produk perawatan rambut herbal",   "prdect_category": "Body Care"},
    {"id": "U040", "nama": "Alat Olahraga Bu Mira",     "pemilik": "Mira Yuniarti",     "kategori": "Kesehatan",  "lokasi": "Purwokerto Timur", "produk_utama": "Alat olahraga rumahan & fitness",   "prdect_category": "Sport"},
]
```

### Logika generate_umkm.py

```python
import pandas as pd, numpy as np, json, joblib, re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Load model terbaik (SVM)
vectorizer = joblib.load("vectorizer.pkl")
svm_model  = joblib.load("svm_model.pkl")
stopwords  = joblib.load("stopwords_set.pkl")
stemmer    = StemmerFactory().create_stemmer()

# 2. Load dataset
df_raw = pd.read_csv("PRDECT-ID Dataset.csv")

# 3. Untuk setiap UMKM:
#    a. Filter reviews dari prdect_category yang cocok
#    b. Sample 15–40 reviews (random, seed=id UMKM untuk reproducibility)
#    c. Preprocess setiap review (fungsi preprocess yang SAMA dengan train.py)
#    d. Jalankan model → predict_proba → sentimen + confidence
#    e. Aspect detection (keyword-based, lihat di bawah)
#    f. Hitung skor UMKM

# 4. Hitung skor UMKM
def hitung_skor(reviews_hasil: list) -> int:
    """
    Skor 0-100 berdasarkan:
    - 60% dari rasio review positif
    - 40% dari rata-rata confidence positif
    """
    if not reviews_hasil:
        return 50
    positif = [r for r in reviews_hasil if r["sentimen"] == "Positive"]
    rasio_positif = len(positif) / len(reviews_hasil)
    rata_confidence = np.mean([r["confidence_positif"] for r in reviews_hasil])
    skor = (rasio_positif * 0.6 + rata_confidence * 0.4) * 100
    return int(round(skor))

def tentukan_status(skor: int) -> str:
    if skor >= 75: return "Baik"
    elif skor >= 55: return "Pantau"
    elif skor >= 40: return "Perlu Perhatian"
    else: return "Kritis"
```

### Aspect Detection Keywords

```python
ASPEK_KEYWORDS = {
    "Kualitas": [
        "kualitas", "bagus", "jelek", "rusak", "cacat", "sesuai", "tidak sesuai",
        "bahan", "material", "awet", "tahan", "rapuh", "original", "asli", "palsu",
        "mantap", "memuaskan", "kecewa", "mengecewakan", "sempurna", "buruk",
        "rasa", "tekstur", "warna", "bentuk", "ukuran", "fungsi", "berfungsi",
        "mati", "error", "lumayan", "oke", "produk", "barang"
    ],
    "Packaging": [
        "packing", "packaging", "kemasan", "bungkus", "kotak", "dos", "kardus",
        "bubble", "wrap", "aman", "selamat", "pecah", "retak", "penyok",
        "lecet", "rapi", "berantakan", "terbuka", "bocor"
    ],
    "Harga": [
        "harga", "mahal", "murah", "worth", "terjangkau", "overpriced",
        "ekonomis", "hemat", "bersaing", "sesuai harga", "nilai", "bayar",
        "ongkir", "gratis ongkir", "promo", "diskon", "biaya"
    ],
    "Pengiriman": [
        "pengiriman", "kirim", "ekspedisi", "kurir", "datang", "tiba",
        "sampai", "cepat", "lambat", "lama", "tepat waktu", "telat",
        "delay", "tracking", "resi", "j&t", "jne", "sicepat", "shopee express"
    ]
}

def deteksi_aspek(teks_asli: str) -> dict:
    teks = teks_asli.lower()
    return {aspek: any(kw in teks for kw in keywords)
            for aspek, keywords in ASPEK_KEYWORDS.items()}
```

### Format Output `umkm_data.json`

```json
{
  "generated_at": "2025-01-15 08:30:00",
  "total_umkm": 40,
  "ringkasan": {
    "kritis": 5,
    "perlu_perhatian": 8,
    "pantau": 15,
    "baik": 12
  },
  "aspek_keluhan_global": {
    "Packaging": 68,
    "Kualitas": 44,
    "Pengiriman": 31,
    "Harga": 18
  },
  "umkm": [
    {
      "id": "U001",
      "nama": "Mendoan Pak Karjo",
      "pemilik": "Karjo Sutrisno",
      "kategori": "Kuliner",
      "lokasi": "Purwokerto Timur",
      "produk_utama": "Mendoan & gorengan khas Banyumas",
      "skor": 72,
      "status": "Pantau",
      "total_review": 28,
      "review_positif": 20,
      "review_negatif": 8,
      "aspek": {
        "Kualitas":    {"skor_positif": 78, "terdeteksi": true,  "keluhan_count": 3},
        "Packaging":   {"skor_positif": 42, "terdeteksi": true,  "keluhan_count": 9},
        "Harga":       {"skor_positif": 65, "terdeteksi": true,  "keluhan_count": 2},
        "Pengiriman":  {"skor_positif": 55, "terdeteksi": false, "keluhan_count": 0}
      },
      "kata_kunci_positif": ["enak", "mantap", "sesuai", "cepat"],
      "kata_kunci_negatif": ["kemasan", "bocor", "tidak rapi"],
      "masalah_utama": "Packaging",
      "rekomendasi": [
        "Ikutkan dalam workshop kemasan bulan depan",
        "Konsultasi desain kemasan dengan mentor Diskop"
      ],
      "contoh_review_positif": "Mendoannya enak banget, sesuai ekspektasi...",
      "contoh_review_negatif": "Sayang kemasannya kurang rapi, agak bocor..."
    }
    // ... 39 UMKM lainnya
  ]
}
```

Untuk `aspek.skor_positif`: persentase review yang POSITIF di antara review yang mengandung kata kunci aspek tersebut.

Untuk `kata_kunci_positif` dan `kata_kunci_negatif`: ambil 3-5 kata yang paling sering muncul di review positif/negatif (setelah preprocess, hitung frekuensi, exclude stopwords).

Untuk `rekomendasi`: generate berdasarkan aturan:
```python
REKOMENDASI_RULES = {
    ("Packaging", "negatif"):   ["Ikutkan dalam pelatihan packaging Diskop", "Konsultasi desain kemasan dengan mentor"],
    ("Kualitas", "negatif"):    ["Kunjungan lapangan untuk evaluasi proses produksi", "Pendampingan quality control"],
    ("Harga", "negatif"):       ["Analisis struktur biaya dengan konsultan Diskop", "Evaluasi strategi penetapan harga"],
    ("Pengiriman", "negatif"):  ["Rekomendasi mitra ekspedisi yang lebih handal", "Pelatihan manajemen pengiriman"],
}
```
Ambil 2-3 rekomendasi berdasarkan aspek dengan skor_positif terendah (paling bermasalah).

---

## FILE 4: `app.py` — Dashboard Multi-Page Diskop

Ini adalah file terbesar dan terpenting. Struktur keseluruhan menggunakan **session state** untuk navigasi antar halaman (bukan `pages/` folder, agar lebih mudah dikontrol):

```python
# Navigasi dikontrol via:
if "halaman" not in st.session_state:
    st.session_state.halaman = "login"
if "login_ok" not in st.session_state:
    st.session_state.login_ok = False
if "umkm_dipilih" not in st.session_state:
    st.session_state.umkm_dipilih = None
```

### Halaman 0: Login

Satu-satunya halaman tanpa sidebar. Layout centered, kartu login di tengah halaman.

**Elemen:**
- Logo/nama sistem: "SIMANTAP UMKM" (Sistem Manajemen Pantau UMKM) — teks saja, tidak perlu logo file
- Judul: "Dinas Koperasi dan UKM Kabupaten Banyumas"
- Subtitle kecil: "Sistem Pemantauan Sentimen Ulasan UMKM Binaan"
- Field: Email, Password (type=password)
- Tombol login merah
- Credentials demo: `admin@diskop.banyumaskab.go.id` / `diskop2025`
- Jika salah: `st.error("Email atau password salah.")`
- Jika benar: set `st.session_state.login_ok = True`, `st.session_state.halaman = "overview"`, lalu `st.rerun()`
- Catatan kecil di bawah: "Demo: gunakan email dan password yang tertera di dokumen teknis"

### Sidebar (muncul di semua halaman KECUALI login)

```
┌─────────────────────────────┐
│  SIMANTAP UMKM              │  ← nama sistem, teks bold merah
│  Diskop Banyumas            │  ← subtitle kecil abu
├─────────────────────────────┤
│  ○ Overview                 │  ← menu item, aktif = border kiri merah + bg merah muda
│  ○ Daftar UMKM              │
│  ○ Analisis Ulasan          │
│  ○ Laporan                  │
├─────────────────────────────┤
│  Diperbarui: hari ini       │  ← kecil, abu
│  [Keluar]                   │  ← tombol kecil di bawah
└─────────────────────────────┘
```

Implementasi sidebar dengan HTML custom atau `st.sidebar` standar yang di-style.

### Halaman 1: Overview

**Header:**
- Judul: "Ringkasan Pemantauan UMKM"
- Subtitle: "Periode: {bulan ini} · {total UMKM} UMKM terpantau"

**4 Metric Card (gunakan `st.columns(4)`):**
Tampilkan menggunakan `st.metric()` atau HTML custom:
- Kritis: angka merah besar
- Perlu Perhatian: angka kuning/amber
- Pantau: angka abu
- Baik: angka hijau

**Alert otomatis (jika ada pattern):**
Cek: apakah ada 1 aspek yang mendominasi keluhan UMKM kritis (>50%)?
Jika ya, tampilkan `st.warning()` dengan pesan seperti:
`"5 dari 8 UMKM kritis bermasalah di aspek Packaging — pertimbangkan workshop bersama"`

**Grafik: Aspek Paling Banyak Dikeluhkan**
Bar chart horizontal menggunakan Plotly (bukan st.bar_chart bawaan):
- Sumbu X: persentase UMKM yang dikeluhkan
- Sumbu Y: nama aspek
- Warna bar: merah (#B91C1C) untuk keluhan >50%, hijau untuk <30%
- Tampilkan hanya aspek dari UMKM dengan status Kritis dan Perlu Perhatian
- Judul chart: "Aspek Dominan Keluhan UMKM Bermasalah"

**Tabel Ringkas UMKM Kritis (5 teratas):**
Tabel sederhana dengan kolom: Nama UMKM | Kategori | Skor | Masalah Utama | Status
Tambahkan tombol "Lihat semua →" yang navigasi ke Daftar UMKM.

### Halaman 2: Daftar UMKM

**Filter bar (horizontal, 3 kolom):**
- Filter status: selectbox (Semua / Kritis / Perlu Perhatian / Pantau / Baik)
- Filter kategori: selectbox (Semua / Kuliner / Fashion / Kerajinan / dll)
- Urutan: selectbox (Skor Terendah / Skor Tertinggi / Nama A-Z)

**Alert pattern (sama seperti Overview, hanya untuk filter aktif):**
Jika filter = Kritis dan ada pattern aspek dominan.

**Tabel UMKM:**
Render sebagai HTML table (bukan st.dataframe) agar bisa di-style dengan warna status.
Kolom: No | Nama UMKM | Kategori | Lokasi | Skor (bar mini + angka) | Masalah Utama | Status (badge berwarna) | [Detail]

Untuk "Skor (bar mini)": gunakan HTML `<div style="background: linear-gradient(...)">` lebar proporsional 0-100.

Tombol [Detail] di setiap baris: ketika diklik, set `st.session_state.umkm_dipilih = id_umkm` dan `st.session_state.halaman = "detail"`, lalu `st.rerun()`.

**Pagination jika UMKM > 20:** Tampilkan 20 per halaman dengan tombol Prev/Next.

### Halaman 3: Detail UMKM

Load data UMKM dari `umkm_data.json` berdasarkan `st.session_state.umkm_dipilih`.

**Header:**
- Tombol "← Kembali ke Daftar" (kecil, di kiri atas)
- Nama UMKM (heading besar) + Badge status (kanan)
- Info baris: Pemilik · Kategori · Lokasi · Produk: {produk_utama}

**Dua kolom utama:**

*Kolom kiri (60%):*

**Sentimen per Aspek** — untuk setiap aspek yang `terdeteksi = true`:
Tampilkan progress bar horizontal:
```
Kualitas    ████████░░  78%  (warna: hijau jika >60, kuning 40-60, merah <40)
Packaging   ████░░░░░░  42%
Harga       ██████░░░░  60%
Pengiriman  [Data belum cukup — aspek tidak disebut dalam ulasan]
```

**Ulasan Terpilih:**
Box putih bergaris:
- Satu contoh_review_positif (dengan label "Positif" badge hijau)
- Satu contoh_review_negatif (dengan label "Negatif" badge merah)

*Kolom kanan (40%):*

**Kata Kunci Dominan:**
Chip-chip kecil: hijau untuk kata kunci positif, merah untuk negatif.

**Rekomendasi Tindakan Diskop:**
List bernomor dengan border kiri merah:
```
1. Ikutkan dalam pelatihan packaging Diskop
2. Konsultasi desain kemasan dengan mentor
```

**Statistik Ringkas:**
- Total review dianalisis: {total_review}
- Positif: {review_positif} ({persentase}%)
- Negatif: {review_negatif} ({persentase}%)

### Halaman 4: Analisis Ulasan (Prediksi Langsung)

Ini adalah halaman untuk input ulasan baru secara manual — berbeda dari 3 halaman sebelumnya yang berbasis umkm_data.json.

**Layout:**
Dua kolom (40/60):

Kolom kiri — Form:
- Dropdown pilih model (Naive Bayes / SVM / Random Forest) — default SVM
- Keterangan singkat model (1 baris teks)
- Text area: "Masukkan ulasan produk"
- Tombol "Analisis" merah

Kolom kanan — Hasil (muncul setelah klik):
- Badge besar POSITIF/NEGATIF dengan warna sesuai
- Confidence: `"Keyakinan model: 87.3%"` — progress bar horizontal merah/hijau
- Aspek terdeteksi: chip-chip kecil per aspek
- Jika negatif + aspek terdeteksi: catatan rekomendasi singkat

Di bawah kolom — Expander "Detail Teknis":
- Teks asli → teks preprocessed
- Dimensi vektor TF-IDF
- Top 10 kata berpengaruh dalam keputusan

### Halaman 5: Laporan

**Statistik keseluruhan bulan ini:**
- Pie chart distribusi status UMKM (Plotly, warna sesuai status)
- Tabel ringkasan per kategori: Kuliner (12 UMKM) → berapa kritis/baik/dll

**Tabel lengkap untuk download:**
Semua UMKM dengan kolom: ID | Nama | Kategori | Lokasi | Skor | Status | Masalah Utama | Rekomendasi 1 | Rekomendasi 2

Tombol download:
```python
csv_data = df_laporan.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    label="Unduh Laporan CSV",
    data=csv_data,
    file_name=f"laporan_umkm_banyumas_{datetime.now().strftime('%Y%m')}.csv",
    mime="text/csv"
)
```

---

## ATURAN TEKNIS WAJIB

**1. Semua @st.cache_resource untuk resource berat:**
```python
@st.cache_resource
def load_models():
    return {
        "vectorizer": joblib.load("vectorizer.pkl"),
        "Naive Bayes": joblib.load("nb_model.pkl"),
        "SVM": joblib.load("svm_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
    }

@st.cache_data
def load_umkm_data():
    with open("umkm_data.json", encoding="utf-8") as f:
        return json.load(f)
```

**2. Cek file exists sebelum load:**
Jika `umkm_data.json` tidak ada, tampilkan instruksi di tengah halaman:
```python
st.info("Data UMKM belum tersedia. Jalankan terlebih dahulu:\n\n`python generate_umkm.py`")
st.stop()
```

**3. Fungsi preprocess di app.py harus IDENTIK dengan train.py:**
Copy paste fungsi `preprocess()` yang sama, load `stopwords_set.pkl`.

**4. Tidak ada file pages/ — gunakan session state untuk navigasi.**

**5. Semua chart Plotly menggunakan template `"plotly_white"` dan warna konsisten dengan design system.**

**6. Komentar kode dalam Bahasa Indonesia.**

**7. Sidebar navigasi harus menyembunyikan dirinya saat di halaman login:**
```python
if not st.session_state.login_ok:
    render_login()
    st.stop()

render_sidebar()  # hanya jika sudah login
render_halaman_aktif()
```

---

## PESAN PENTING UNTUK KAMU (OPUS)

Proyek ini akan di-demo live kepada penguji (dosen). Setiap halaman harus berfungsi penuh — tidak boleh ada placeholder, tombol mati, atau error yang tidak di-handle.

Penguji AKAN melakukan ini saat demo:
1. Login dengan credentials yang diberikan
2. Lihat Overview → tanya "berapa UMKM yang kritis?"
3. Klik filter Kritis di Daftar UMKM → tanya "kenapa ini kritis?"
4. Klik salah satu UMKM → lihat detail aspek dan rekomendasi
5. Pergi ke Analisis Ulasan → input teks baru → lihat prediksi
6. Klik Download di Laporan

Setiap skenario di atas harus berjalan mulus tanpa error.

Tulis ketiga file (`train.py`, `generate_umkm.py`, `app.py`) secara lengkap dan siap dijalankan. Tidak boleh ada bagian yang disingkat dengan `# ... lanjutkan`, `# TODO`, atau placeholder apapun. Kode harus 100% lengkap.
