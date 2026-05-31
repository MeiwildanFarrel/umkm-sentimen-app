# -*- coding: utf-8 -*-
"""
=============================================================================
  SIMANTAP UMKM - Dinas Koperasi dan UKM Kabupaten Banyumas
  ---------------------------------------------------------------------------
  train.py : Script pelatihan model klasifikasi sentimen ulasan produk.

  Dijalankan SEKALI. Menghasilkan:
    - vectorizer.pkl          (TF-IDF vectorizer)
    - nb_model.pkl            (Multinomial Naive Bayes)
    - svm_model.pkl           (SVM Linear, terkalibrasi -> punya predict_proba)
    - rf_model.pkl            (Random Forest)
    - stopwords_set.pkl       (set stopword, agar identik di seluruh proyek)
    - evaluation_results.json (metrik evaluasi untuk dashboard)

  Pipeline : Load CSV -> Preprocessing -> TF-IDF -> Training 3 model -> Evaluasi
=============================================================================
"""

import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH   = "PRDECT-ID Dataset.csv"
VECTORIZER_PKL = "vectorizer.pkl"
NB_MODEL_PKL   = "nb_model.pkl"
SVM_MODEL_PKL  = "svm_model.pkl"
RF_MODEL_PKL   = "rf_model.pkl"
STOPWORDS_PKL  = "stopwords_set.pkl"
EVAL_JSON      = "evaluation_results.json"
RANDOM_STATE   = 42

# ─────────────────────────────────────────────────────────────────────────────
# 2. DAFTAR STOPWORD BAHASA INDONESIA
# ─────────────────────────────────────────────────────────────────────────────
# Basis stopword diambil dari NLTK (corpus 'indonesian'). Jika NLTK gagal
# diunduh (mis. tanpa internet), digunakan daftar cadangan manual.
def _muat_stopwords_dasar():
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("indonesian"))
    except Exception:
        # Daftar cadangan bila NLTK tidak tersedia
        return {
            "yang", "dan", "di", "ke", "dari", "ini", "itu", "ada", "adalah",
            "akan", "dengan", "untuk", "pada", "dalam", "oleh", "atau", "karena",
            "sebagai", "agar", "jika", "kalau", "ketika", "saat", "setelah",
            "sebelum", "hingga", "sampai", "antara", "tentang", "secara",
            "bahwa", "merupakan", "menjadi", "dapat", "harus", "masih",
            "telah", "sudah", "saya", "anda", "kami", "kita", "kamu", "dia",
            "mereka", "juga", "saja", "hanya", "namun", "tetapi", "tapi",
            "sehingga", "maka", "lalu", "kemudian", "serta", "pula", "para",
            "akhirnya", "selama", "sambil", "kepada", "daripada", "menurut",
            "tersebut", "begitu", "demikian", "adapun", "sangat", "sekali",
        }


STOPWORDS_ID = _muat_stopwords_dasar()

# Tambahan kata yang tidak relevan untuk sentimen produk
STOPWORDS_ID.update([
    "yg", "dgn", "utk", "jg", "sy", "aja", "udah", "udh", "gak", "ga",
    "gk", "bgt", "bngt", "banget", "sama", "jadi", "bisa", "ada", "ini",
    "itu", "yang", "dan", "di", "ke", "dari", "nya", "ku", "mu", "sih",
    "dong", "deh", "loh", "lho", "kok", "kan", "ya", "dg", "br", "tp",
    "tpi", "krna", "krn", "karna", "makasih", "terimakasih", "terima",
    "kasih", "hatur", "nuhun", "maturnuwun", "matur",
])

# CATATAN PENTING — kata negasi DIPERTAHANKAN (tidak ikut dibuang):
#   Pada analisis sentimen, kata negasi menentukan makna kalimat.
#   "tidak bagus" != "bagus". Bila negasi dibuang, prediksi bisa terbalik
#   dan menyebabkan kegagalan saat demo (mis. ulasan "produk tidak bagus"
#   justru terprediksi POSITIF). Karena itu kata negasi di bawah ini
#   dikeluarkan kembali dari daftar stopword.
KATA_NEGASI = {
    "tidak", "tidaklah", "tidakkah", "tak", "takkan", "tiada", "bukan",
    "bukankah", "bukanlah", "bukannya", "jangan", "janganlah", "belum",
    "belumlah", "kurang", "tanpa", "gak", "ga", "gk", "nggak", "enggak",
    "ngga", "kagak",
}
STOPWORDS_ID = STOPWORDS_ID - KATA_NEGASI

# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def preprocess(text: str) -> str:
    """Bersihkan -> tokenisasi -> buang stopword -> stemming. Hasil: string."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD & FILTER DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 66)
print("  SIMANTAP UMKM - TRAINING MODEL KLASIFIKASI SENTIMEN")
print("=" * 66)
print()
print("[1/6] Memuat dataset '" + DATASET_PATH + "' ...")

df = pd.read_csv(DATASET_PATH)
df = df[["Category", "Customer Review", "Sentiment"]].dropna()
df.columns = ["category", "review", "label"]
df = df[df["label"].isin(["Positive", "Negative"])].reset_index(drop=True)

pos_count = int((df["label"] == "Positive").sum())
neg_count = int((df["label"] == "Negative").sum())
print("      [OK] Total data terpakai : " + str(len(df)))
print("      -> Positive : " + str(pos_count))
print("      -> Negative : " + str(neg_count))
print("      -> Jumlah stopword aktif : " + str(len(STOPWORDS_ID)))

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print()
print("[2/6] Preprocessing teks (cleaning + stopword + stemming) ...")
tqdm.pandas(desc="      -> Memproses ulasan")
df["review_clean"] = df["review"].progress_apply(preprocess)

# Buang baris yang menjadi kosong setelah preprocessing
df = df[df["review_clean"].str.strip() != ""].reset_index(drop=True)
print("      [OK] Preprocessing selesai. Contoh:")
for _, row in df[["review", "review_clean"]].head(2).iterrows():
    print("      Asli  : " + str(row["review"])[:72])
    print("      Bersih: " + str(row["review_clean"])[:72])

# ─────────────────────────────────────────────────────────────────────────────
# 6. TF-IDF & SPLIT 80/20
# ─────────────────────────────────────────────────────────────────────────────
print()
print("[3/6] Ekstraksi fitur TF-IDF & pembagian data ...")

X = df["review_clean"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
n_features = int(X_train_tfidf.shape[1])

print("      -> Data latih : " + str(len(X_train)) + " sampel")
print("      -> Data uji   : " + str(len(X_test)) + " sampel")
print("      -> Fitur TF-IDF: " + str(n_features))

# ─────────────────────────────────────────────────────────────────────────────
# 7. PELATIHAN, EVALUASI, & CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
LABELS = ["Negative", "Positive"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluasi_model(nama, model):
    """Latih, evaluasi pada data uji, dan jalankan 5-fold cross-validation."""
    print()
    print("  >> " + nama)
    model.fit(X_train_tfidf, y_train)
    pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, pred)
    f1w = f1_score(y_test, pred, average="weighted")
    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=LABELS)  # [[TN,FP],[FN,TP]]

    cv = cross_val_score(
        model, X_train_tfidf, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1
    )

    print("     Accuracy       : " + f"{acc * 100:.2f}%")
    print("     F1-Score (wtd) : " + f"{f1w * 100:.2f}%")
    print("     CV F1 (5-fold) : "
          + f"{cv.mean() * 100:.2f}% (+/- {cv.std() * 100:.2f}%)")

    return {
        "accuracy": round(float(acc), 4),
        "f1_weighted": round(float(f1w), 4),
        "f1_positive": round(float(rep["Positive"]["f1-score"]), 4),
        "f1_negative": round(float(rep["Negative"]["f1-score"]), 4),
        "precision_positive": round(float(rep["Positive"]["precision"]), 4),
        "recall_positive": round(float(rep["Positive"]["recall"]), 4),
        "precision_negative": round(float(rep["Negative"]["precision"]), 4),
        "recall_negative": round(float(rep["Negative"]["recall"]), 4),
        "confusion_matrix": cm.tolist(),
        "cv_f1_mean": round(float(cv.mean()), 4),
        "cv_f1_std": round(float(cv.std()), 4),
    }


print()
print("[4/6] Melatih & mengevaluasi 3 model ...")
print("-" * 66)

evaluation = {}

# Model 1: Multinomial Naive Bayes
nb_model = MultinomialNB(alpha=0.5)
evaluation["Naive Bayes"] = evaluasi_model("Model 1: Multinomial Naive Bayes", nb_model)

# Model 2: SVM Linear -> dibungkus CalibratedClassifierCV agar punya predict_proba
svm_base = LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE)
svm_model = CalibratedClassifierCV(svm_base, cv=3)
evaluation["SVM"] = evaluasi_model("Model 2: SVM Linear (Calibrated)", svm_model)

# Model 3: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
)
evaluation["Random Forest"] = evaluasi_model("Model 3: Random Forest", rf_model)

# ─────────────────────────────────────────────────────────────────────────────
# 8. RINGKASAN & MODEL TERBAIK
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 66)
print("  RINGKASAN PERBANDINGAN MODEL")
print("=" * 66)
print(f"{'Model':<16}{'Accuracy':>11}{'F1-Score':>11}{'CV F1 (mean+/-std)':>24}")
print("-" * 66)
for nama, m in evaluation.items():
    print(
        f"{nama:<16}"
        f"{m['accuracy'] * 100:>10.2f}%"
        f"{m['f1_weighted'] * 100:>10.2f}%"
        f"{m['cv_f1_mean'] * 100:>15.2f}% +/- {m['cv_f1_std'] * 100:.2f}%"
    )
print("=" * 66)

best_model = max(evaluation, key=lambda k: evaluation[k]["f1_weighted"])
print()
print("  [BEST] Model terbaik (F1-Score weighted): " + best_model)

evaluation["best_model"] = best_model
evaluation["dataset_info"] = {
    "total_samples": int(len(df)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "positive_count": pos_count,
    "negative_count": neg_count,
    "n_features": n_features,
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. SIMPAN SEMUA OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print()
print("[5/6] Menyimpan model, vectorizer, & stopword ...")
joblib.dump(vectorizer, VECTORIZER_PKL)
joblib.dump(nb_model, NB_MODEL_PKL)
joblib.dump(svm_model, SVM_MODEL_PKL)
joblib.dump(rf_model, RF_MODEL_PKL)
joblib.dump(STOPWORDS_ID, STOPWORDS_PKL)
print("      [OK] vectorizer.pkl, nb_model.pkl, svm_model.pkl,")
print("           rf_model.pkl, stopwords_set.pkl tersimpan.")

print()
print("[6/6] Menyimpan hasil evaluasi ...")
with open(EVAL_JSON, "w", encoding="utf-8") as f:
    json.dump(evaluation, f, indent=2, ensure_ascii=False)
print("      [OK] evaluation_results.json tersimpan.")

print()
print("=" * 66)
print("  [SELESAI] Pelatihan selesai.")
print("  Langkah berikutnya : python generate_umkm.py")
print("=" * 66)
