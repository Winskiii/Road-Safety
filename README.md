# Proyek Machine Learning: Prediksi Tingkat Keparahan Kecelakaan Lalu Lintas

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20Optuna-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

Proyek ini bertujuan untuk membangun model klasifikasi machine learning yang dapat memprediksi tingkat keparahan korban dalam sebuah kecelakaan lalu lintas. Analisis ini mencakup seluruh alur kerja machine learning, mulai dari pembersihan data, analisis data eksploratif (EDA), hingga evaluasi model.

---

## 📑 Daftar Isi
1.  [Deskripsi Proyek & Tujuan](#-deskripsi-proyek--tujuan)
2.  [Dataset](#-dataset)
3.  [Metodologi Analisis](#️-metodologi-analisis)
    - [1. Import Library](#1-import-library-yang-diperlukan)
    - [2. Membaca Dataset](#2-membaca-dataset)
    - [3. Pengecekan Awal Data](#3-pengecekan-awal-data-sanity-check)
    - [4. Analisis Data Eksploratif (EDA)](#4-analisis-data-eksploratif-eda)
    - [5. Penanganan Data Hilang](#5-penanganan-data-hilang-missing-value-treatments)
    - [6. Penanganan Outlier](#6-penanganan-outlier-outliers-treatments)
    - [7. Penanganan Duplikat](#7-penanganan-duplikat--nilai-sampah)
    - [8. Normalisasi Data](#8-normalisasi--penskalaan-data)
    - [9. Encoding Data Kategorikal](#9-encoding-data-kategorikal)
4.  [Pemodelan & Hasil](#-pemodelan--hasil)
5.  [Instalasi & Requirements](#️-instalasi--requirements)
6.  [Cara Menjalankan](#-cara-menjalankan)

---

## 📝 Deskripsi Proyek & Tujuan

**Deskripsi:**
Proyek ini mengolah dataset kecelakaan lalu lintas untuk mengidentifikasi faktor-faktor kunci yang mempengaruhi tingkat keparahan cedera korban.

**Tujuan:**
- Menganalisis pola dan karakteristik dari data kecelakaan.
- Membangun model prediksi dengan akurasi tinggi untuk mengklasifikasikan tingkat keparahan korban ke dalam kategori: **Fatal, Serius, atau Ringan**.

---

## 📚 Dataset
Dataset yang digunakan adalah `train_road_safety.csv`. Dataset ini berisi **46,649 data kecelakaan** dengan 21 fitur awal, termasuk informasi seperti usia korban, kelas korban (pengemudi, penumpang, pejalan kaki), dan detail demografis lainnya.

---

## 🛠️ Metodologi Analisis

Berikut adalah alur kerja yang dilakukan dalam proyek ini.

### 1. Import Library yang Diperlukan
Langkah awal adalah mengimpor semua library yang dibutuhkan untuk analisis dan pemodelan.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import optuna
```
###2. Membaca Dataset
Memuat data dari file .csv ke dalam DataFrame Pandas.
```python
df = pd.read_csv('train_road_safety.csv')
```
###3. Pengecekan Awal Data (Sanity Check)
Memahami struktur data, tipe data, dan mengidentifikasi anomali awal.

Dimensi Data: 46,649 baris dan 21 kolom.
Data Hilang: Teridentifikasi pada beberapa kolom (status, collision_year, dll).
Statistik: Ditemukan nilai konstan pada collision_year dan nilai -1 sebagai placeholder.
###4. Analisis Data Eksploratif (EDA)
Menemukan pola dan wawasan dari data melalui visualisasi.

Distribusi Target: Ditemukan bahwa data tidak seimbang, di mana jumlah korban luka ringan jauh lebih banyak dari korban fatal.
Deteksi Outlier: Boxplot menunjukkan adanya outlier pada fitur numerik.
Analisis Bivariat: Hubungan antar fitur dianalisis untuk menemukan korelasi dan pola yang relevan dengan tingkat keparahan.

###5. Penanganan Data Hilang (Missing Value Treatments)
Penghapusan Kolom: Kolom yang tidak informatif (status, collision_year, lsoa_of_casualty) dihapus.
Imputasi: Nilai yang hilang pada kolom kategorikal diisi menggunakan modus (nilai yang paling sering muncul).

###6. Penanganan Outlier (Outliers Treatments)
Untuk meningkatkan robustisitas model, outlier ditangani dengan metode capping (pembatasan) berdasarkan rentang interkuartil (IQR).

```python
def cap_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Menerapkan capping
    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])
    return df

# Contoh penerapan:
# df_treated = cap_outliers(df.copy(), 'age_of_casualty')
```

###7. Penanganan Duplikat & Nilai Sampah
Duplikat: Dilakukan pengecekan untuk memastikan tidak ada baris data yang identik.
Nilai Sampah: Nilai -1 yang teridentifikasi sebelumnya dipertahankan karena model berbasis pohon dapat menanganinya sebagai kategori terpisah.
```python
# Kode untuk memeriksa duplikat
duplicate_rows = df.duplicated().sum()
print(f"Jumlah baris duplikat ditemukan: {duplicate_rows}")

# Jika ditemukan, hapus dengan:
# df.drop_duplicates(inplace=True)
```

###8. Normalisasi / Penskalaan Data
Fitur numerik diskalakan untuk menyamakan rentang nilainya, yang penting untuk beberapa algoritma. RobustScaler dipilih karena efektif menangani outlier.
```python
from sklearn.preprocessing import RobustScaler

numerical_features = ['collision_index', 'vehicle_reference', 'casualty_reference', 'age_of_casualty']
scaler = RobustScaler()

# df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

###9. Encoding Data Kategorikal
Fitur kategorikal diubah menjadi format numerik dengan pendekatan yang tepat:

OrdinalEncoder: Untuk fitur yang memiliki tingkatan (age_band_of_casualty).
OneHotEncoder: Untuk fitur nominal yang tidak memiliki urutan (casualty_class, sex_of_casualty, dll).
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ordinal_features = ['age_band_of_casualty', 'casualty_imd_decile']
nominal_features = ['casualty_class', 'sex_of_casualty', 'casualty_type'] # dan fitur nominal lainnya

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(), ordinal_features),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), nominal_features)
    ],
    remainder='passthrough'
)
# X_encoded = preprocessor.fit_transform(X)
```

##🤖 Pemodelan & Hasil
Seleksi Model: Beberapa model (KNN, Decision Tree, XGBoost, Gradient Boosting, dll.) dievaluasi menggunakan 5-fold cross-validation. Model berbasis ansambel seperti XGBoost dan Gradient Boosting menunjukkan performa terbaik.
Tuning Hiperparameter: Optuna digunakan untuk mencari kombinasi hiperparameter terbaik untuk model-model teratas.
Model Final dan Hasil: Model Gradient Boosting Classifier yang telah di-tuning memberikan performa terbaik pada data validasi, dengan akurasi ~96.75%. Evaluasi lebih lanjut dengan classification report dan confusion matrix dilakukan untuk memahami performa model pada setiap kelas.
⚙️ Instalasi & Requirements
Untuk menjalankan notebook ini, Anda memerlukan library Python berikut yang dapat diinstal melalui pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna
```
##🚀 Cara Menjalankan
Lakukan clone pada repositori ini.
Instal semua library yang dibutuhkan sesuai dengan bagian Requirements.
Letakkan file train_road_safety.csv di direktori yang sama dengan notebook.
Jalankan notebook FPML.ipynb melalui Jupyter Notebook, JupyterLab, atau Google Colab.
