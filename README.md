Proyek Machine Learning: Prediksi Tingkat Keparahan Kecelakaan Lalu Lintas
📝 Deskripsi Proyek
Proyek ini bertujuan untuk membangun model klasifikasi machine learning yang dapat memprediksi tingkat keparahan korban dalam sebuah kecelakaan lalu lintas (casualty_severity). Dataset yang digunakan berisi berbagai fitur terkait detail kecelakaan, kendaraan, dan data demografis korban. Proses analisis meliputi pembersihan data, analisis data eksploratif (EDA), rekayasa fitur, hingga perbandingan dan tuning beberapa model klasifikasi untuk mendapatkan performa terbaik.

🎯 Tujuan
Tujuan utama dari proyek ini adalah:

Menganalisis faktor-faktor yang paling berpengaruh terhadap tingkat keparahan kecelakaan.
Membangun model prediksi dengan akurasi tinggi untuk mengklasifikasikan tingkat keparahan korban ke dalam kategori: Fatal, Serius, atau Ringan.
📁 Dataset
Dataset yang digunakan adalah train_road_safety.csv, yang berisi 46.649 data kecelakaan dengan 21 fitur awal, termasuk informasi seperti usia korban, kelas korban (pengemudi, penumpang, pejalan kaki), tipe kendaraan, dan detail demografis lainnya.

🛠️ Metodologi
Proyek ini mengikuti alur kerja machine learning yang terstruktur sebagai berikut:

1. Import Library yang Diperlukan
Langkah pertama adalah mengimpor semua library Python yang dibutuhkan untuk analisis dan pemodelan.

Python

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
2. Membaca Dataset
Data dimuat dari file train_road_safety.csv ke dalam DataFrame Pandas untuk dianalisis lebih lanjut.

Python

df = pd.read_csv('train_road_safety.csv')
3. Pengecekan Awal Data (Sanity Check)
Pemeriksaan awal dilakukan untuk memahami struktur data, tipe data, dan mengidentifikasi adanya data yang hilang.

Dimensi Data: 46,649 baris dan 21 kolom.
Tipe Data: Campuran int64, float64, dan object.
Data Hilang: Ditemukan data hilang pada beberapa kolom seperti status, collision_year, dan sex_of_casualty.
Statistik Deskriptif: Ditemukan bahwa kolom collision_year memiliki nilai konstan dan beberapa fitur numerik memiliki nilai -1 yang kemungkinan merupakan placeholder.
4. Analisis Data Eksploratif (EDA)
EDA dilakukan untuk menemukan pola, anomali, dan wawasan dari data.

Distribusi Usia Korban: Histogram menunjukkan bahwa korban mayoritas berada di rentang usia dewasa muda (20-40 tahun).
Deteksi Outlier: Boxplot menunjukkan adanya nilai pencilan (outlier) pada beberapa fitur numerik seperti age_of_casualty dan casualty_reference.
Analisis Data Tidak Seimbang: Analisis pada variabel target (casualty_severity) menunjukkan bahwa data tidak seimbang, di mana jumlah korban dengan luka ringan jauh lebih banyak daripada korban fatal. EDA lebih lanjut membandingkan distribusi fitur antar kelas target untuk menemukan pola unik pada kelas minoritas (fatal).
5. Penanganan Data Hilang (Missing Value Treatments)
Penghapusan Kolom: Kolom yang tidak relevan atau memiliki terlalu banyak data hilang seperti status, collision_year, dan lsoa_of_casualty dihapus.
Imputasi: Data yang hilang pada kolom kategorikal (sex_of_casualty, casualty_home_area_type, dll.) diisi menggunakan nilai yang paling sering muncul (modus).
6. Penanganan Outlier (Outliers Treatments)
(Ini adalah langkah tambahan yang direkomendasikan)

Setelah outlier terdeteksi di tahap EDA, salah satu cara menanganinya adalah dengan metode capping menggunakan rentang interkuartil (IQR). Ini akan membatasi nilai ekstrem ke batas atas dan bawah yang wajar.

Python

# Kode untuk menangani outlier dengan metode IQR Capping
def cap_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])
    return df

# Contoh penerapan pada kolom usia
# df_treated = cap_outliers(df.copy(), 'age_of_casualty')

# print("Perbandingan statistik 'age_of_casualty' sebelum dan sesudah capping:")
# print("Sebelum:\n", df['age_of_casualty'].describe())
# print("\nSesudah:\n", df_treated['age_of_casualty'].describe())
7. Penanganan Duplikat & Nilai Sampah
(Ini adalah langkah tambahan untuk memastikan kebersihan data)

Pengecekan Duplikat: Memeriksa dan menghapus baris data yang identik.
Nilai Sampah (Garbage Values): Pengecekan ini bersifat domain-spesifik. Dalam kasus ini, nilai -1 telah diidentifikasi. Karena model berbasis pohon dapat menangani nilai ini sebagai kategori terpisah, nilai tersebut dibiarkan. Namun, untuk model lain, nilai ini mungkin perlu diimputasi.
Python

# Kode untuk memeriksa dan menghapus duplikat
duplicate_rows = df.duplicated().sum()
print(f"Jumlah baris duplikat: {duplicate_rows}")

# Jika ada, hapus duplikat
# df.drop_duplicates(inplace=True)
8. Normalisasi / Penskalaan (Normalization)
Fitur numerik perlu diskalakan agar memiliki rentang nilai yang seragam. Ini penting untuk performa model yang sensitif terhadap jarak. RobustScaler direkomendasikan karena ketahanannya terhadap outlier.

Python

# Kode untuk menerapkan penskalaan pada fitur numerik
from sklearn.preprocessing import RobustScaler

# Kolom numerik yang perlu di-scale (setelah encoding)
numerical_features = ['collision_index', 'vehicle_reference', 'casualty_reference', 'age_of_casualty', 'enhanced_casualty_severity']

scaler = RobustScaler()
# df[numerical_features] = scaler.fit_transform(df[numerical_features])
9. Encoding Data Kategorikal
Fitur kategorikal diubah menjadi format numerik. Pendekatan terbaik adalah membedakan antara fitur ordinal (memiliki urutan) dan nominal (tidak memiliki urutan).

OrdinalEncoder: Digunakan untuk fitur seperti age_band_of_casualty.
OneHotEncoder: Digunakan untuk fitur seperti casualty_class dan sex_of_casualty untuk menghindari asumsi urutan yang salah.
Python

# Kode untuk encoding yang tepat menggunakan ColumnTransformer
# (Kode lengkap telah disediakan di jawaban sebelumnya, ini adalah ringkasannya)

# ordinal_features = ['age_band_of_casualty', 'casualty_imd_decile']
# nominal_features = ['casualty_class', 'sex_of_casualty', ...]

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('ord', OrdinalEncoder(), ordinal_features),
#         ('ohe', OneHotEncoder(handle_unknown='ignore'), nominal_features)
#     ],
#     remainder='passthrough'
# )

# X_encoded = preprocessor.fit_transform(X)
🤖 Pemodelan & Hasil
Beberapa model klasifikasi dievaluasi menggunakan 5-fold stratified cross-validation. Tiga model teratas (XGBoost, Gradient Boosting, dan Random Forest) kemudian digabungkan dalam sebuah Voting Classifier dan di-tuning menggunakan Optuna untuk mencari hiperparameter terbaik.

Model Terbaik: Gradient Boosting Classifier menunjukkan performa sedikit lebih unggul pada data validasi.
Akurasi Validasi Final: Model terbaik mencapai akurasi ~96.75% pada data uji yang belum pernah dilihat sebelumnya.
⚙️ Requirements & Instalasi
Untuk menjalankan notebook ini, Anda memerlukan library Python berikut:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
Anda dapat menginstalnya menggunakan pip:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna

🚀 Cara Menjalankan
Pastikan semua library di atas sudah terinstal.
Letakkan file dataset train_road_safety.csv di direktori yang sama dengan notebook.
Jalankan file FPML.ipynb menggunakan Jupyter Notebook atau JupyterLab.
