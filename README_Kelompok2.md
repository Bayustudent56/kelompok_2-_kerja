# 📊 Analisis Data X (Twitter): Dunia Kerja, Gaji UMR, dan Fresh Graduate

Proyek **Tugas Kelompok 2** yang menganalisis percakapan di X (Twitter) seputar dunia kerja di Indonesia, khususnya topik **gaji UMR, lowongan kerja, dan fresh graduate**. 

Proyek ini terbagi menjadi dua lingkungan eksekusi utama:
1. **Google Colab Notebook (`Tugas_Kelompok_2.ipynb`)**: Digunakan untuk proses *crawling* data dari X, penggabungan data (*merging*), pembersihan teks (*cleaning*), *clustering* topik (K-Means), dan klasifikasi sentimen menggunakan model *pre-trained* IndoBERT.
2. **Dashboard Web Interaktif (`app.py`)**: Aplikasi berbasis **Streamlit** yang dijalankan secara lokal untuk menampilkan visualisasi data, evaluasi sentimen manual, hasil *clustering*, serta pemodelan dan kalkulator **Regresi Linear** untuk memprediksi jumlah *likes*.

> ⚠️ **Catatan Penting:** File notebook (`Tugas_Kelompok_2.ipynb`) berisi *magic command* khusus Linux/Colab (`!pip`, `!sudo apt-get`, `!npx`, dsb.) sehingga **wajib dijalankan di Google Colab**, sedangkan file `app.py` dijalankan menggunakan perintah `streamlit run app.py`.

---

## 📑 Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Alur Pengerjaan](#alur-pengerjaan)
- [Struktur File Proyek](#struktur-file-proyek)
- [Teknologi dan Pustaka yang Digunakan](#teknologi-dan-pustaka-yang-digunakan)
- [Panduan Instalasi dan Menjalankan Proyek](#panduan-instalasi-dan-menjalankan-proyek)
  - [Bagian 1: Menjalankan Pengolahan Data di Google Colab](#bagian-1-menjalankan-pengolahan-data-di-google-colab)
  - [Bagian 2: Menjalankan Dashboard Web Streamlit](#bagian-2-menjalankan-dashboard-web-streamlit)
- [Struktur File Output](#struktur-file-output)
- [Detail Tiap Tahap Analisis](#detail-tiap-tahap-analisis)
- [Penyelesaian Masalah (Troubleshooting)](#penyelesaian-masalah-troubleshooting)
- [Keterbatasan Proyek](#keterbatasan-proyek)
- [Anggota Kelompok](#anggota-kelompok)

---

## 🔍 Gambaran Umum

| Item | Keterangan |
|---|---|
| **Sumber Data** | Postingan di X (Twitter) |
| **Kata Kunci Pencarian** | `gaji UMR OR lowongan kerja OR fresh graduate` |
| **Rentang Waktu** | 1 Januari 2026 sampai 10 Oktober 2026 |
| **Bahasa Data** | Indonesia (`in` / `id`) |
| **Alat Crawling** | [`tweet-harvest` (v2.7.1)](https://github.com/helmisatria/tweet-harvest) |
| **Platform Pengolahan** | Google Colab (Python 3 & Node.js 20) |
| **Platform Visualisasi** | Streamlit Web Dashboard (`app.py`) |

---

## 🔄 Alur Pengerjaan

```text
[Tahap 1: Google Colab (Tugas_Kelompok_2.ipynb)]
Crawling Data  ──►  Merging  ──►  Cleaning Data  ──►  Clustering (K-Means)  ──►  Analisis Sentimen (IndoBERT)
                                                                                          │
                                                     Unduh: hasil_sentimen_lengkap.csv ◄──┘
                                                                                          │
[Tahap 2: Dashboard Streamlit (app.py)]                                                   ▼
Tab Regresi Linear  ◄──  Tab Clustering  ◄──  Tab Sentimen & Evaluasi  ◄──  Tab Data (Unggah CSV)
```

1. **Crawling Data**: Mengambil postingan X dengan `tweet-harvest` (2 sesi crawling dengan target *limit* 1000 dan 500 postingan).
2. **Merging**: Menggabungkan file hasil crawling (`kerja.csv` dan `kerja2.csv`) serta menghapus duplikasi berdasarkan `id_str` / `full_text`.
3. **Cleaning Data**: Membersihkan teks (`Cleaned_Text`), menormalkan *timestamp* dan metrik interaksi, memfilter bahasa Indonesia, serta menyaring relevansi topik dunia kerja.
4. **Clustering Data**: Mengelompokkan teks ke dalam 3 topik utama menggunakan ekstraksi fitur TF-IDF, algoritma K-Means, evaluasi *Silhouette Score*, dan reduksi dimensi PCA.
5. **Analisis Sentimen**: Mengklasifikasikan postingan ke dalam sentimen **Positif**, **Netral**, atau **Negatif** menggunakan model IndoBERT, lalu mengekspor hasilnya ke `hasil_sentimen_lengkap.csv`.
6. **Dashboard & Regresi Linear**: Mengunggah CSV hasil Colab ke aplikasi Streamlit untuk memvisualisasikan distribusi data, menguji akurasi sentimen dengan label manual, melihat detail cluster, dan memprediksi `like_count` berdasarkan `reply_count` dan `retweet_count`.

---

## 📂 Struktur File Proyek

```text
├── Tugas_Kelompok_2.ipynb          # Notebook Google Colab untuk crawling hingga analisis sentimen
├── app.py                          # Aplikasi utama dashboard web interaktif berbasis Streamlit
├── requirements.txt                # Daftar paket Python untuk menjalankan aplikasi Streamlit
├── dataset/                        # (Opsional) Folder lokal untuk file evaluasi sentimen manual
│   └── hasil_komparasi_sentimen.xlsx
└── README.md                       # Dokumentasi proyek
```

---

## 🛠️ Teknologi dan Pustaka yang Digunakan

### Lingkungan Google Colab (`Tugas_Kelompok_2.ipynb`)
- **Node.js 20** & **`tweet-harvest`**: Pengambilan data cuitan X berbasis Chromium.
- **pandas** & **NumPy**: Pengolahan tabel DataFrame dan komputasi numerik.
- **NLTK**: Penghapusan *stopwords* bahasa Indonesia dan bahasa Inggris.
- **Hugging Face `transformers`** & **PyTorch (`torch`)**: Menjalankan model IndoBERT (`mdhugol/indonesia-bert-sentiment-classification`).
- **scikit-learn**: Pemodelan TF-IDF, K-Means Clustering, dan PCA.
- **Matplotlib** & **openpyxl**: Visualisasi grafik dan pengelolaan file Excel.

### Lingkungan Lokal / Dashboard (`requirements.txt`)
- `streamlit`: Framework antarmuka aplikasi dashboard web interaktif.
- `pandas==2.2.3`: Pengolahan data CSV dan Excel.
- `numpy==2.1.3`: Operasi numerik dan perhitungan array.
- `scikit-learn==1.6.1`: Algoritma TF-IDF, K-Means, dan Regresi Linear.
- `matplotlib==3.10.0` & `seaborn`: Pembuatan grafik distribusi sentimen dan plot regresi.
- `openpyxl==3.1.5`: Membaca file Excel (`.xlsx`) pada fitur evaluasi sentimen manual.

---

## 🚀 Panduan Instalasi dan Menjalankan Proyek

### Bagian 1: Menjalankan Pengolahan Data di Google Colab

#### A. Prasyarat Colab
1. **Akun Google**: Untuk mengakses [Google Colab](https://colab.research.google.com).
2. **Akun X (Twitter)**: Untuk mengambil `auth_token` sebagai syarat crawling.
3. **Akselerator GPU (Disarankan)**: Agar proses klasifikasi IndoBERT berjalan cepat, ubah runtime melalui menu **Runtime > Change runtime type > Hardware accelerator: T4 GPU**.

#### B. Cara Mendapatkan Auth Token X (Twitter)
1. Buka dan login ke [x.com](https://x.com) melalui browser (Chrome/Firefox/Edge).
2. Tekan tombol **F12** (atau klik kanan > **Inspect**) untuk membuka *Developer Tools*.
3. Pilih tab **Application** (di Chrome) atau **Storage** (di Firefox).
4. Di menu sebelah kiri, buka **Cookies > `https://x.com`**.
5. Cari cookie bernama **`auth_token`**, lalu salin (*copy*) nilai (*Value*) dari cookie tersebut.
6. Masukkan nilai tersebut ke variabel di dalam notebook:
   ```python
   twitter_auth_token = "ISI_TOKEN_KAMU_DI_SINI"
   ```
   > ⚠️ **Keamanan:** Jangan pernah membagikan atau mengunggah `auth_token` asli ke GitHub karena token tersebut memberikan akses langsung ke akun X kamu.

#### C. Urutan Eksekusi Sel di Google Colab
Buka file `Tugas_Kelompok_2.ipynb` di Google Colab, lalu jalankan sel secara berurutan dari atas ke bawah:

| Tahap | Nama Bagian di Notebook | Deskripsi Proses | Output yang Dihasilkan |
|---|---|---|---|
| **1** | `# Crawling Data` | Menginstal `pandas`, Node.js 20, dependensi sistem Linux (`libgtk-3-0`, `libnss3`, dll.), menyimpan `twitter_auth_token`, dan menjalankan 2 sel crawling `tweet-harvest`. | `/content/tweets-data/kerja.csv` (512 baris) & `kerja2.csv` (394 baris) |
| **2** | `# Merging` | Membaca semua file CSV di `/content/tweets-data/`, menggabungkannya dengan `pd.concat`, dan menghapus duplikat (`id_str`/`full_text`). | `/content/merge_kerja.csv` (650 baris) |
| **3** | `# Cleaning data` | Mengunduh *stopwords* NLTK, membersihkan teks (`comprehensive_cleansing`), menormalkan metrik & tanggal, memfilter bahasa (`in`/`id`), dan menyaring relevansi kata kunci dunia kerja. | `/content/cleaning_kerja_relevan.csv` (516 baris) |
| **4** | `# Clustering Data` | Membaca `cleaning_kerja_relevan.csv`, menyusun *stopwords* khusus (mempertahankan kata negasi), melakukan ekstraksi TF-IDF, melatih K-Means (3 cluster), menghitung *Silhouette Score*, dan membuat plot PCA. | `/content/kerja_clustering.csv`, tabel ringkasan cluster, & plot PCA |
| **5** | `# Analisis Sentimen Postingan` | Menginstal `transformers` & `torch`. Saat tombol **Choose Files** muncul, unggah file `cleaning_kerja_relevan.csv` (unduh terlebih dahulu dari panel **Files** di sebelah kiri Colab ke komputermu, lalu unggah kembali melalui tombol tersebut). Sel selanjutnya akan menjalankan model IndoBERT, menampilkan grafik distribusi, dan mengunduh hasil akhirnya secara otomatis. | `hasil_sentimen_lengkap.csv` (Terunduh otomatis ke komputer) |

---

### Bagian 2: Menjalankan Dashboard Web Streamlit

Setelah mendapatkan file **`hasil_sentimen_lengkap.csv`** dari Google Colab, kamu dapat menjalankan aplikasi web interaktif di komputer lokal.

#### A. Instalasi Dependensi Lokal
1. Pastikan **Python** (versi 3.10 ke atas) sudah terpasang di komputermu.
2. Buka terminal (*Command Prompt*, *PowerShell*, atau *Terminal*) dan arahkan ke direktori proyek:
   ```bash
   cd path/ke/folder/proyek
   ```
3. *(Opsional)* Buat dan aktifkan *virtual environment*:
   ```bash
   python -m venv venv
   # Untuk Windows:
   venv\Scripts\activate
   # Untuk macOS/Linux:
   source venv/bin/activate
   ```
4. Instal seluruh pustaka yang tercantum di dalam `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

#### B. Menjalankan Aplikasi `app.py`
Jalankan perintah berikut di terminal:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

#### C. Panduan Penggunaan Fitur di Dashboard Streamlit
1. **Unggah Dataset Utama (Sidebar Kiri)**:
   - Klik tombol **Browse files** pada menu **📂 Unggah Data** di panel sebelah kiri.
   - Pilih file **`hasil_sentimen_lengkap.csv`** yang telah diunduh dari Google Colab.
2. **Tab 1 — `📁 Data`**:
   - Menampilkan metrik total postingan, jumlah kolom, dan tabel status kelengkapan 10 kolom standar Colab (`id`, `created_at`, `text`, `Cleaned_Text`, `like_count`, `retweet_count`, `reply_count`, `is_relevant`, `sentimen_sistem`, `skor`).
   - Menampilkan pratinjau 10 baris pertama dataset dan informasi jumlah data kosong/tidak valid.
3. **Tab 2 — `😃 Sentimen`**:
   - Menampilkan tabel ringkasan kategori sentimen beserta persentasenya dan grafik batang distribusi sentimen.
   - Menampilkan 20 baris pertama hasil klasifikasi beserta nilai `skor` kepercayaan model.
   - **🔍 Ringkasan Pemeriksaan Manual**: Unggah file komparasi sentimen manual (`.xlsx`, `.xls`, atau `.csv`) yang memiliki kolom `id`, `text`, `label_manual`, `sentimen_sistem`, dan `skor` (atau simpan di dalam folder lokal `dataset/hasil_komparasi_sentimen.xlsx`). Sistem akan otomatis menampilkan status kecocokan (`✅ Sesuai` / `❌ Tidak Sesuai`) dan persentase akurasinya.
4. **Tab 3 — `🏷️ Clustering`**:
   - Menampilkan metrik ukuran masing-masing cluster (Cluster 0, 1, dan 2).
   - Menampilkan rincian tiap cluster yang mencakup: tema topik, persentase jumlah postingan, 5 kata dominan (TF-IDF tertinggi), dan 3 contoh postingan asli, disertai analisis tumpang tindih (*overlap*) kata.
5. **Tab 4 — `📈 Regresi`**:
   - Menampilkan hasil pelatihan model Regresi Linear dengan metrik **MAE (*Mean Absolute Error*)**, **R² Score**, **Koefisien Reply**, dan **Koefisien Retweet**, serta grafik *scatter plot* **Likes Aktual vs Prediksi**.
   - **🧮 Formulir Estimasi Likes**: Masukkan nilai **Jumlah Reply (Balasan)** dan **Jumlah Retweet** pada panel formulir, lalu klik **Estimasi Likes** untuk mendapatkan estimasi jumlah *likes* secara langsung.

---

## 📊 Struktur File Output

| Nama File | Tahap Pembuatan | Keterangan |
|---|---|---|
| `kerja.csv` | Crawling 1 (Colab) | Hasil crawling pertama (target limit 1000, tersimpan 512 baris di `/content/tweets-data/`) |
| `kerja2.csv` | Crawling 2 (Colab) | Hasil crawling kedua (target limit 500, tersimpan 394 baris di `/content/tweets-data/`) |
| `merge_kerja.csv` | Merging (Colab) | Data gabungan hasil crawling setelah penghapusan duplikat (`650 baris`) |
| `cleaning_kerja_relevan.csv` | Cleaning (Colab) | Data bersih berbahasa Indonesia dan relevan dengan topik dunia kerja (`516 baris`) |
| `kerja_clustering.csv` | Clustering (Colab) | Dataset bersih yang telah dilengkapi kolom `clean_text` dan label `cluster` |
| `hasil_sentimen_lengkap.csv` | Analisis Sentimen (Colab) | **Dataset utama untuk diunggah ke `app.py`**, berisi data bersih ditambah kolom `id`, `sentimen_sistem`, dan `skor` |

---

## 🔬 Detail Tiap Tahap Analisis

### 1. Crawling & Merging Data
- Menggunakan `npx --yes tweet-harvest` dengan kata kunci `'gaji UMR OR lowongan kerja OR fresh graduate since:2026-01-01 until:2026-10-10'`.
- Seluruh file CSV di dalam folder `/content/tweets-data/` digabungkan, diberi penanda kolom `Nama_Keyword` berdasarkan nama file asal, dan dibersihkan dari duplikasi berdasarkan kolom `id_str` atau `full_text`.

### 2. Cleaning Data & Filter Relevansi
Fungsi `comprehensive_cleansing` membersihkan teks melalui tahapan:
- *Case folding* (mengubah seluruh huruf menjadi huruf kecil).
- Menghapus tautan URL (`http`, `https`, `www`, `x.com`) dan entitas HTML (`&amp;`).
- Menghapus *mention* (`@username`), *hashtag* (`#topik`), tanda baca/simbol, dan angka.
- Menghapus *stopwords* bahasa Indonesia dari pustaka NLTK dan merapikan spasi berlebih.

Tahap normalisasi dan penyaringan lanjutan:
- Mengonversi kolom `created_at` menjadi tipe data *datetime* berstandar UTC.
- Mengisi nilai kosong (`NaN`) pada kolom metrik (`favorite_count`, `quote_count`, `reply_count`, `retweet_count`) dengan angka `0` dan mengubahnya ke tipe *integer*.
- Menghapus duplikat berdasarkan `Cleaned_Text` dan memfilter hanya cuitan berbahasa Indonesia (`lang` bernilai `in` atau `id`).
- **Filter Relevansi Topik**: Menyaring hanya postingan yang mengandung minimal satu dari daftar kata kunci dunia kerja (`gaji`, `umr`, `umk`, `upah`, `fresh graduate`, `lowongan`, `loker`, `cari kerja`, `mencari kerja`, `kerja`, `karyawan`, `pegawai`, `boss`, `atasan`, `kantor`, `resign`, `layoff`, `phk`, `overwork`, `burnout`, `work life balance`, `freelance`, `magang`, `internship`, `cv`, `interview`, `pengangguran`, `biaya hidup`, `hrd`, `rekrutmen`).
- Mengubah nama kolom `full_text` menjadi `text` dan `favorite_count` menjadi `like_count`.

### 3. Clustering Data Teks (TF-IDF & K-Means)
- **Daftar Stopwords Khusus**: Menggabungkan *stopwords* NLTK bahasa Indonesia dan Inggris, **mempertahankan kata negasi** (`tidak`, `bukan`, `belum`, `jangan`, `tak`) agar konteks kalimat tidak berubah, serta menambahkan *stopwords* bahasa gaul/informal (`ya`, `yg`, `ga`, `yuk`, `dah`, `ngga`, `engga`, `ygy`, `oot`). Total terdapat 959 *stopwords*.
- **Ekstraksi Fitur**: Menggunakan `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` yang menangkap *unigram* dan *bigram*.
- **Pemodelan**: Menggunakan `KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=42)`.
- **Evaluasi & Visualisasi**: Menghitung *Silhouette Score* dengan metrik jarak *cosine* (diperoleh skor `0.0155` pada Colab) dan memetakan titik data beserta posisi *centroid* ke dalam grafik 2 dimensi menggunakan `PCA(n_components=2, random_state=0)`.
- **Hasil Pengelompokan Tema Cluster (di Colab)**:
  - **Cluster 0 (365 postingan)**: *Tantangan Fresh Graduate dan Ekspektasi Gaji* (Kata dominan: `gaji, umr, kerja, gaji umr, lowongan`).
  - **Cluster 1 (31 postingan)**: *Informasi Lowongan Kerja dan Rekrutmen* (Kata dominan: `oktober, posisi, pt, lowongan, baca`).
  - **Cluster 2 (120 postingan)**: *Peluang Kerja dan Kesulitan Mendapatkan Pekerjaan* (Kata dominan: `fresh graduate, fresh, graduate, pengalaman, kerja`).

### 4. Analisis Sentimen (IndoBERT)
- Menggunakan pipeline `text-classification` dengan model [`mdhugol/indonesia-bert-sentiment-classification`](https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification) (`batch_size=32`, `truncation=True`, `max_length=128`).
- Pemetaan label: `LABEL_0` = **Positif**, `LABEL_1` = **Netral**, `LABEL_2` = **Negatif**.
- **Hasil Klasifikasi pada 516 Postingan**:
  - **Netral**: 260 postingan (`50.39%`)
  - **Negatif**: 211 postingan (`40.89%`)
  - **Positif**: 45 postingan (`8.72%`)
- *(Catatan: Pada `app.py`, apabila pengguna mengunggah CSV yang belum memiliki kolom `sentimen_sistem`, aplikasi menyediakan fallback otomatis berbasis kamus kata positif dan negatif).*

### 5. Regresi Linear (Prediksi Jumlah Likes di `app.py`)
- **Variabel Target ($y$)**: `like_count` (Jumlah Likes).
- **Variabel Fitur ($X$)**: `reply_count` (Jumlah Balasan) dan `retweet_count` (Jumlah Retweet). *(Aplikasi juga mengekstrak kolom `word_count` dan `hashtag_count` pada saat pembacaan data).*
- **Pembagian Data**: 80% data latih (*training set*) dan 20% data uji (*testing set*) dengan `random_state=42`.
- **Evaluasi**: Mengukur nilai **MAE (*Mean Absolute Error*)**, **R² Score**, serta menampilkan koefisien regresi masing-masing fitur dan memvisualisasikan perbandingan nilai aktual vs prediksi terhadap garis ideal ($y = \hat{y}$).

---

## 🛠️ Penyelesaian Masalah (Troubleshooting)

| Masalah yang Muncul | Penyebab & Solusi |
|---|---|
| **Crawling berhenti / error JSON response di Colab** | Akun X terkena batas pengambilan data (*rate limit*) atau `auth_token` tidak valid/kedaluwarsa. Tunggu beberapa saat atau ganti dengan `auth_token` baru. |
| **Error nama paket saat `apt-get install` di Colab** | Nama paket berakhiran `t64` (seperti `libatk1.0-0t64`, `libasound2t64`) diperuntukkan bagi Google Colab berbasis Ubuntu 24.04 (Noble). Jika menggunakan Ubuntu versi lama, hapus akhiran `t64`. |
| **`NameError: name 'pipeline' / 'torch' is not defined` pada sel Sentimen Colab** | Tambahkan baris `import torch` dan `from transformers import pipeline` di bagian atas sel pemodelan IndoBERT sebelum memanggil `pipeline(...)`. |
| **`FileNotFoundError` saat menjalankan sel di Colab** | Sesi runtime Colab terputus sehingga file di `/content/` terhapus. Jalankan ulang dari tahap sebelumnya atau unggah file CSV cadangan ke panel **Files** di sebelah kiri Colab. |
| **Analisis sentimen berjalan sangat lambat** | Runtime Colab masih menggunakan CPU. Ubah ke **T4 GPU** melalui menu **Runtime > Change runtime type**. |
| **Peringatan `Dataset tidak valid!` di Streamlit** | File CSV yang diunggah ke `app.py` tidak memiliki salah satu dari 4 kolom wajib inti (`created_at`, `text`, `Cleaned_Text`, `like_count`). Pastikan file yang diunggah adalah `hasil_sentimen_lengkap.csv` atau `cleaning_kerja_relevan.csv`. |

---

## ⚠️ Keterbatasan Proyek

1. **Ketergantungan Data Crawling**: Jumlah dan isi data hasil *crawling* sangat bergantung pada algoritma pencarian X, batas *rate limit* akun, serta ketersediaan cuitan pada rentang tanggal yang ditentukan.
2. **Konteks Sarkasme pada Sentimen**: Model IndoBERT yang digunakan merupakan model *pre-trained* tanpa pelatihan ulang (*fine-tuning*) khusus pada bahasa gaul X, sehingga cuitan bernada sarkasme atau sindiran halus terkadang terklasifikasi sebagai sentimen **Netral**.
3. **Irisan Kata pada Clustering**: Penentuan jumlah cluster ($K=3$) ditetapkan secara manual. Masih terdapat tumpang tindih (*overlap*) pada kata-kata umum seperti `kerja`, `gaji`, dan `umr` karena baik pengumuman lowongan maupun keluhan pekerja sama-sama menggunakan kosakata tersebut.
4. **Sifat Linear dan Prediksi Negatif pada Regresi**: Prediksi jumlah *likes* hanya menggunakan dua fitur interaksi (`reply_count` dan `retweet_count`) tanpa memperhitungkan jumlah pengikut (*followers*), waktu unggah, maupun keterlibatan media gambar/video. Selain itu, karena distribusi *likes* sangat miring (*skewed*), titik potong (*intercept*) model dapat bernilai negatif dan menghasilkan angka estimasi negatif ketika jumlah *reply* dan *retweet* bernilai sangat kecil.

---

## 👥 Anggota Kelompok

| Nama Lengkap | NIM |
|---|---|
| **NESHIA SEPTIARIFA** | 24051130001 |
| **ANGGER SEKARAYU LANGGENG BUSHAE** | 24051130003 |
| **BAYU ARYA WIJAYA** | 24051130010 |
| **AUDI ARIZ Q** | 24051130016 |

---
*Dibuat untuk memenuhi tugas perkuliahan Analisis Data.*