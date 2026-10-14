import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Data X - Dunia Kerja",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Posting X (Twitter) - Topik Dunia Kerja")
st.markdown("Aplikasi web interaktif untuk analisis sentimen, clustering teks, dan regresi linear prediksi likes.")

# -------------------------------------------------------------------
# SIDEBAR - UNGGAH DATASET (TANPA FILE INTERNAL)
# -------------------------------------------------------------------
st.sidebar.header("📂 Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("1. Unggah CSV Dataset Utama", type=["csv"])
uploaded_eval = st.sidebar.file_uploader("2. Unggah CSV Evaluasi Sentimen (Opsional)", type=["csv"])

# Jika belum ada file yang diunggah, tampilkan pesan instruksi dan hentikan eksekusi
if uploaded_file is None:
    st.info("👋 **Selamat Datang!** Silakan unggah file CSV dataset utama Anda (misal: `cleaning_kerja_relevan.csv`) melalui panel di sebelah kiri (Sidebar) untuk memulai analisis.")
    st.stop()

# Membaca dataset dari file yang diunggah
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"⚠️ Gagal membaca file CSV. Pastikan format file benar. Detail error: {e}")
    st.stop()

# -------------------------------------------------------------------
# PEMERIKSAAN KOLOM WAJIB
# -------------------------------------------------------------------
kolom_wajib = ['created_at', 'text', 'Cleaned_Text', 'like_count']
missing_cols = [col for col in kolom_wajib if col not in df.columns]

# -------------------------------------------------------------------
# TAB APLIKASI
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📁 Data", "😃 Sentimen", "🏷️ Clustering", "📈 Regresi"])

# ===================================================================
# TAB 1: DATA
# ===================================================================
with tab1:
    st.header("📁 Pratinjau & Ringkasan Dataset")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Postingan", f"{len(df)} baris")
    col2.metric("Jumlah Kolom", f"{len(df.columns)} kolom")
    col3.metric("Status Kolom Wajib", "Lengkap ✅" if not missing_cols else "Tidak Lengkap ❌")
    
    st.subheader("Pemeriksaan Kolom Wajib")
    col_status = pd.DataFrame({
        "Nama Kolom Wajib": kolom_wajib,
        "Status Ketersediaan": ["✅ Tersedia" if col in df.columns else "❌ Tidak Tersedia" for col in kolom_wajib]
    })
    st.table(col_status)

    if missing_cols:
        st.error(f"⚠️ **Dataset tidak valid!** Kolom wajib berikut tidak ditemukan: **{', '.join(missing_cols)}**. Silakan unggah dataset yang sesuai.")
        st.stop()
    
    st.subheader("Pratinjau Dataset (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Informasi Pembersihan & Baris Data")
    null_cleaned = df['Cleaned_Text'].isna().sum()
    null_likes = pd.to_numeric(df['like_count'], errors='coerce').isna().sum()
    st.markdown(f"""
    * **Total baris di dalam dataset utama:** `{len(df)}` postingan unik.
    * **Baris dengan teks bersih (`Cleaned_Text`) kosong:** `{null_cleaned}` baris.
    * **Baris dengan jumlah likes (`like_count`) tidak valid/kosong:** `{null_likes}` baris.
    *(Catatan: Seluruh analisis menggunakan dataset utama yang sama. Jika ada baris bernilai kosong pada target/fitur tertentu, baris tersebut akan dikeluarkan khusus pada tahap analisis terkait).*
    """)

# Jika kolom wajib tidak lengkap, hentikan agar tab lain tidak error
if missing_cols:
    st.stop()

# ===================================================================
# TAB 2: SENTIMEN
# ===================================================================
with tab2:
    st.header("😃 Analisis Sentimen")
    
    pos_words = set([
        'gaji', 'bagus', 'lulus', 'alhamdulillah', 'peluang', 'tertarik', 
        'semangat', 'bismillah', 'gratis', 'info', 'sesuai', 'rekomendasi', 
        'terbaru', 'welcome', 'mampu', 'naik', 'berkah', 'sukses', 'terbuka'
    ])
    neg_words = set([
        'sedih', 'scam', 'sulit', 'turun', 'sakit', 'ngos', 'burnout', 
        'layoff', 'phk', 'overwork', 'pengangguran', 'susah', 'minimal', 
        'rugi', 'kecewa', 'revisi', 'menyedihkan', 'sulitnya', 'beban', 'keluhan'
    ])

    def get_sentiment(text_val):
        words = str(text_val).lower().split()
        pos = sum(1 for w in words if w in pos_words)
        neg = sum(1 for w in words if w in neg_words)
        if pos > neg:
            return 'Positif'
        elif neg > pos:
            return 'Negatif'
        else:
            return 'Netral'

    df['sentiment'] = df['Cleaned_Text'].fillna('').apply(get_sentiment)
    
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percent = (df['sentiment'].value_counts(normalize=True) * 100).round(2)
    df_sent_summary = pd.DataFrame({
        'Kategori Sentimen': sentiment_counts.index,
        'Jumlah Postingan': sentiment_counts.values, 
        'Persentase (%)': sentiment_percent.values
    })
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Ringkasan Distribusi Sentimen")
        st.dataframe(df_sent_summary, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Grafik Distribusi Sentimen")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            x=sentiment_counts.index, 
            y=sentiment_counts.values, 
            hue=sentiment_counts.index,
            palette='coolwarm', 
            legend=False,
            ax=ax
        )
        ax.set_ylabel("Jumlah Postingan")
        ax.set_xlabel("Kategori Sentimen")
        ax.set_title("Distribusi Sentimen Postingan Topik Dunia Kerja")
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 3, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        
    st.subheader("Tabel Hasil Klasifikasi Postingan")
    st.dataframe(df[['text', 'Cleaned_Text', 'sentiment']].head(20), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Ringkasan Pemeriksaan Manual (20 Sampel Acak)")
    
    if uploaded_eval is not None:
        df_eval = pd.read_csv(uploaded_eval)
        st.dataframe(df_eval.head(20), use_container_width=True)
        if 'label_manual' in df_eval.columns and 'prediksi_sistem' in df_eval.columns:
            sesuai = (df_eval['label_manual'] == df_eval['prediksi_sistem']).sum()
            total_eval = len(df_eval)
            acc = (sesuai / total_eval) * 100
            st.success(f"💡 **Hasil Evaluasi:** {sesuai} dari {total_eval} sampel sesuai (**{acc:.2f}%**).")
    else:
        st.info("ℹ️ Anda belum mengunggah file `evaluasi_sentimen.csv` di Sidebar. Berikut adalah tabel sampel 20 postingan acak dari dataset utama untuk pemeriksaan manual:")
        df_sample20 = df[['text', 'Cleaned_Text', 'sentiment']].sample(n=min(20, len(df)), random_state=42).copy()
        df_sample20.rename(columns={'sentiment': 'prediksi_sistem'}, inplace=True)
        st.dataframe(df_sample20, use_container_width=True)

    with st.expander("📖 Pembahasan Ketidaksesuaian Sentimen (Lexicon vs Pembacaan Manual)", expanded=True):
        st.markdown("""
        **Analisis Penyebab Ketidaksesuaian Klasifikasi:**
        1. **Kata Kunci 'Gaji' pada Konteks Keluhan:** Kata `'gaji'` dimasukkan ke dalam kamus positif, sehingga kalimat keluhan seperti *"Gaji UMR sekarang tidak cukup untuk biaya hidup"* sering terklasifikasi sebagai **Positif** atau **Netral**, padahal secara manual bermakna **Negatif**.
        2. **Kalimat Sindiran / Sarkasme:** Postingan seperti *"Mimpi basah para pejuang loker: tidak ada batas usia, gaji 12 juta"* sebenarnya menyindir kondisi syarat loker Indonesia yang rumit (Negatif/Kritik), namun terbaca **Positif** karena mengandung kata positif.
        3. **Iklan Lowongan Kerja (Loker):** Banyak postingan berisi syarat loker seperti *"Terbuka untuk Fresh Graduate, mampu berbahasa Inggris"* yang secara konteks bersifat informatif (**Netral**), tetapi diklasifikasikan **Positif** karena adanya kata `'mampu'` atau `'terbuka'`.
        """)

# ===================================================================
# TAB 3: CLUSTERING
# ===================================================================
with tab3:
    st.header("🏷️ Clustering Data Teks (TF-IDF & K-Means)")
    
    # Memastikan tidak ada baris kosong saat TF-IDF
    df_cluster = df.dropna(subset=['Cleaned_Text']).copy()
    dropped_cluster = len(df) - len(df_cluster)
    if dropped_cluster > 0:
        st.caption(f"*Catatan: Sebanyak {dropped_cluster} baris dikeluarkan dari clustering karena teks bersih kosong.*")
    else:
        st.caption(f"*Catatan: Seluruh {len(df_cluster)} baris dataset utama digunakan dalam clustering (0 baris dikeluarkan).*")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df_cluster['Cleaned_Text'].astype(str))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans.fit_predict(X_tfidf)
    
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    # Penamaan tema yang disesuaikan dengan hasil kata dominan pada data Anda
    tema_map = {
        0: "Informasi Lowongan Kerja & Kualifikasi Pelamar",
        1: "Tantangan Fresh Graduate & Syarat Pengalaman Kerja",
        2: "Diskusi Gaji UMR, Keluhan Pekerja, dan Biaya Hidup"
    }
    
    # Ringkasan ukuran cluster
    cluster_counts = df_cluster['cluster'].value_counts().sort_index()
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Ukuran Cluster 0", f"{cluster_counts.get(0, 0)} postingan")
    col_c2.metric("Ukuran Cluster 1", f"{cluster_counts.get(1, 0)} postingan")
    col_c3.metric("Ukuran Cluster 2", f"{cluster_counts.get(2, 0)} postingan")
    st.markdown("---")

    for c_id in range(3):
        st.subheader(f"🔹 Cluster {c_id}: {tema_map[c_id]}")
        c_count = (df_cluster['cluster'] == c_id).sum()
        top_words = ", ".join([terms[ind] for ind in order_centroids[c_id, :5]])
        
        st.write(f"- **Jumlah Postingan**: `{c_count}` postingan ({(c_count/len(df_cluster)*100):.1f}%)")
        st.write(f"- **5 Kata Dominan**: `{top_words}`")
        st.write("- **3 Contoh Postingan**:")
        
        samples = df_cluster[df_cluster['cluster'] == c_id]['text'].head(3).tolist()
        for idx, sample_text in enumerate(samples, 1):
            st.info(f"**Contoh {idx}:** {sample_text}")
        st.markdown("---")

    st.subheader("💡 Interpretasi Hasil Clustering")
    st.markdown("""
    * **Apa yang membedakan isi ketiga cluster?**
      * **Cluster 0** berfokus pada penyebaran informasi lowongan kerja (*job vacancy*), syarat pendaftaran, dan posisi yang dibutuhkan perusahaan.
      * **Cluster 1** berfokus pada diskusi mengenai kesulitan lulusan baru (*fresh graduate*) dalam menembus pasar kerja yang kerap menuntut pengalaman kerja.
      * **Cluster 2** berfokus pada perbincangan seputar nominal gaji, standar UMR di berbagai daerah, serta keluhan keseharian pekerja terkait biaya hidup.
    * **Apakah masih terdapat pembahasan yang tumpang tindih?**
      * **Ya, masih terdapat tumpang tindih (*overlap*)**, terutama penggunaan kata umum seperti `kerja`, `gaji`, dan `umr` yang muncul di Cluster 0 maupun Cluster 2. Hal ini wajar karena pengumuman lowongan kerja sering kali turut mencantumkan informasi gaji UMR, sementara diskusi keluhan pekerja juga membahas topik gaji dan pekerjaan.
    """)

# ===================================================================
# TAB 4: REGRESI
# ===================================================================
with tab4:
    st.header("📈 Regresi Linear Sederhana (Estimasi Likes)")
    
    # Menghitung fitur dari teks asli sesuai ketentuan tugas
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    df['hashtag_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x)))
    df['like_count_num'] = pd.to_numeric(df['like_count'], errors='coerce')
    
    # Memfilter hanya baris dengan target like_count yang valid (tanpa mengganti NaN menjadi 0)
    df_reg_clean = df.dropna(subset=['like_count_num']).copy()
    dropped_reg = len(df) - len(df_reg_clean)
    
    st.caption(f"*Catatan Data Regresi: Menggunakan {len(df_reg_clean)} baris valid dari total {len(df)} baris ({dropped_reg} baris dikeluarkan karena nilai likes tidak valid/kosong).*")
    
    X_reg = df_reg_clean[['word_count', 'hashtag_count']]
    y_reg = df_reg_clean['like_count_num']
    
    # Pembagian data 80% latih dan 20% uji dengan random_state tetap
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    model_reg = LinearRegression()
    model_reg.fit(X_train, y_train)
    y_pred = model_reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Mean Absolute Error)", f"{mae:,.2f}")
    col2.metric("R² Score (Koefisien Determinasi)", f"{r2:.4f}")
    col3.metric("Rasio Data (Latih : Uji)", f"{len(X_train)} : {len(X_test)}")
    
    col_g1, col_g2 = st.columns([1.3, 1])
    with col_g1:
        st.subheader("Grafik Likes Aktual vs Hasil Prediksi")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(y_test, y_pred, color='royalblue', alpha=0.6, label='Data Uji (Aktual vs Prediksi)')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal (Prediksi = Aktual)')
        ax.set_xlabel("Jumlah Likes Aktual")
        ax.set_ylabel("Jumlah Likes Prediksi")
        ax.set_title("Perbandingan Likes Aktual dan Prediksi pada Data Uji")
        ax.legend()
        st.pyplot(fig)
        
    with col_g2:
        st.subheader("🧮 Formulir Estimasi Jumlah Likes")
        with st.form("form_likes"):
            input_words = st.number_input("Masukkan Jumlah Kata dalam Postingan:", min_value=1, max_value=500, value=25)
            input_hashtags = st.number_input("Masukkan Jumlah Hashtag (#):", min_value=0, max_value=50, value=1)
            submit_btn = st.form_submit_button("Hitung Estimasi Likes")
            
            if submit_btn:
                input_df = pd.DataFrame({'word_count': [input_words], 'hashtag_count': [input_hashtags]})
                pred_val = model_reg.predict(input_df)[0]
                st.success(f"📌 Estimasi Jumlah Likes: **{pred_val:,.2f}** likes")
                if pred_val < 0:
                    st.warning("⚠️ **Catatan Penting:** Hasil estimasi bernilai negatif tidak bermakna sebagai jumlah likes di dunia nyata (karena likes minimal 0). Hal ini menunjukkan keterbatasan model regresi linear sederhana.")

    st.subheader("📖 Interpretasi Evaluasi Model Regresi")
    st.markdown(f"""
    1. **Makna Nilai MAE (`{mae:,.2f}`):** Rata-rata selisih absolut antara jumlah *likes* aktual dan jumlah *likes* yang diprediksi oleh model adalah sekitar **{mae:,.0f} likes**. Nilai error yang besar ini disebabkan oleh adanya beberapa cuitan *viral* (outlier) yang memiliki puluhan ribu *likes*, sementara mayoritas cuitan lainnya hanya memiliki sedikit *likes*.
    2. **Makna Nilai $R^2$ (`{r2:.4f}`):** Nilai $R^2$ yang mendekati nol atau bernilai **negatif** menunjukkan bahwa model garis regresi linear bekerja lebih buruk dibandingkan sekadar menebak menggunakan nilai rata-rata (*mean*) *likes*. Artinya, variasi jumlah kata dan jumlah hashtag hampir tidak mampu menjelaskan variasi jumlah *likes* (0%).
    3. **Keterbatasan Dua Fitur Masukan:** Jumlah *likes* di X (Twitter) tidak ditentukan oleh panjang pendeknya kalimat atau banyaknya hashtag, melainkan sangat dipengaruhi oleh faktor eksternal seperti **jumlah pengikut (*followers*) akun, keviralan topik, waktu publikasi (*prime time*), dan daya tarik emosional isi pesan**.
    """)

# ===================================================================
# BAGIAN AKHIR: TEMUAN UTAMA & KETERBATASAN
# ===================================================================
st.markdown("---")
st.header("📌 Kesimpulan Analisis")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🔍 3 Temuan Utama")
    st.markdown("""
    1. **Dominasi Pembahasan Gaji UMR & Biaya Hidup**: Dari hasil clustering (230 dari 516 postingan), percakapan warganet paling banyak menyoroti isu standar gaji UMR dan keluhan mengenai kecukupan upah terhadap biaya hidup.
    2. **Keresahan Fresh Graduate terhadap Syarat Kerja**: Terdapat kelompok diskusi khusus (96 postingan) yang mengeluhkan tingginya persyaratan lowongan kerja bagi lulusan baru (*fresh graduate*), terutama keharusan memiliki pengalaman kerja.
    3. **Panjang Teks & Hashtag Tidak Menentukan Likes**: Evaluasi regresi linear membuktikan bahwa jumlah kata dan jumlah hashtag bukan indikator penentu tingginya *engagement* (*likes*) pada topik dunia kerja ($R^2 = -0.0182$).
    """)

with col_b:
    st.subheader("⚠️ 2 Keterbatasan Data & Metode")
    st.markdown("""
    1. **Keterbatasan Metode Sentimen Lexicon Sederhana**: Klasifikasi berbasis pencocokan kata (*lexicon*) belum mampu menangkap konteks kalimat secara utuh, seperti sarkasme, kalimat negasi kompleks, atau kata kunci positif (seperti kata *"gaji"*) yang sebenarnya digunakan dalam kalimat keluhan.
    2. **Keterbatasan Fitur Prediksi & Ketimpangan Distribusi Likes**: Dataset memiliki distribusi *likes* yang sangat timpang (*skewed*) akibat adanya sedikit postingan *viral*, serta tidak tersedianya fitur penting seperti jumlah *followers* pengunggah dan usia postingan dalam pemodelan regresi.
    """)