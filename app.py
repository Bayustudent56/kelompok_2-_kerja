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
st.markdown("Aplikasi web interaktif untuk menampilkan hasil analisis sentimen, clustering teks, dan regresi linear prediksi likes.")

# -------------------------------------------------------------------
# SIDEBAR - UNGGAH CSV HASIL COLAB
# -------------------------------------------------------------------
st.sidebar.header("📂 Unggah Data")
uploaded_file = st.sidebar.file_uploader("Unggah CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.info("👋 **Selamat Datang!** Silakan unggah file CSV hasil pemrosesan Google Colab Anda melalui panel di sebelah kiri (Sidebar) untuk menampilkan dashboard.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"⚠️ Gagal membaca file CSV. Detail error: {e}")
    st.stop()

# -------------------------------------------------------------------
# PEMERIKSAAN KOLOM WAJIB
# -------------------------------------------------------------------
kolom_wajib = ['created_at', 'text', 'Cleaned_Text', 'like_count']
missing_cols = [col for col in kolom_wajib if col not in df.columns]

# -------------------------------------------------------------------
# PENGECEKAN OTOMATIS KOLOM HASIL COLAB (TANPA INDOBERT/TRANSFORMERS)
# Jika CSV yang diupload sudah ada kolom dari Colab, langsung tampilkan!
# Jika belum ada, buat dengan cepat agar aplikasi tidak pernah error.
# -------------------------------------------------------------------
if not missing_cols:
    # 1. Cek kolom sentimen
    if 'sentiment' not in df.columns:
        pos_words = set(['gaji', 'bagus', 'lulus', 'alhamdulillah', 'peluang', 'tertarik', 'semangat', 'bismillah', 'gratis', 'info', 'sesuai', 'rekomendasi', 'terbaru', 'welcome', 'mampu', 'naik', 'berkah', 'sukses', 'terbuka'])
        neg_words = set(['sedih', 'scam', 'sulit', 'turun', 'sakit', 'ngos', 'burnout', 'layoff', 'phk', 'overwork', 'pengangguran', 'susah', 'minimal', 'rugi', 'kecewa', 'revisi', 'menyedihkan', 'sulitnya', 'beban', 'keluhan'])
        def get_sentiment(text_val):
            words = str(text_val).lower().split()
            pos = sum(1 for w in words if w in pos_words)
            neg = sum(1 for w in words if w in neg_words)
            return 'Positif' if pos > neg else ('Negatif' if neg > pos else 'Netral')
        df['sentiment'] = df['Cleaned_Text'].fillna('').apply(get_sentiment)

    # 2. Cek kolom cluster & kata dominan
    if 'cluster' not in df.columns or 'cluster_top_words' not in df.columns:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_tfidf = vectorizer.fit_transform(df['Cleaned_Text'].fillna('').astype(str))
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_tfidf)
        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        top_words_map = {c_id: ", ".join([terms[ind] for ind in order_centroids[c_id, :5]]) for c_id in range(3)}
        df['cluster_top_words'] = df['cluster'].map(top_words_map)

    # 3. Cek kolom fitur regresi
    if 'word_count' not in df.columns:
        df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    if 'hashtag_count' not in df.columns:
        df['hashtag_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x)))

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
        st.error(f"⚠️ **Dataset tidak valid!** Kolom wajib berikut tidak ditemukan: **{', '.join(missing_cols)}**.")
        st.stop()
    
    st.subheader("Pratinjau Dataset (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)
    
    null_cleaned = df['Cleaned_Text'].isna().sum()
    null_likes = pd.to_numeric(df['like_count'], errors='coerce').isna().sum()
    st.info(f"ℹ️ **Info Dataset Utama:** Terdapat `{len(df)}` baris total. Baris teks bersih kosong: `{null_cleaned}` baris. Baris likes tidak valid: `{null_likes}` baris.")

if missing_cols:
    st.stop()

# ===================================================================
# TAB 2: SENTIMEN (MENAMPILKAN HASIL SENTIMEN)
# ===================================================================
with tab2:
    st.header("😃 Hasil Analisis Sentimen")
    
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percent = (df['sentiment'].value_counts(normalize=True) * 100).round(2)
    df_sent_summary = pd.DataFrame({
        'Kategori Sentimen': sentiment_counts.index,
        'Jumlah Postingan': sentiment_counts.values, 
        'Persentase (%)': sentiment_percent.values
    })
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Ringkasan Kategori")
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
        ax.set_xlabel("Sentimen")
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 3, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        
    st.subheader("Tabel Hasil Klasifikasi Postingan")
    st.dataframe(df[['text', 'Cleaned_Text', 'sentiment']].head(20), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Ringkasan Pemeriksaan Manual (20 Sampel Acak)")
    
    df_eval = df[['text', 'Cleaned_Text', 'sentiment']].sample(n=min(20, len(df)), random_state=42).copy()
    df_eval.rename(columns={'sentiment': 'prediksi_sistem'}, inplace=True)
    
    label_kelompok = [
        'Positif', 'Negatif', 'Netral', 'Positif', 'Negatif',
        'Netral', 'Netral', 'Netral', 'Negatif', 'Negatif',
        'Netral', 'Positif', 'Positif', 'Netral', 'Negatif',
        'Negatif', 'Netral', 'Negatif', 'Negatif', 'Negatif'
    ]
    df_eval['label_manual'] = label_kelompok[:len(df_eval)]
    df_eval['Status'] = np.where(df_eval['label_manual'] == df_eval['prediksi_sistem'], '✅ Sesuai', '❌ Tidak Sesuai')
    
    sesuai = (df_eval['label_manual'] == df_eval['prediksi_sistem']).sum()
    acc = (sesuai / len(df_eval)) * 100
    
    col_e1, col_e2 = st.columns(2)
    col_e1.metric("Jumlah Sesuai", f"{sesuai} dari {len(df_eval)} sampel")
    col_e2.metric("Persentase Kesesuaian", f"{acc:.2f}%")
    
    st.dataframe(df_eval[['text', 'label_manual', 'prediksi_sistem', 'Status']], use_container_width=True)
    
    st.subheader("📌 Pembahasan 3 Contoh Ketidaksesuaian")
    st.markdown("""
    1. **Keluhan Gaji di Bawah UMR:** *"ngeluh ngeluh gaji under UMR ga worth it tapi tetep dilamar 😔😔😔"*  
       * **Label Manual:** `Negatif` | **Prediksi Sistem:** `Positif` $\rightarrow$ Terjadi karena terdapat kata *"gaji"* di kamus positif, sedangkan kata *"ngeluh"* belum masuk kamus negatif.
    2. **Keluhan Susah Cari Kerja:** *"tbh gue mau nangis aja si, temen gue banyak bgt lulusan sma/smk susah cari kerja bgt dan gaji mereka jauh di bawah umr..."*  
       * **Label Manual:** `Negatif` | **Prediksi Sistem:** `Netral` $\rightarrow$ Skor seri antara kata positif (*"gaji"*) dan negatif (*"susah"*), padahal konteks utuhnya adalah kesedihan.
    3. **Iklan Lowongan Kerja:** *"Like & Share | Lowongan Kerja Kasir di Arei Adventure Store Jogja 3 - Yogyakarta (Gaji UMR Jogja)"*  
       * **Label Manual:** `Netral` | **Prediksi Sistem:** `Positif` $\rightarrow$ Kalimat informatif pengumuman lowongan kerja terbaca positif karena memuat kata *"gaji"*.
    """)

# ===================================================================
# TAB 3: CLUSTERING (MENAMPILKAN HASIL CLUSTER)
# ===================================================================
with tab3:
    st.header("🏷️ Hasil Clustering Data Teks (K-Means)")
    
    tema_map = {
        0: "Informasi Lowongan Kerja & Kualifikasi Pelamar",
        1: "Tantangan Fresh Graduate & Syarat Pengalaman Kerja",
        2: "Diskusi Gaji UMR, Keluhan Pekerja, dan Biaya Hidup"
    }
    
    cluster_counts = df['cluster'].value_counts().sort_index()
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Ukuran Cluster 0", f"{cluster_counts.get(0, 0)} postingan")
    col_c2.metric("Ukuran Cluster 1", f"{cluster_counts.get(1, 0)} postingan")
    col_c3.metric("Ukuran Cluster 2", f"{cluster_counts.get(2, 0)} postingan")
    st.markdown("---")

    for c_id in range(3):
        st.subheader(f"🔹 Cluster {c_id}: {tema_map.get(c_id, f'Tema {c_id}')}")
        df_c = df[df['cluster'] == c_id]
        c_count = len(df_c)
        top_words = df_c['cluster_top_words'].iloc[0] if c_count > 0 else "-"
        
        st.write(f"- **Ukuran Cluster**: `{c_count}` postingan ({(c_count/len(df)*100):.1f}%)")
        st.write(f"- **5 Kata Dominan**: `{top_words}`")
        st.write("- **3 Contoh Postingan**:")
        
        for idx, sample_text in enumerate(df_c['text'].head(3).tolist(), 1):
            st.info(f"**Contoh {idx}:** {sample_text}")
        st.markdown("---")

    st.subheader("💡 Penjelasan & Perbedaan Cluster")
    st.markdown("""
    * **Perbedaan Ketiga Cluster:** Cluster 0 berisi pengumuman lowongan kerja, Cluster 1 fokus pada keluh kesah *fresh graduate* terkait syarat pengalaman kerja, dan Cluster 2 berisi obrolan tentang besaran gaji UMR dan biaya hidup.
    * **Tumpang Tindih (Overlap):** Masih terdapat tumpang tindih pada kata umum seperti `kerja`, `gaji`, dan `umr` karena informasi lowongan kerja maupun keluhan pekerja sama-sama sering menyebutkan istilah tersebut.
    """)

# ===================================================================
# TAB 4: REGRESI (EVALUASI & FORMULIR INTERAKTIF)
# ===================================================================
with tab4:
    st.header("📈 Regresi Linear (Estimasi Jumlah Likes)")
    
    df['like_count_num'] = pd.to_numeric(df['like_count'], errors='coerce')
    df_reg_clean = df.dropna(subset=['like_count_num']).copy()
    dropped_reg = len(df) - len(df_reg_clean)
    st.caption(f"*Catatan: Menggunakan {len(df_reg_clean)} baris valid ({dropped_reg} baris dikeluarkan karena target likes kosong/tidak valid).*")
    
    X_reg = df_reg_clean[['word_count', 'hashtag_count']]
    y_reg = df_reg_clean['like_count_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    model_reg = LinearRegression()
    model_reg.fit(X_train, y_train)
    y_pred = model_reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("MAE (Mean Absolute Error)", f"{mae:,.2f}")
    col2.metric("R² Score", f"{r2:.4f}")
    
    col_g1, col_g2 = st.columns([1.3, 1])
    with col_g1:
        st.subheader("Grafik Likes Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(y_test, y_pred, color='royalblue', alpha=0.6, label='Data Uji')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal')
        ax.set_xlabel("Likes Aktual")
        ax.set_ylabel("Likes Prediksi")
        ax.legend()
        st.pyplot(fig)
        
    with col_g2:
        st.subheader("🧮 Formulir Estimasi Likes")
        with st.form("form_likes"):
            input_words = st.number_input("Jumlah Kata dalam Postingan:", min_value=1, max_value=500, value=20)
            input_hashtags = st.number_input("Jumlah Hashtag (#):", min_value=0, max_value=50, value=1)
            submit_btn = st.form_submit_button("Estimasi Likes")
            
            if submit_btn:
                input_df = pd.DataFrame({'word_count': [input_words], 'hashtag_count': [input_hashtags]})
                pred_val = model_reg.predict(input_df)[0]
                st.success(f"📌 Estimasi Jumlah Likes: **{pred_val:,.2f}** likes")
                if pred_val < 0:
                    st.warning("⚠️ Catatan: Nilai negatif tidak bermakna sebagai jumlah likes nyata dan menunjukkan keterbatasan model regresi linear sederhana.")

    st.subheader("📖 Interpretasi Hasil Regresi")
    st.markdown(f"""
    * **Makna MAE (`{mae:,.2f}`):** Rata-rata kesalahan prediksi model terhadap data uji adalah sekitar {mae:,.0f} likes, dipengaruhi adanya postingan *viral* (outlier) dengan puluhan ribu likes.
    * **Makna $R^2$ (`{r2:.4f}`):** Nilai $R^2$ negatif/mendekati nol menunjukkan bahwa jumlah kata dan hashtag saja tidak mampu menjelaskan variasi jumlah likes. Faktor seperti jumlah *followers*, waktu unggah, dan isi topik jauh lebih berpengaruh.
    """)

# ===================================================================
# BAGIAN AKHIR: TEMUAN & KETERBATASAN
# ===================================================================
st.markdown("---")
st.header("📌 Kesimpulan Analisis")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🔍 3 Temuan Utama")
    st.markdown("""
    1. **Isu Gaji UMR Mendominasi**: Mayoritas percakapan (230 postingan di Cluster 2) membahas standar gaji UMR dan biaya hidup.
    2. **Tantangan Fresh Graduate**: Kelompok diskusi tersendiri (96 postingan di Cluster 1) menyoroti sulitnya syarat pengalaman kerja bagi lulusan baru.
    3. **Jumlah Kata & Hashtag Tidak Menentukan Likes**: Evaluasi regresi ($R^2 = -0.0182$) membuktikan panjang cuitan dan hashtag tidak berkorelasi kuat dengan jumlah likes.
    """)

with col_b:
    st.subheader("⚠️ 2 Keterbatasan Data & Metode")
    st.markdown("""
    1. **Keterbatasan Lexicon Sentimen**: Kamus kata sederhana belum dapat membedakan kalimat keluhan atau sarkasme yang memuat kata positif seperti *"gaji"*.
    2. **Keterbatasan Fitur Regresi**: Ketidaktersediaan data jumlah *followers* akun dan adanya *outlier* cuitan viral membuat estimasi regresi linear dua variabel memiliki error (MAE) yang besar.
    """)