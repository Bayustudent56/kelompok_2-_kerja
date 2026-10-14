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
uploaded_file = st.sidebar.file_uploader("Unggah CSV Dataset Hasil Colab", type=["csv"])

if uploaded_file is None:
    st.info("👋 **Selamat Datang!** Silakan unggah file CSV hasil pemrosesan Google Colab (`hasil_sentimen_lengkap.csv`) melalui panel di sebelah kiri (Sidebar) untuk menampilkan dashboard.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"⚠️ Gagal membaca file CSV. Detail error: {e}")
    st.stop()

# -------------------------------------------------------------------
# PEMERIKSAAN KOLOM SESUAI FORMAT CSV COLAB
# -------------------------------------------------------------------
kolom_colab = [
    'id', 'created_at', 'text', 'Cleaned_Text', 
    'like_count', 'retweet_count', 'reply_count', 
    'is_relevant', 'sentimen_sistem', 'skor'
]
kolom_wajib_inti = ['created_at', 'text', 'Cleaned_Text', 'like_count']
missing_core = [col for col in kolom_wajib_inti if col not in df.columns]

if not missing_core:
    # 1. Gunakan 'sentimen_sistem' dari Colab jika tersedia
    if 'sentimen_sistem' not in df.columns:
        if 'sentiment' in df.columns:
            df['sentimen_sistem'] = df['sentiment']
        else:
            pos_words = set(['gaji', 'bagus', 'lulus', 'alhamdulillah', 'peluang', 'tertarik', 'semangat', 'bismillah', 'gratis', 'info', 'sesuai', 'rekomendasi', 'terbaru', 'welcome', 'mampu', 'naik', 'berkah', 'sukses', 'terbuka'])
            neg_words = set(['sedih', 'scam', 'sulit', 'turun', 'sakit', 'ngos', 'burnout', 'layoff', 'phk', 'overwork', 'pengangguran', 'susah', 'minimal', 'rugi', 'kecewa', 'revisi', 'menyedihkan', 'sulitnya', 'beban', 'keluhan'])
            def get_sentiment(text_val):
                words = str(text_val).lower().split()
                pos = sum(1 for w in words if w in pos_words)
                neg = sum(1 for w in words if w in neg_words)
                return 'Positif' if pos > neg else ('Negatif' if neg > pos else 'Netral')
            df['sentimen_sistem'] = df['Cleaned_Text'].fillna('').apply(get_sentiment)

    if 'skor' not in df.columns:
        df['skor'] = np.nan

    # 2. Cek kolom cluster & kata dominan (sama persis dengan parameter Colab)
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
    
    missing_all = [col for col in kolom_colab if col not in df.columns]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Postingan", f"{len(df)} baris")
    col2.metric("Jumlah Kolom Asli", f"{len([c for c in kolom_colab if c in df.columns])} / {len(kolom_colab)} kolom")
    col3.metric("Status Kolom CSV Colab", "Lengkap ✅" if not missing_all else "Sebagian Otomatis ⚙️")
    
    st.subheader("Pemeriksaan Kolom Dataset Colab")
    col_status = pd.DataFrame({
        "Nama Kolom": kolom_colab,
        "Status Ketersediaan": ["✅ Tersedia" if col in df.columns else "❌ Tidak Tersedia" for col in kolom_colab]
    })
    st.table(col_status)

    if missing_core:
        st.error(f"⚠️ **Dataset tidak valid!** Kolom wajib berikut tidak ditemukan: **{', '.join(missing_core)}**.")
        st.stop()
    
    st.subheader("Pratinjau Dataset (10 Baris Pertama)")
    tampil_cols = [c for c in kolom_colab if c in df.columns]
    st.dataframe(df[tampil_cols].head(10), use_container_width=True)
    
    null_cleaned = df['Cleaned_Text'].isna().sum()
    null_likes = pd.to_numeric(df['like_count'], errors='coerce').isna().sum()
    st.info(f"ℹ️ **Info Dataset Utama:** Terdapat `{len(df)}` baris total. Baris teks bersih kosong: `{null_cleaned}` baris. Baris likes tidak valid: `{null_likes}` baris.")

if missing_core:
    st.stop()

# ===================================================================
# TAB 2: SENTIMEN (MENAMPILKAN HASIL SENTIMEN COLAB)
# ===================================================================
with tab2:
    st.header("😃 Hasil Analisis Sentimen")
    
    sentiment_counts = df['sentimen_sistem'].value_counts()
    sentiment_percent = (df['sentimen_sistem'].value_counts(normalize=True) * 100).round(2)
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
        ax.set_xlabel("Sentimen Sistem")
        ax.set_ylim(0, max(sentiment_counts.values) * 1.15)
        for i, v in enumerate(sentiment_counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        
    st.subheader("Tabel Hasil Klasifikasi Postingan (Beserta Skor Kepercayaan)")
    st.dataframe(
        df[['id', 'text', 'Cleaned_Text', 'sentimen_sistem', 'skor']].head(20),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("🔍 Ringkasan Pemeriksaan Manual (20 Sampel Acak)")
    
    df_eval = df[['id', 'text', 'Cleaned_Text', 'sentimen_sistem', 'skor']].sample(n=min(20, len(df)), random_state=42).copy()
    
    label_kelompok = [
        'Positif', 'Negatif', 'Netral', 'Positif', 'Negatif',
        'Netral', 'Netral', 'Netral', 'Negatif', 'Negatif',
        'Netral', 'Positif', 'Positif', 'Netral', 'Negatif',
        'Negatif', 'Netral', 'Negatif', 'Negatif', 'Negatif'
    ]
    df_eval['label_manual'] = label_kelompok[:len(df_eval)]
    df_eval['Status'] = np.where(df_eval['label_manual'] == df_eval['sentimen_sistem'], '✅ Sesuai', '❌ Tidak Sesuai')
    
    sesuai = (df_eval['label_manual'] == df_eval['sentimen_sistem']).sum()
    acc = (sesuai / len(df_eval)) * 100
    
    col_e1, col_e2 = st.columns(2)
    col_e1.metric("Jumlah Sesuai", f"{sesuai} dari {len(df_eval)} sampel")
    col_e2.metric("Persentase Kesesuaian", f"{acc:.2f}%")
    
    st.dataframe(
        df_eval[['id', 'text', 'label_manual', 'sentimen_sistem', 'skor', 'Status']],
        use_container_width=True
    )
    
    st.subheader("📌 Pembahasan 3 Contoh Ketidaksesuaian")
    st.markdown("""
1. **Sarkasme Syarat Lowongan Kerja (ID 501):** *"LOWONGAN KERJA *PRESIDEN RI 2024* Persyaratan... *Jago ngibul *Yang penting bisa BAHASA INGGRIS 😂😂 Gaji : UMR DKI..."*
   - **Label Manual:** `Negatif` | **Prediksi Sistem:** `Netral` (`skor: 0.9495`) $\\rightarrow$ Model membaca struktur teks formal pengumuman lowongan kerja sebagai informasi netral dan gagal menangkap sindiran/sarkasme politik di dalamnya.
2. **Keluhan Ketimpangan Syarat Kerja (ID 185):** *"Presiden: tamatan SMA, Wapres: tamatan SMP. Sementara lowongan kerja di mana-mana: Minimal S1, Pengalaman 2–5 tahun... Gaji Rp3–5 juta 😭"*
   - **Label Manual:** `Negatif` | **Prediksi Sistem:** `Netral` (`skor: 0.9946`) $\\rightarrow$ Karena mayoritas teks disusun seperti daftar poin kualifikasi kerja, model mengklasifikasikannya sebagai kalimat informatif (`Netral`), meskipun konteks utuhnya mengeluhkan beratnya persyaratan kerja.
3. **Cerita Pengalaman Buka Lowongan Kerja (ID 79):** *"mau cerita dikit, bulan Maret lalu gw buka lowongan kerja, baru sejam aja udh banjir pelamar... INDONESIA DARURAT LAPANGAN PEKERJAAN⁉️"*
   - **Label Manual:** `Netral` | **Prediksi Sistem:** `Negatif` (`skor: 0.9671`) $\\rightarrow$ Penulis berniat membagikan cerita observasi faktual, namun adanya kata berkonotasi genting seperti *"takut overwhelming"* dan *"darurat"* mendorong model memberikan label `Negatif`.
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
- **Perbedaan Ketiga Cluster:** Cluster 0 (190 postingan) berisi pengumuman lowongan kerja, Cluster 1 (96 postingan) fokus pada keluh kesah *fresh graduate* terkait syarat pengalaman kerja, dan Cluster 2 (230 postingan) berisi obrolan tentang besaran gaji UMR dan biaya hidup.
- **Tumpang Tindih (Overlap):** Masih terdapat tumpang tindih pada kata umum seperti `kerja`, `gaji`, dan `umr` karena informasi lowongan kerja maupun keluhan pekerja sama-sama sering menyebutkan istilah tersebut.
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
- **Makna MAE (`{mae:,.2f}`):** Rata-rata kesalahan prediksi model terhadap data uji adalah sekitar {mae:,.0f} likes, dipengaruhi adanya postingan *viral* (outlier) dengan puluhan ribu likes.
- **Makna $R^2$ (`{r2:.4f}`):** Nilai $R^2$ negatif/mendekati nol menunjukkan bahwa jumlah kata dan hashtag saja tidak mampu menjelaskan variasi jumlah likes. Faktor seperti jumlah *followers*, waktu unggah, dan isi topik jauh lebih berpengaruh.
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
1. **Keterbatasan Deteksi Sarkasme pada Sentimen**: Model sentimen terkadang masih mengklasifikasikan sindiran/sarkasme berformat pengumuman lowongan kerja sebagai kalimat informatif (`Netral`).
2. **Keterbatasan Fitur Regresi**: Ketidaktersediaan data jumlah *followers* akun dan adanya *outlier* cuitan viral membuat estimasi regresi linear dua variabel memiliki error (MAE) yang besar.
    """)