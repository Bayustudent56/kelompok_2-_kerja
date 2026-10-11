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
# SIDEBAR - PENGATURAN DATASET
# -------------------------------------------------------------------
st.sidebar.header("📂 Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("Unggah CSV Dataset", type=["csv"])

@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("cleaning_kerja_relevan.csv")
    except Exception:
        return None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()
    if df is not None:
        st.sidebar.info("Menggunakan dataset default (`cleaning_kerja_relevan.csv`).")

# -------------------------------------------------------------------
# PEMERIKSAAN KOLOM WAJIB
# -------------------------------------------------------------------
kolom_wajib = ['text', 'Cleaned_Text', 'like_count']

if df is None:
    st.error("⚠️ Dataset belum diunggah atau file default tidak ditemukan!")
    st.stop()

missing_cols = [col for col in kolom_wajib if col not in df.columns]
if missing_cols:
    st.error(f"⚠️ Dataset tidak valid. Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")
    st.stop()

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
    col1.metric("Total Postingan", len(df))
    col2.metric("Jumlah Kolom", len(df.columns))
    col3.metric("Kolom Wajib", "Lengkap ✅" if not missing_cols else "Kurang ❌")
    
    st.subheader("Status Kolom Wajib")
    col_status = {col: ("Tersedia" if col in df.columns else "Tidak Ada") for col in kolom_wajib}
    st.json(col_status)
    
    st.subheader("Sample Dataset (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)

# ===================================================================
# TAB 2: SENTIMEN
# ===================================================================
with tab2:
    st.header("😃 Analisis Sentimen")
    
    pos_words = set(['gaji', 'bagus', 'lulus', 'alhamdulillah', 'peluang', 'tertarik', 'semangat', 'bismillah', 'gratis', 'info', 'sesuai', 'rekomendasi', 'terbaru', 'welcome', 'mampu'])
    neg_words = set(['sedih', 'scam', 'sulit', 'turun', 'sakit', 'ngos', 'burnout', 'layoff', 'phk', 'overwork', 'pengangguran', 'susah', 'minimal', 'rugi', 'kecewa', 'revisi'])

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

    df['sentiment'] = df['Cleaned_Text'].apply(get_sentiment)
    
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percent = (df['sentiment'].value_counts(normalize=True) * 100).round(2)
    df_sent_summary = pd.DataFrame({'Jumlah': sentiment_counts, 'Persentase (%)': sentiment_percent})
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ringkasan Sentimen")
        st.dataframe(df_sent_summary)
    
    with col2:
        st.subheader("Grafik Distribusi Sentimen")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', ax=ax)
        ax.set_ylabel("Jumlah Postingan")
        ax.set_xlabel("Sentimen")
        st.pyplot(fig)
        
    st.subheader("Hasil Klasifikasi Postingan")
    st.dataframe(df[['text', 'Cleaned_Text', 'sentiment']].head(20), use_container_width=True)
    
    st.subheader("Pemeriksaan Manual Sampel (20 Data)")
    try:
        df_eval = pd.read_csv("evaluasi_sentimen.csv")
        st.dataframe(df_eval.head(20), use_container_width=True)
        if 'label_manual' in df_eval.columns and 'prediksi_sistem' in df_eval.columns:
            acc = (df_eval['label_manual'] == df_eval['prediksi_sistem']).mean() * 100
            st.info(f"💡 Persentase Kesesuaian Label Manual (20 Sampel): **{acc:.2f}%**")
    except Exception:
        st.warning("File `evaluasi_sentimen.csv` belum ditemukan untuk ringkasan manual.")

# ===================================================================
# TAB 3: CLUSTERING
# ===================================================================
with tab3:
    st.header("🏷️ Clustering Data Teks (K-Means)")
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['Cleaned_Text'].fillna(''))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_tfidf)
    
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    tema_map = {
        0: "Tantangan Fresh Graduate & Pengalaman Kerja",
        1: "Informasi Lowongan Kerja Regional / BUMN",
        2: "Diskusi Gaji UMR, Keluhan, dan Biaya Hidup"
    }
    
    for c_id in range(3):
        st.subheader(f"🔹 Cluster {c_id}: {tema_map[c_id]}")
        c_count = (df['cluster'] == c_id).sum()
        top_words = ", ".join([terms[ind] for ind in order_centroids[c_id, :5]])
        
        st.write(f"- **Ukuran Cluster**: {c_count} postingan")
        st.write(f"- **5 Kata Dominan**: `{top_words}`")
        
        st.write("- **3 Contoh Postingan**:")
        samples = df[df['cluster'] == c_id]['text'].head(3).tolist()
        for idx, sample_text in enumerate(samples, 1):
            st.caption(f"  {idx}. {sample_text}")
        st.markdown("---")

# ===================================================================
# TAB 4: REGRESI
# ===================================================================
with tab4:
    st.header("📈 Regresi Linear (Estimasi Likes)")
    
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['hashtag_count'] = df['text'].apply(lambda x: len(re.findall(r'#\w+', str(x))))
    
    df_reg_clean = df.dropna(subset=['like_count']).copy()
    
    X_reg = df_reg_clean[['word_count', 'hashtag_count']]
    y_reg = df_reg_clean['like_count']
    
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    model_reg = LinearRegression()
    model_reg.fit(X_train, y_train)
    y_pred = model_reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
    col2.metric("R² Score", f"{r2:.4f}")
    
    st.subheader("Grafik Likes Aktual vs Hasil Prediksi")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Data Uji')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal')
    ax.set_xlabel("Likes Aktual")
    ax.set_ylabel("Likes Prediksi")
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Formulir Estimasi Jumlah Likes")
    with st.form("form_likes"):
        input_words = st.number_input("Jumlah Kata dalam Postingan", min_value=1, max_value=200, value=20)
        input_hashtags = st.number_input("Jumlah Hashtag", min_value=0, max_value=20, value=1)
        submit_btn = st.form_submit_button("Estimasi Likes")
        
        if submit_btn:
            pred_val = model_reg.predict([[input_words, input_hashtags]])[0]
            st.success(f"Estimasi Jumlah Likes: **{pred_val:.2f}** likes")
            if pred_val < 0:
                st.warning("⚠️ Catatan: Hasil negatif menunjukkan keterbatasan model regresi linear dua variabel.")

# ===================================================================
# BAGIAN AKHIR: TEMUAN & KETERBATASAN
# ===================================================================
st.markdown("---")
st.header("📌 Kesimpulan Analisis")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("3 Temuan Utama")
    st.markdown("""
    1. **Isu Gaji UMR Mendominasi**: Percakapan terbanyak membahas ketidaksesuaian gaji UMR dengan biaya hidup.
    2. **Tantangan Fresh Graduate**: Banyak keluhan mengenai syarat lamaran kerja yang mensyaratkan pengalaman untuk posisi pemula.
    3. **Pengaruh Kata & Hashtag Kecil**: Jumlah kata dan hashtag saja tidak cukup kuat untuk memprediksi jumlah likes secara akurat.
    """)

with col_b:
    st.subheader("2 Keterbatasan Data / Metode")
    st.markdown("""
    1. **Fitur Regresi Terbatas**: Jumlah likes di media sosial lebih dipengaruhi oleh jangkauan akun, jam posting, dan keviralan konten.
    2. **Lexicon Sentimen Sederhana**: Penggunaan daftar kata kunci belum sepenuhnya menangkap sarkasme atau konteks bahasa gaul.
    """)