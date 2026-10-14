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
        st.pyplot(fig, clear_figure=True)
        
    st.subheader("Tabel Hasil Klasifikasi Postingan (Beserta Skor Kepercayaan)")
    st.dataframe(
        df[['id', 'text', 'Cleaned_Text', 'sentimen_sistem', 'skor']].head(20),
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("🔍 Ringkasan Pemeriksaan Manual")
    
    # Fitur Upload File Excel (.xlsx) atau CSV (.csv)
    uploaded_eval = st.file_uploader(
        "📂 Upload file Hasil Komparasi Sentimen (.xlsx / .csv):", 
        type=["xlsx", "xls", "csv"], 
        key="upload_eval_komparasi"
    )
    
    df_eval = None
    
    # 1. Cek apakah user sudah mengunggah file
    if uploaded_eval is not None:
        try:
            if uploaded_eval.name.endswith('.csv'):
                df_eval = pd.read_csv(uploaded_eval)
            else:
                df_eval = pd.read_excel(uploaded_eval)
            st.success(f"✅ Menampilkan data dari file upload: **{uploaded_eval.name}**")
        except Exception as e:
            st.error(f"❌ Gagal membaca file yang di-upload: {e}")
            
    # 2. Jika belum di-upload, cek apakah ada file bawaan lokal di folder dataset/
    else:
        for path_lokal in [
            "dataset/hasil_komparasi_sentimen.xlsx",
            "dataset/hasil_komparasi_sentimen (2) (1).xlsx",
            "dataset/evaluasi_sentimen_manual.xlsx"
        ]:
            try:
                df_eval = pd.read_excel(path_lokal)
                st.info(f"ℹ️ Menggunakan file bawaan lokal: **{path_lokal}**")
                break
            except Exception:
                pass

    # 3. Tampilkan data HANYA JIKA df_eval berhasil dimuat (dari upload atau file lokal)
    kolom_wajib = ['id', 'text', 'label_manual', 'sentimen_sistem', 'skor']
    
    if df_eval is not None:
        # Validasi kolom wajib
        if all(kol in df_eval.columns for kol in kolom_wajib):
            # Pastikan kolom Status terisi dan akurat
            if 'Status' not in df_eval.columns:
                df_eval['Status'] = np.where(df_eval['label_manual'] == df_eval['sentimen_sistem'], '✅ Sesuai', '❌ Tidak Sesuai')
            
            # Hitung metrik kesesuaian berdasarkan isi tabel
            sesuai = (df_eval['label_manual'] == df_eval['sentimen_sistem']).sum()
            acc = (sesuai / len(df_eval)) * 100 if len(df_eval) > 0 else 0
            
            col_e1, col_e2 = st.columns(2)
            col_e1.metric("Jumlah Sesuai", f"{sesuai} dari {len(df_eval)} sampel")
            col_e2.metric("Persentase Kesesuaian", f"{acc:.2f}%")
            
            st.dataframe(
                df_eval[['id', 'text', 'label_manual', 'sentimen_sistem', 'skor', 'Status']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(f"⚠️ Kolom pada file tidak lengkap. Pastikan file memiliki kolom: {', '.join(kolom_wajib)}")
    else:
        # Jika belum ada file yang di-upload dan tidak ada file lokal, tampilkan pesan panduan
        st.info("👆 Silakan upload file hasil komparasi sentimen (.xlsx / .csv) di atas untuk melihat ringkasan pemeriksaan manual dan metrik kesesuaian.")

# TAB 3: CLUSTERING (MENAMPILKAN HASIL CLUSTER)
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
with tab4:
    st.header("📈 Regresi Linear (Estimasi Jumlah Likes)")
    
    # 1. Konversi tipe data ke numerik (Target y + Fitur X)
    df['like_count_num'] = pd.to_numeric(df['like_count'], errors='coerce')
    df['reply_count_num'] = pd.to_numeric(df['reply_count'], errors='coerce')
    df['retweet_count_num'] = pd.to_numeric(df['retweet_count'], errors='coerce')
    
    # Bersihkan baris yang memiliki nilai kosong (NaN) pada ketiga kolom tersebut
    kolom_wajib = ['like_count_num', 'reply_count_num', 'retweet_count_num']
    df_reg_clean = df.dropna(subset=kolom_wajib).copy()
    dropped_reg = len(df) - len(df_reg_clean)
    
    st.caption(
        f"*Catatan: Menggunakan {len(df_reg_clean):,} baris valid "
        f"({dropped_reg:,} baris dikeluarkan karena data likes, reply, atau retweet kosong/tidak valid).*"
    )
    
    # 2. Pastikan jumlah baris cukup untuk di-split
    if len(df_reg_clean) < 10:
        st.error("⚠️ Data valid terlalu sedikit (kurang dari 10 baris) untuk menjalankan model regresi.")
    else:
        # X = Reply & Retweet, y = Likes
        X_reg = df_reg_clean[['reply_count_num', 'retweet_count_num']]
        y_reg = df_reg_clean['like_count_num']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        model_reg = LinearRegression()
        model_reg.fit(X_train, y_train)
        y_pred = model_reg.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 3. Tampilkan Metrik Evaluasi & Koefisien
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE (Mean Absolute Error)", f"{mae:,.2f}")
        col2.metric("R² Score", f"{r2:.4f}")
        col3.metric("Koef. Reply", f"{model_reg.coef_[0]:,.2f}")
        col4.metric("Koef. Retweet", f"{model_reg.coef_[1]:,.2f}")
        
        col_g1, col_g2 = st.columns([1.3, 1])
        with col_g1:
            st.subheader("Grafik Likes Aktual vs Prediksi")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(y_test, y_pred, color='royalblue', alpha=0.6, edgecolors='w', linewidth=0.5, label='Data Uji')
            
            # Garis Ideal (Aktual = Prediksi)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Garis Ideal (y = ŷ)')
            
            ax.set_xlabel("Likes Aktual")
            ax.set_ylabel("Likes Prediksi")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Mencegah memory leak di Streamlit
            
        with col_g2:
            st.subheader("🧮 Formulir Estimasi Likes")
            with st.form("form_likes"):
                input_reply = st.number_input("Jumlah Reply (Balasan):", min_value=0, max_value=100000, value=10, step=1)
                input_retweet = st.number_input("Jumlah Retweet:", min_value=0, max_value=100000, value=25, step=1)
                submit_btn = st.form_submit_button("Estimasi Likes")
                
                if submit_btn:
                    input_df = pd.DataFrame({
                        'reply_count_num': [input_reply],
                        'retweet_count_num': [input_retweet]
                    })
                    pred_val = model_reg.predict(input_df)[0]
                    
                    st.success(f"📌 Estimasi Jumlah Likes: **{pred_val:,.2f}** likes")
                    if pred_val < 0:
                        st.warning(
                            "⚠️ Catatan: Nilai negatif muncul karena titik potong (intercept) model bernilai negatif "
                            "ketika jumlah reply dan retweet sangat kecil."
                        )
                    else:
                        st.info(f"💡 Dibulatkan menjadi sekitar **{max(0, round(pred_val)):,}** likes.")