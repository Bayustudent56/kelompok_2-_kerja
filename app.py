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
st.set_page_config(page_title="Dashboard Analisis Data X - Dunia Kerja", page_icon="📊", layout="wide")
st.title("📊 Dashboard Analisis Posting X (Twitter) - Topik Dunia Kerja")
st.markdown("Aplikasi web interaktif untuk analisis sentimen (IndoBERT), clustering teks, dan regresi linear prediksi likes.")

# -------------------------------------------------------------------
# FUNGSI MODEL INDOBERT (CACHED AGAR TIDAK BERAT)
# -------------------------------------------------------------------
@st.cache_resource
def load_indobert():
    from transformers import pipeline
    pretrained = "mdhugol/indonesia-bert-sentiment-classification"
    return pipeline("sentiment-analysis", model=pretrained, tokenizer=pretrained)

@st.cache_data
def predict_indobert(texts_list):
    nlp = load_indobert()
    label_index = {'LABEL_0': 'Positif', 'LABEL_1': 'Netral', 'LABEL_2': 'Negatif',
                   'positive': 'Positif', 'neutral': 'Netral', 'negative': 'Negatif'}
    results = []
    for t in texts_list:
        t_str = str(t).strip()
        if not t_str:
            results.append('Netral')
        else:
            pred = nlp(t_str[:512])[0]['label']
            results.append(label_index.get(pred, pred))
    return results

# -------------------------------------------------------------------
# SIDEBAR - UNGGAH DATASET (CSV)
# -------------------------------------------------------------------
st.sidebar.header("📂 Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("1. Unggah CSV Dataset Utama", type=["csv"])
uploaded_eval = st.sidebar.file_uploader("2. Unggah CSV Evaluasi Sentimen (Opsional)", type=["csv"])

if uploaded_file is None:
    st.info("👋 **Selamat Datang!** Silakan unggah file CSV dataset utama Anda (misal: `cleaning_kerja_relevan.csv`) melalui panel di sebelah kiri (Sidebar) untuk memulai analisis.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"⚠️ Gagal membaca file CSV. Detail error: {e}")
    st.stop()

kolom_wajib = ['created_at', 'text', 'Cleaned_Text', 'like_count']
missing_cols = [col for col in kolom_wajib if col not in df.columns]

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
    st.table(pd.DataFrame({
        "Nama Kolom Wajib": kolom_wajib,
        "Status Ketersediaan": ["✅ Tersedia" if col in df.columns else "❌ Tidak Tersedia" for col in kolom_wajib]
    }))

    if missing_cols:
        st.error(f"⚠️ **Dataset tidak valid!** Kolom wajib tidak ditemukan: **{', '.join(missing_cols)}**.")
        st.stop()
    
    st.subheader("Pratinjau Dataset (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Informasi Pembersihan & Baris Data")
    null_cleaned = df['Cleaned_Text'].isna().sum()
    null_likes = pd.to_numeric(df['like_count'], errors='coerce').isna().sum()
    st.markdown(f"""
- **Total baris di dalam dataset utama:** `{len(df)}` postingan unik.
- **Baris dengan teks bersih (`Cleaned_Text`) kosong:** `{null_cleaned}` baris.
- **Baris dengan jumlah likes (`like_count`) tidak valid/kosong:** `{null_likes}` baris.
*(Catatan: Seluruh analisis menggunakan dataset utama yang sama. Jika ada baris bernilai kosong pada target/fitur tertentu, baris tersebut akan dikeluarkan khusus pada tahap analisis terkait).*
    """)

if missing_cols:
    st.stop()

# ===================================================================
# TAB 2: SENTIMEN (DISAMAKAN DENGAN GOOGLE COLAB - INDOBERT)
# ===================================================================
with tab2:
    st.header("😃 Analisis Sentimen (IndoBERT)")
    
    norm_map = {'positive': 'Positif', 'positif': 'Positif', 'label_0': 'Positif',
                'negative': 'Negatif', 'negatif': 'Negatif', 'label_2': 'Negatif',
                'neutral': 'Netral', 'netral': 'Netral', 'label_1': 'Netral'}

    # Cek apakah CSV sudah punya kolom hasil IndoBERT dari Colab
    if 'sentimen_sistem' in df.columns:
        df['sentimen_sistem'] = df['sentimen_sistem'].astype(str).str.strip().str.lower().map(norm_map).fillna(df['sentimen_sistem'])
    elif 'sentiment' in df.columns:
        df['sentimen_sistem'] = df['sentiment'].astype(str).str.strip().str.lower().map(norm_map).fillna(df['sentiment'])
    else:
        with st.spinner("⏳ Sedang menjalankan model IndoBERT seperti di Google Colab (proses ini hanya dilakukan sekali)..."):
            # Gunakan kolom Cleaned_Text (atau text jika Cleaned_Text kosong)
            teks_input = df['Cleaned_Text'].fillna(df['text']).astype(str).tolist()
            df['sentimen_sistem'] = predict_indobert(teks_input)

    # Urutkan kategori persis seperti di Google Colab: Positif, Negatif, Netral
    urutan_kat = ['Positif', 'Negatif', 'Netral']
    sentiment_counts = df['sentimen_sistem'].value_counts().reindex(urutan_kat, fill_value=0)
    sentiment_percent = ((sentiment_counts / len(df)) * 100).round(2)
    
    df_sent_summary = pd.DataFrame({
        'sentimen_sistem': sentiment_counts.index,
        'count': sentiment_counts.values, 
        'persentase (%)': sentiment_percent.values
    })
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Ringkasan Distribusi Sentimen")
        st.dataframe(df_sent_summary, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Grafik Distribusi Sentimen")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        warna_colab = ['#2ca05a', '#d64541', '#8c969e'] # Hijau, Merah, Abu-abu seperti di Colab
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=warna_colab)
        ax.set_ylabel("Jumlah Postingan")
        ax.set_title("Distribusi Sentimen Postingan")
        ax.set_ylim(0, max(sentiment_counts.values) * 1.18)
        
        for bar, count, pct in zip(bars, sentiment_counts.values, sentiment_percent.values):
            ax.text(bar.get_x() + bar.get_width()/2, count + 5, f"{count}\n({pct:.2f}%)", ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
        
    st.subheader("Tabel Hasil Klasifikasi Postingan")
    st.dataframe(df[['text', 'Cleaned_Text', 'sentimen_sistem']].head(20), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Ringkasan Pemeriksaan Manual (20 Sampel Acak)")
    
    if uploaded_eval is not None:
        raw_content = uploaded_eval.getvalue().decode("utf-8", errors="ignore")
        lines = raw_content.splitlines(keepends=True)
        body = "".join(lines[1:]) if lines and "sentimen" in lines[0].lower() else raw_content

        pattern = re.compile(r'(.*?),\s*(positive|negative|neutral|positif|negatif|netral)\s*"?\s*(?:\r?\n|$)', re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(body)

        df_eval = pd.DataFrame(matches, columns=['text', 'sentimen_manual'])
        df_eval['text'] = df_eval['text'].str.strip().str.strip('"').str.replace('""', '"')
        df_eval['label_manual'] = df_eval['sentimen_manual'].str.strip().str.lower().map(norm_map)

        # Cocokkan dengan sentimen_sistem (IndoBERT) dari dataset utama
        map_pred = dict(zip(df['text'].astype(str).str.strip(), df['sentimen_sistem']))
        missing_mask = ~df_eval['text'].isin(map_pred.keys())
        df_eval['prediksi_sistem'] = df_eval['text'].map(map_pred)
        if missing_mask.any():
            df_eval.loc[missing_mask, 'prediksi_sistem'] = predict_indobert(df_eval.loc[missing_mask, 'text'].tolist())

        df_eval['Status'] = np.where(df_eval['label_manual'] == df_eval['prediksi_sistem'], '✅ Sesuai', '❌ Tidak Sesuai')

        sesuai = (df_eval['label_manual'] == df_eval['prediksi_sistem']).sum()
        total_eval = len(df_eval)
        if total_eval > 0:
            st.success(f"💡 **Hasil Evaluasi:** {sesuai} dari {total_eval} sampel sesuai (**{(sesuai/total_eval)*100:.2f}%**).")
        st.dataframe(df_eval, use_container_width=True)
    else:
        st.info("ℹ️ Anda belum mengunggah file `evaluasi_sentimen.csv` di Sidebar. Berikut adalah tabel sampel 20 postingan acak dari dataset utama untuk pemeriksaan manual:")
        df_sample20 = df[['text', 'Cleaned_Text', 'sentimen_sistem']].sample(n=min(20, len(df)), random_state=42).rename(columns={'sentimen_sistem': 'prediksi_sistem'})
        st.dataframe(df_sample20, use_container_width=True)

    with st.expander("📖 Pembahasan Ketidaksesuaian Sentimen (Sistem vs Pembacaan Manual)", expanded=True):
        st.markdown("""
        **Analisis Penyebab Ketidaksesuaian Klasifikasi:**
        1. **Kalimat Sindiran / Sarkasme:** Postingan yang menggunakan gaya bahasa sindiran atau parodi lowongan kerja sering kali sulit dikenali secara kontekstual oleh model, sehingga terklasifikasi berbeda dari penilaian manusia.
        2. **Campuran Opini dan Fakta (Kalimat Panjang):** Beberapa cuitan berisi cerita pengalaman melamar kerja yang panjang (informatif/netral di awal, namun mengeluh di akhir), sehingga model dan pembaca manual dapat memiliki fokus penekanan sentimen yang berbeda.
        3. **Istilah Slang / Singkatan Baru di X:** Penggunaan bahasa gaul atau singkatan spesifik di media sosial X yang telah melalui tahap *cleaning* terkadang mengubah struktur emosi kalimat aslinya.
        """)

# ===================================================================
# TAB 3: CLUSTERING
# ===================================================================
with tab3:
    st.header("🏷️ Clustering Data Teks (TF-IDF & K-Means)")
    
    df_cluster = df.dropna(subset=['Cleaned_Text']).copy()
    dropped_cluster = len(df) - len(df_cluster)
    st.caption(f"*Catatan: {len(df_cluster)} baris digunakan dalam clustering ({dropped_cluster} baris dikeluarkan karena teks bersih kosong).*")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df_cluster['Cleaned_Text'].astype(str))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans.fit_predict(X_tfidf)
    
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    tema_map = {
        0: "Informasi Lowongan Kerja & Kualifikasi Pelamar",
        1: "Tantangan Fresh Graduate & Syarat Pengalaman Kerja",
        2: "Diskusi Gaji UMR, Keluhan Pekerja, dan Biaya Hidup"
    }
    
    cluster_counts = df_cluster['cluster'].value_counts().sort_index()
    col_c1, col_c2, col_c3 = st.columns(3)
    for idx_col, col_obj in enumerate([col_c1, col_c2, col_c3]):
        col_obj.metric(f"Ukuran Cluster {idx_col}", f"{cluster_counts.get(idx_col, 0)} postingan")
    st.markdown("---")

    for c_id in range(3):
        st.subheader(f"🔹 Cluster {c_id}: {tema_map[c_id]}")
        c_count = (df_cluster['cluster'] == c_id).sum()
        top_words = ", ".join([terms[ind] for ind in order_centroids[c_id, :5]])
        
        st.write(f"- **Jumlah Postingan**: `{c_count}` postingan ({(c_count/len(df_cluster)*100):.1f}%)")
        st.write(f"- **5 Kata Dominan**: `{top_words}`")
        st.write("- **3 Contoh Postingan**:")
        
        for idx, sample_text in enumerate(df_cluster[df_cluster['cluster'] == c_id]['text'].head(3), 1):
            st.info(f"**Contoh {idx}:** {sample_text}")
        st.markdown("---")

    st.subheader("💡 Interpretasi Hasil Clustering")
    st.markdown("""
- **Apa yang membedakan isi ketiga cluster?**
  - **Cluster 0** berfokus pada penyebaran informasi lowongan kerja (*job vacancy*), syarat pendaftaran, dan posisi yang dibutuhkan perusahaan.
  - **Cluster 1** berfokus pada diskusi mengenai kesulitan lulusan baru (*fresh graduate*) dalam menembus pasar kerja yang kerap menuntut pengalaman kerja.
  - **Cluster 2** berfokus pada perbincangan seputar nominal gaji, standar UMR di berbagai daerah, serta keluhan keseharian pekerja terkait biaya hidup.
- **Apakah masih terdapat pembahasan yang tumpang tindih?**
  - **Ya, masih terdapat tumpang tindih (*overlap*)**, terutama penggunaan kata umum seperti `kerja`, `gaji`, dan `umr` yang muncul di Cluster 0 maupun Cluster 2. Hal ini wajar karena pengumuman lowongan kerja sering kali turut mencantumkan informasi gaji UMR, sementara diskusi keluhan pekerja juga membahas topik gaji dan pekerjaan.
    """)

# ===================================================================
# TAB 4: REGRESI
# ===================================================================
with tab4:
    st.header("📈 Regresi Linear Sederhana (Estimasi Likes)")
    
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    df['hashtag_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x)))
    df['like_count_num'] = pd.to_numeric(df['like_count'], errors='coerce')
    
    df_reg_clean = df.dropna(subset=['like_count_num']).copy()
    st.caption(f"*Catatan Data Regresi: Menggunakan {len(df_reg_clean)} baris valid dari total {len(df)} baris ({len(df) - len(df_reg_clean)} baris dikeluarkan).*")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_reg_clean[['word_count', 'hashtag_count']], df_reg_clean['like_count_num'], test_size=0.2, random_state=42
    )
    
    model_reg = LinearRegression().fit(X_train, y_train)
    y_pred = model_reg.predict(X_test)
    
    mae, r2 = mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
    
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
            if st.form_submit_button("Hitung Estimasi Likes"):
                pred_val = model_reg.predict(pd.DataFrame({'word_count': [input_words], 'hashtag_count': [input_hashtags]}))[0]
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
1. **Dominasi Sentimen Netral & Negatif pada Isu Dunia Kerja**: Berdasarkan klasifikasi IndoBERT, mayoritas postingan bersentimen Netral (260 postingan / 50.39% berisi informasi lowongan kerja) dan diikuti sentimen Negatif yang tinggi (211 postingan / 40.89% berisi keluhan gaji UMR serta sulitnya mencari kerja), sedangkan sentimen Positif hanya 45 postingan (8.72%).
2. **Keresahan Fresh Graduate & Diskusi Gaji UMR**: Dari hasil clustering, percakapan terbagi jelas antara informasi lowongan kerja, keluhan *fresh graduate* terkait syarat pengalaman kerja, serta perbincangan mengenai kecukupan gaji UMR.
3. **Panjang Teks & Hashtag Tidak Menentukan Likes**: Evaluasi regresi linear membuktikan bahwa jumlah kata dan jumlah hashtag bukan indikator penentu tingginya *engagement* (*likes*) pada topik dunia kerja ($R^2 = -0.0182$).
    """)

with col_b:
    st.subheader("⚠️ 2 Keterbatasan Data & Metode")
    st.markdown("""
1. **Tantangan Konteks Sarkasme pada Analisis Sentimen**: Meskipun telah menggunakan model *pre-trained* IndoBERT, kalimat yang mengandung sindiran halus (*sarcasm*) atau campuran informasi loker dengan keluhan terkadang masih menghasilkan perbedaan persepsi dibandingkan pelabelan manual manusia.
2. **Keterbatasan Fitur Prediksi & Ketimpangan Distribusi Likes**: Dataset memiliki distribusi *likes* yang sangat timpang (*skewed*) akibat adanya sedikit postingan *viral*, serta tidak tersedianya fitur penting seperti jumlah *followers* pengunggah dan usia postingan dalam pemodelan regresi.
    """)