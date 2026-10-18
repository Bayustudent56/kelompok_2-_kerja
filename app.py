import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
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
st.sidebar.header(" Unggah Data")
uploaded_file = st.sidebar.file_uploader("Unggah CSV Dataset Hasil Colab", type=["csv"])

if uploaded_file is None:
    st.info(" **Selamat Datang!** Silakan unggah file CSV hasil pemrosesan Google Colab (`hasil_sentimen_lengkap.csv`) melalui panel di sebelah kiri (Sidebar) untuk menampilkan dashboard.")
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

    # Clustering diproses di Tab 3 agar dapat memakai label dari hasil Colab.

    # 3. Cek kolom fitur regresi
    if 'word_count' not in df.columns:
        df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    if 'hashtag_count' not in df.columns:
        df['hashtag_count'] = df['text'].astype(str).apply(lambda x: len(re.findall(r'#\w+', x)))

# -------------------------------------------------------------------
# TAB APLIKASI
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([" Data", " Sentimen", " Clustering", " Regresi"])

# ===================================================================
# TAB 1: DATA
with tab1:
    st.header(" Pratinjau & Ringkasan Dataset")
    
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
    st.dataframe(df[tampil_cols].head(10), width="stretch")
    
    null_cleaned = df['Cleaned_Text'].isna().sum()
    null_likes = pd.to_numeric(df['like_count'], errors='coerce').isna().sum()
    st.info(f"ℹ️ **Info Dataset Utama:** Terdapat `{len(df)}` baris total. Baris teks bersih kosong: `{null_cleaned}` baris. Baris likes tidak valid: `{null_likes}` baris.")

if missing_core:
    st.stop()

# ===================================================================
# TAB 2: SENTIMEN (MENAMPILKAN HASIL SENTIMEN COLAB)
with tab2:
    st.header(" Hasil Analisis Sentimen")
    
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
        st.dataframe(df_sent_summary, width="stretch", hide_index=True)
    
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
    kolom_hasil_sentimen = [c for c in ['id', 'text', 'Cleaned_Text', 'sentimen_sistem', 'skor'] if c in df.columns]
    st.dataframe(
        df[kolom_hasil_sentimen].head(20),
        width="stretch"
    )
    
    st.markdown("---")
    st.subheader(" Ringkasan Pemeriksaan Manual")
    
    # Fitur Upload File Excel (.xlsx) atau CSV (.csv)
    uploaded_eval = st.file_uploader(
        "Upload file Hasil Komparasi Sentimen (.xlsx / .csv):", 
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
            

    # 2. Tampilkan data HANYA JIKA df_eval berhasil dimuat (dari upload atau file lokal)
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
                width="stretch",
                hide_index=True
            )
        else:
            st.warning(f"⚠️ Kolom pada file tidak lengkap. Pastikan file memiliki kolom: {', '.join(kolom_wajib)}")
    

# TAB 3: CLUSTERING (VISUALISASI + INTERPRETASI)
with tab3:
    st.header("🏷️ Analisis Clustering Percakapan Dunia Kerja")
    st.caption("Pengelompokan teks menggunakan TF-IDF dan K-Means dengan 3 cluster.")

    uploaded_cluster = st.file_uploader(
        "Opsional: unggah kerja_clustering.csv agar label cluster sama dengan hasil Colab",
        type=["csv"],
        key="upload_kerja_clustering"
    )

    # Salinan khusus untuk clustering agar bagian tab lain tidak berubah.
    df_cluster = df.copy()
    pakai_label_colab = False

    if uploaded_cluster is not None:
        try:
            df_colab = pd.read_csv(uploaded_cluster)
            kolom_teks_cluster = next(
                (c for c in ["text", "full_text", "teks_asli"] if c in df_colab.columns),
                None
            )
            kolom_label_cluster = next(
                (c for c in ["cluster", "Cluster"] if c in df_colab.columns),
                None
            )
            if kolom_teks_cluster is None or kolom_label_cluster is None:
                st.warning("File clustering harus memiliki kolom teks (text/full_text) dan cluster.")
            else:
                # Normalisasi teks untuk pencocokan yang lebih tahan terhadap spasi dan kapitalisasi.
                df_cluster["_key_cluster"] = df_cluster["text"].fillna("").astype(str).str.strip().str.lower()
                df_colab["_key_cluster"] = df_colab[kolom_teks_cluster].fillna("").astype(str).str.strip().str.lower()
                label_map = (df_colab.drop_duplicates("_key_cluster")
                             .set_index("_key_cluster")[kolom_label_cluster].to_dict())
                label_terpasang = df_cluster["_key_cluster"].map(label_map)
                jumlah_cocok = int(label_terpasang.notna().sum())
                if jumlah_cocok > 0:
                    df_cluster["cluster"] = label_terpasang
                    df_cluster["cluster"] = pd.to_numeric(df_cluster["cluster"], errors="coerce")
                    df_cluster = df_cluster[df_cluster["cluster"].notna()].copy()
                    df_cluster["cluster"] = df_cluster["cluster"].astype(int)
                    pakai_label_colab = True
                    st.success(f"Label cluster dari Colab berhasil dicocokkan untuk {jumlah_cocok} postingan.")
                    if jumlah_cocok < len(df):
                        st.info(f"{len(df) - jumlah_cocok} postingan tidak memiliki pasangan teks di file clustering dan tidak disertakan pada visualisasi ini.")
                else:
                    st.warning("Tidak ada teks yang cocok antara dataset utama dan file clustering. K-Means akan dihitung ulang untuk tab ini.")
        except Exception as e:
            st.warning(f"File clustering gagal dibaca ({e}). K-Means akan dihitung ulang untuk tab ini.")

    # Teks yang digunakan: prioritaskan Cleaned_Text; fallback ke teks asli.
    kolom_teks_model = "Cleaned_Text" if "Cleaned_Text" in df_cluster.columns else "text"
    teks_model = df_cluster[kolom_teks_model].fillna("").astype(str).str.strip()
    valid_teks = teks_model.ne("")
    df_cluster = df_cluster.loc[valid_teks].copy()
    teks_model = teks_model.loc[valid_teks]

    if len(df_cluster) < 3:
        st.warning("Data teks valid terlalu sedikit untuk menjalankan clustering. Periksa file CSV yang diunggah.")
    else:
        # Stopword umum Bahasa Indonesia; negasi penting tetap dipertahankan.
        stopwords_id = {
            "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada", "adalah",
            "atau", "karena", "jadi", "saat", "dalam", "akan", "ada", "juga", "lebih", "sudah", "belum",
            "bisa", "dapat", "agar", "para", "oleh", "sebagai", "nya", "sih", "dong", "nih", "deh", "lah",
            "aja", "aku", "saya", "kamu", "dia", "mereka", "kami", "kita", "rt", "https", "http", "co"
        }
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=list(stopwords_id),
            min_df=1
        )
        try:
            X_tfidf = vectorizer.fit_transform(teks_model)
        except ValueError:
            st.error("Teks tidak memiliki kosakata yang cukup untuk clustering. Periksa kolom teks bersih.")
            st.stop()

        if not pakai_label_colab:
            model_kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=42)
            df_cluster["cluster"] = model_kmeans.fit_predict(X_tfidf)
            st.info("K-Means dihitung di Streamlit. Untuk mempertahankan label persis dari Colab, unggah kerja_clustering.csv.")

        # Pastikan label numerik dan tampilkan hanya cluster yang benar-benar ada.
        df_cluster["cluster"] = pd.to_numeric(df_cluster["cluster"], errors="coerce")
        df_cluster = df_cluster[df_cluster["cluster"].notna()].copy()
        df_cluster["cluster"] = df_cluster["cluster"].astype(int)
        cluster_ids = sorted(df_cluster["cluster"].unique().tolist())
        if not cluster_ids:
            st.warning("Tidak ditemukan label cluster yang valid.")
            st.stop()

        # Hitung kata dominan berdasarkan rata-rata bobot TF-IDF dalam setiap cluster.
        terms = vectorizer.get_feature_names_out()
        teks_index = df_cluster.index.tolist()
        # X_tfidf dan df_cluster tetap sebaris setelah validasi teks; gunakan posisi yang selaras.
        if X_tfidf.shape[0] != len(df_cluster):
            # Bangun ulang matriks agar selalu selaras jika beberapa teks kosong terbuang.
            X_tfidf = vectorizer.fit_transform(df_cluster[kolom_teks_model].fillna("").astype(str))
            terms = vectorizer.get_feature_names_out()
        top_words_map = {}
        for cid in cluster_ids:
            positions = np.flatnonzero(df_cluster["cluster"].to_numpy() == cid)
            if len(positions):
                mean_scores = np.asarray(X_tfidf[positions].mean(axis=0)).ravel()
                top_idx = mean_scores.argsort()[::-1]
                top_words_map[cid] = [(terms[i], float(mean_scores[i])) for i in top_idx if mean_scores[i] > 0][:10]
            else:
                top_words_map[cid] = []

        cluster_counts = df_cluster["cluster"].value_counts().sort_index()
        total_cluster = len(df_cluster)
        st.subheader("Ringkasan Dataset Clustering")
        metrics = st.columns(3)
        metrics[0].metric("Total postingan dianalisis", f"{total_cluster:,}")
        metrics[1].metric("Jumlah cluster ditemukan", str(len(cluster_ids)))
        metrics[2].metric("Metode", "TF-IDF + K-Means")

        # Tema awal berbasis kata dominan; perlu diverifikasi dengan contoh teks.
        def tema_otomatis(cid, words):
            gabung = " ".join(w for w, _ in words).lower()
            if any(k in gabung for k in ["umr", "gaji", "biaya hidup", "upah"]):
                return "Diskusi Gaji, UMR, dan Biaya Hidup"
            if any(k in gabung for k in ["lowongan", "posisi", "rekrutmen", "pt", "lamaran"]):
                return "Informasi Lowongan Kerja dan Rekrutmen"
            if any(k in gabung for k in ["fresh graduate", "pengalaman", "lulusan", "sarjana"]):
                return "Tantangan Fresh Graduate dan Pengalaman Kerja"
            return f"Tema Cluster {cid} (perlu ditinjau)"

        tema_map = {cid: tema_otomatis(cid, top_words_map.get(cid, [])) for cid in cluster_ids}
        # Distribusi jumlah dan persentase cluster.
        st.markdown("---")
        st.subheader("Distribusi Anggota Cluster")
        distribusi = pd.DataFrame({
            "Cluster": [f"Cluster {cid}" for cid in cluster_counts.index],
            "Jumlah postingan": cluster_counts.values,
            "Persentase (%)": (cluster_counts.values / total_cluster * 100).round(2)
        })
        col_chart, col_table = st.columns([1.6, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bars = ax.bar(distribusi["Cluster"], distribusi["Jumlah postingan"])
            ax.set_ylabel("Jumlah postingan")
            ax.set_title("Jumlah Postingan per Cluster")
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            for bar, pct in zip(bars, distribusi["Persentase (%)"]):
                ax.annotate(f"{int(bar.get_height())}\n({pct:.1f}%)",
                            (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha="center", va="bottom", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        with col_table:
            st.dataframe(distribusi, width="stretch", hide_index=True)

        # Silhouette score menggunakan cosine; hanya dihitung bila data dan cluster memadai.
        try:
            if len(df_cluster) > len(cluster_ids) and len(cluster_ids) > 1 and len(df_cluster) <= 5000:
                score = silhouette_score(X_tfidf, df_cluster["cluster"], metric="cosine")
                st.metric("Silhouette Score (cosine)", f"{score:.3f}")
                st.caption("Nilai silhouette mendekati 1 menunjukkan pemisahan lebih baik dan nilai dekat 0 menandakan kelompok saling berdekatan.")
        except Exception:
            pass

        # Visualisasi 2D menggunakan TruncatedSVD agar cocok untuk matriks TF-IDF sparse.
        st.markdown("---")
        st.subheader("Peta Sebaran Postingan (TF-IDF → 2D)")
        if X_tfidf.shape[0] >= 3 and X_tfidf.shape[1] >= 2:
            try:
                svd = TruncatedSVD(n_components=2, random_state=42)
                koordinat = svd.fit_transform(X_tfidf)
                fig, ax = plt.subplots(figsize=(9, 5.5))
                for cid in cluster_ids:
                    mask = df_cluster["cluster"].to_numpy() == cid
                    ax.scatter(koordinat[mask, 0], koordinat[mask, 1], alpha=0.65, s=35, label=f"Cluster {cid}")
                ax.set_xlabel("Komponen 1")
                ax.set_ylabel("Komponen 2")
                ax.set_title("Sebaran Dokumen Berdasarkan Kemiripan Teks")
                ax.legend()
                ax.grid(linestyle=":", alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                st.caption("Peta 2D adalah proyeksi untuk membantu membaca pola; titik yang berdekatan cenderung memiliki representasi teks lebih mirip, tetapi visualisasi ini bukan bukti mutlak pemisahan topik.")
            except Exception as e:
                st.warning(f"Visualisasi 2D tidak dapat dibuat: {e}")

        # WordCloud: gunakan seluruh kosakata berbobot TF-IDF per cluster,
        # bukan hanya 10 kata teratas, agar visualisasi lebih penuh dan informatif.
        st.markdown("---")
        st.subheader("WordCloud Tiap Cluster")
        st.caption("Ukuran kata menggambarkan bobot relatif kata dalam cluster")
        cols_wc = st.columns(min(3, len(cluster_ids)))
        for idx, cid in enumerate(cluster_ids):
            with cols_wc[idx % len(cols_wc)]:
                positions_wc = np.flatnonzero(df_cluster["cluster"].to_numpy() == cid)
                if len(positions_wc) > 0:
                    mean_scores_wc = np.asarray(X_tfidf[positions_wc].mean(axis=0)).ravel()
                    frekuensi = {
                        terms[i]: float(mean_scores_wc[i])
                        for i in range(len(terms))
                        if mean_scores_wc[i] > 0
                    }
                else:
                    frekuensi = {}

                st.markdown(f"**Cluster {cid} — {tema_map.get(cid, 'Tema belum ditentukan')}**")
                if frekuensi:
                    wc = WordCloud(
                        width=1000,
                        height=600,
                        background_color="white",
                        colormap="viridis",
                        max_words=60,
                        min_font_size=9,
                        max_font_size=100,
                        margin=3,
                        prefer_horizontal=0.9,
                        relative_scaling=0.45,
                        collocations=False,
                        random_state=42,
                    ).generate_from_frequencies(frekuensi)
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    fig.tight_layout(pad=0)
                    st.pyplot(fig, clear_figure=True, width="stretch")
                    plt.close(fig)
                else:
                    st.caption("Kata dominan belum tersedia.")

        # Detail cluster: jumlah, kata dominan, dan 3 contoh postingan asli.
        st.markdown("---")
        st.subheader("Detail dan Interpretasi Setiap Cluster")
        for cid in cluster_ids:
            df_c = df_cluster[df_cluster["cluster"] == cid]
            words = top_words_map.get(cid, [])
            tema = tema_map.get(cid, f"Tema Cluster {cid}")
            with st.expander(f"Cluster {cid} — {tema} ({len(df_c)} postingan)", expanded=True):
                st.write(f"**Jumlah:** {len(df_c):,} postingan ({len(df_c)/total_cluster*100:.1f}%).")
                st.write("**Lima kata/frasa dominan:** " + (", ".join(w for w, _ in words[:5]) if words else "Belum tersedia"))
                st.write("**Contoh postingan:**")
                kolom_contoh = "text" if "text" in df_c.columns else kolom_teks_model
                for nomor, (_, baris) in enumerate(df_c[kolom_contoh].dropna().drop_duplicates().head(3).items(), start=1):
                    st.info(f"Contoh {nomor}: {baris}")

        st.markdown("---")
        st.subheader("💡 Perbedaan Cluster dan Tumpang Tindih")
        st.markdown("""
        **Perbedaan isi ketiga cluster**

        - **Cluster 0 — Gaji dan UMR:** lebih menonjolkan pembahasan gaji, UMR, serta lowongan kerja. Cluster ini dapat memuat diskusi tentang besaran upah dan kesesuaian gaji dengan kebutuhan pencari kerja.
        - **Cluster 1 — Informasi lowongan dan rekrutmen:** lebih banyak menonjolkan kata seperti *lowongan kerja*, *posisi*, dan *PT*, sehingga cenderung berisi informasi atau pengumuman rekrutmen.
        - **Cluster 2 — Fresh graduate dan pengalaman:** menonjolkan kata *fresh graduate*, *pengalaman*, dan *kerja*. Cluster ini cenderung berkaitan dengan pengalaman atau persyaratan bagi pencari kerja yang baru lulus.

        **Tumpang tindih antarkluster**

        Terdapat indikasi tumpang tindih terutama antara **Cluster 0 dan Cluster 2**. Pembahasan gaji/lowongan kerja dapat muncul bersamaan dengan pengalaman dan kesulitan fresh graduate mencari pekerjaan. Cluster 1 juga dapat bersinggungan dengan keduanya karena unggahan lowongan kerja bisa mencantumkan gaji maupun syarat pengalaman. Dengan demikian, batas topik antarkluster belum
        sepenuhnya tegas.

        **Catatan interpretasi:** tema di atas merupakan interpretasi awal dari kata dominan dan contoh postingan, bukan label yang ditentukan langsung oleh algoritma. K-Means mengelompokkan teks berdasarkan kemiripan fitur TF-IDF, bukan berdasarkan tema yang sudah diketahui. Jika contoh postingan dalam suatu cluster ternyata beragam atau tidak konsisten, hasil tersebut perlu dilaporkan sebagai cluster yang temanya belum jelas—tidak perlu memaksakan interpretasi atau mencari jumlah cluster terbaik.
        """)
        with st.expander("Lihat data clustering yang digunakan"):
            cols_show = [c for c in ["text", "Cleaned_Text", "cluster"] if c in df_cluster.columns]
            st.dataframe(df_cluster[cols_show], width="stretch", hide_index=True)
# ===================================================================
# TAB 4: REGRESI (EVALUASI & FORMULIR INTERAKTIF)
with tab4:
    st.header(" Regresi Linear (Estimasi Jumlah Likes)")
    
    # 1. Konversi tipe data ke numerik (Target y + Fitur X)
    df['like_count_num'] = pd.to_numeric(df['like_count'], errors='coerce')
    # Jika kolom interaksi tidak tersedia pada CSV, isi NaN agar aplikasi tidak langsung crash.
    if 'reply_count' not in df.columns:
        df['reply_count'] = np.nan
    if 'retweet_count' not in df.columns:
        df['retweet_count'] = np.nan

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
