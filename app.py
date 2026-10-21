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
from sklearn.metrics.pairwise import cosine_similarity

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
    

# ===================================================================
# TAB 3: CLUSTERING (VISUALISASI + INTERPRETASI)
# ===================================================================
KOLOM_WAJIB_CLUSTER = ["text", "Cleaned_Text"]

@st.cache_data(show_spinner=False)
def siapkan_stopwords():
    """Stopwords Indonesia + Inggris, negasi dipertahankan."""
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as nltk_stopwords
        daftar = list(nltk_stopwords.words("indonesian")) + list(nltk_stopwords.words("english"))
        sumber = "NLTK"
    except Exception:
        # Cadangan jika NLTK/unduhan tidak tersedia (hasil bisa sedikit berbeda dari Colab)
        daftar = [
            "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada", "adalah",
            "atau", "karena", "jadi", "saat", "dalam", "akan", "ada", "juga", "lebih", "sudah",
            "bisa", "dapat", "agar", "para", "oleh", "sebagai", "nya", "sih", "dong", "nih", "deh",
            "lah", "aja", "aku", "saya", "kamu", "dia", "mereka", "kami", "kita", "rt", "https",
            "http", "co",
        ]
        sumber = "cadangan"

    negasi = {"tidak", "bukan", "belum", "jangan", "tak"}
    daftar = [k for k in daftar if k not in negasi]
    daftar.extend(["ya", "yg", "ga", "yuk", "dah", "ngga", "engga", "ygy", "oot"])
    return sorted(set(daftar)), sumber


@st.cache_data(show_spinner="Menjalankan TF-IDF dan K-Means...")
def hitung_clustering(teks_bersih, stopwords_final):
    """TF-IDF + K-Means (3 cluster)"""
    vectorizer = TfidfVectorizer(
        stop_words=stopwords_final,
        max_features=5000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(list(teks_bersih))

    # Pastikan terdapat minimal 3 representasi teks yang berbeda untuk 3 cluster.
    if X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("Data teks valid terlalu sedikit atau tidak memiliki kosakata.")
    if X.shape[0] < 3:
        raise ValueError("Minimal 3 postingan diperlukan untuk 3 cluster.")

    model = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    centroids = model.cluster_centers_
    urutan = centroids.argsort()[:, ::-1]

    kata_dominan = {}
    frekuensi_wc = {}
    for c in range(centroids.shape[0]):
        kata_dominan[c] = [
            (terms[i], float(centroids[c, i])) for i in urutan[c, :15] if centroids[c, i] > 0
        ]
        frekuensi_wc[c] = {
            terms[i]: float(centroids[c, i]) for i in np.flatnonzero(centroids[c] > 0)
        }

    skor = None
    n_label = len(set(labels))
    if 1 < n_label < X.shape[0]:
        try:
            skor = float(silhouette_score(X, labels, metric="cosine"))
        except ValueError:
            skor = None

    # Postingan ambigu: hampir sama dekatnya ke dua centroid
    sim = np.sort(cosine_similarity(X, centroids), axis=1)
    n_ambigu = int(((sim[:, -1] - sim[:, -2]) < 0.05).sum())

    koordinat = None
    if X.shape[0] >= 3 and X.shape[1] >= 2:
        koordinat = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)

    return {
        "labels": labels,
        "kata_dominan": kata_dominan,
        "frekuensi_wc": frekuensi_wc,
        "silhouette": skor,
        "n_ambigu": n_ambigu,
        "koordinat": koordinat,
    }


def beri_nama_tema(kata_top5):
    """Nama tema diturunkan dari 5 kata dominan hasil clustering"""
    kata = set(" ".join(kata_top5).split())
    if kata & {"fresh", "graduate"}:
        return "Tantangan Fresh Graduate dan Persyaratan Pengalaman"
    if kata & {"posisi", "pt", "oktober", "pendaftaran", "penempatan", "kualifikasi"}:
        return "Informasi Lowongan Kerja dan Rekrutmen"
    if kata & {"gaji", "umr", "upah"}:
        return "Diskusi Gaji, UMR, dan Tantangan Mencari Kerja"
    return None


def tampilkan_clustering(df_utama):
    st.header("Analisis Clustering Percakapan Dunia Kerja")
    st.caption("Pengelompokan teks menggunakan TF-IDF dan K-Means dengan 3 cluster.")

    # 1. Pemeriksaan kolom wajib (return, bukan st.stop, agar tab Regresi tetap tampil)
    hilang = [k for k in KOLOM_WAJIB_CLUSTER if k not in df_utama.columns]
    if hilang:
        st.error(
            "Clustering tidak dapat dijalankan karena kolom wajib berikut tidak tersedia: "
            + ", ".join(hilang)
            + ". Pastikan dataset utama memiliki kolom text dan Cleaned_Text."
        )
        return

    # 2. Persiapan data: buang teks bersih kosong dan teks asli duplikat (seperti di notebook)
    jumlah_awal = len(df_utama)
    df_cluster = df_utama.copy()
    df_cluster["clean_text"] = df_cluster["Cleaned_Text"].fillna("").astype(str).str.strip()

    n_kosong = int((df_cluster["clean_text"] == "").sum())
    df_cluster = df_cluster[df_cluster["clean_text"] != ""].copy()

    sebelum_dup = len(df_cluster)
    df_cluster = df_cluster.drop_duplicates(subset="text").reset_index(drop=True)
    n_duplikat = sebelum_dup - len(df_cluster)

    st.caption(
        f"Dataset utama: {jumlah_awal:,} baris. Baris dikeluarkan dari clustering: "
        f"{n_kosong:,} karena Cleaned_Text kosong dan {n_duplikat:,} karena teks asli duplikat. "
        f"Baris yang dianalisis: {len(df_cluster):,}."
    )

    if len(df_cluster) < 3:
        st.warning("Data teks valid terlalu sedikit untuk menjalankan clustering. Periksa file CSV yang diunggah.")
        return

    # 3. Clustering
    stopwords_final, sumber_stopwords = siapkan_stopwords()
    if sumber_stopwords != "NLTK":
        st.warning("Stopwords NLTK tidak dapat dimuat, memakai daftar cadangan")

    try:
        hasil = hitung_clustering(tuple(df_cluster["clean_text"]), stopwords_final)
    except ValueError:
        st.error("Teks tidak memiliki kosakata yang cukup untuk clustering. Periksa kolom Cleaned_Text.")
        return
    except Exception as e:
        st.error(f"Clustering gagal dijalankan: {e}")
        return

    df_cluster["cluster"] = hasil["labels"]
    cluster_ids = sorted(df_cluster["cluster"].unique().tolist())
    total = len(df_cluster)
    cluster_counts = df_cluster["cluster"].value_counts().sort_index()
    kata_dominan = hasil["kata_dominan"]

    # 4. Nama tema berdasarkan kata dominan (setiap nama dipakai paling banyak sekali)
    tema_map = {}
    dipakai = set()
    for cid in cluster_ids:
        top5 = [w for w, _ in kata_dominan.get(cid, [])[:5]]
        nama = beri_nama_tema(top5)
        if nama is None or nama in dipakai:
            nama = f"Tema Cluster {cid} belum jelas (perlu ditinjau)"
        dipakai.add(nama)
        tema_map[cid] = nama

    st.subheader("Ringkasan Dataset Clustering")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total postingan dianalisis", f"{total:,}")
    m2.metric("Jumlah cluster", str(len(cluster_ids)))
    m3.metric("Metode", "TF-IDF + K-Means")
    st.caption("Nama tema ditentukan dari kata dominan dan contoh postingan setelah clustering, bukan ditetapkan sebelumnya.")

    # 5. Distribusi anggota cluster
    st.markdown("---")
    st.subheader("Distribusi Anggota Cluster")
    distribusi = pd.DataFrame({
        "Cluster": [f"Cluster {cid}" for cid in cluster_counts.index],
        "Tema": [tema_map[cid] for cid in cluster_counts.index],
        "Jumlah postingan": cluster_counts.values,
        "Persentase (%)": (cluster_counts.values / total * 100).round(2),
    })
    col_chart, col_table = st.columns([1.6, 1])
    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(distribusi["Cluster"], distribusi["Jumlah postingan"])
        ax.set_ylabel("Jumlah postingan")
        ax.set_title("Jumlah Postingan per Cluster")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_ylim(0, distribusi["Jumlah postingan"].max() * 1.2)
        for bar, pct in zip(bars, distribusi["Persentase (%)"]):
            ax.annotate(
                f"{int(bar.get_height())}\n({pct:.1f}%)",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9,
            )
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with col_table:
        st.dataframe(distribusi, width="stretch", hide_index=True)

    # 6. Kualitas pemisahan
    if hasil["silhouette"] is not None:
        st.metric("Silhouette Score (cosine)", f"{hasil['silhouette']:.3f}")
        st.caption("Nilai mendekati 1 berarti pemisahan lebih baik; nilai dekat 0 berarti cluster saling berdekatan.")
    else:
        st.caption("Silhouette score tidak dapat dihitung untuk data ini.")

    # 7. Peta sebaran 2D
    st.markdown("---")
    st.subheader("Peta Sebaran Postingan (TF-IDF → 2D)")
    koordinat = hasil["koordinat"]
    if koordinat is not None:
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
        st.caption("Peta 2D hanyalah proyeksi untuk membantu membaca pola, bukan bukti mutlak pemisahan topik.")
    else:
        st.info("Visualisasi 2D tidak tersedia karena jumlah fitur teks berbeda terlalu sedikit.")

    # 8. WordCloud per cluster
    st.markdown("---")
    st.subheader("WordCloud Tiap Cluster")
    st.caption("Ukuran kata menggambarkan bobot relatif kata dalam cluster. Baca bersama contoh postingan.")
    cols_wc = st.columns(len(cluster_ids))
    for idx, cid in enumerate(cluster_ids):
        with cols_wc[idx]:
            st.markdown(f"**Cluster {cid}: {tema_map[cid]}**")
            frekuensi = hasil["frekuensi_wc"].get(cid, {})
            if frekuensi:
                wc = WordCloud(
                    width=1000, height=600, background_color="white", colormap="viridis",
                    max_words=60, min_font_size=9, max_font_size=100, margin=3,
                    prefer_horizontal=0.9, relative_scaling=0.45, collocations=False,
                    random_state=42,
                ).generate_from_frequencies(frekuensi)
                fig, ax = plt.subplots(figsize=(8, 4.8))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig.tight_layout(pad=0)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            else:
                st.caption("Kata dominan belum tersedia.")

    # 9. Word Count per cluster
    st.markdown("---")
    st.subheader("Word Count per Cluster")
    df_cluster["jumlah_kata"] = df_cluster["clean_text"].str.split().str.len()
    ringkasan_kata = (
        df_cluster.groupby("cluster")["jumlah_kata"]
        .agg(jumlah_postingan="count", total_kata="sum", rata_rata_kata="mean",
             kata_terpendek="min", kata_terpanjang="max")
        .reset_index()
    )
    ringkasan_kata["Tema"] = ringkasan_kata["cluster"].map(tema_map)
    ringkasan_kata["rata_rata_kata"] = ringkasan_kata["rata_rata_kata"].round(2)
    st.dataframe(ringkasan_kata, width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        [f"Cluster {cid}" for cid in ringkasan_kata["cluster"]],
        ringkasan_kata["rata_rata_kata"],
    )
    ax.set_ylabel("Rata-rata jumlah kata per postingan")
    ax.set_title("Rata-rata Word Count per Cluster")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    # 10. Sepuluh kata yang paling sering muncul pada setiap cluster
    st.subheader("10 Kata Paling Sering Muncul per Cluster")
    for cid in cluster_ids:
        kumpulan_kata = " ".join(df_cluster.loc[df_cluster["cluster"] == cid, "clean_text"])
        hitung_kata = pd.Series(kumpulan_kata.split()).value_counts().head(10)
        st.markdown(f"**Cluster {cid} — {tema_map[cid]}**")
        if not hitung_kata.empty:
            st.dataframe(
                hitung_kata.rename_axis("Kata").reset_index(name="Frekuensi"),
                width="stretch", hide_index=True,
            )
        else:
            st.caption("Belum ada kata untuk dihitung.")

    # 11. Detail tiap cluster: jumlah, 5 kata dominan, 3 contoh, tema
    st.markdown("---")
    st.subheader("Detail dan Interpretasi Setiap Cluster")
    for cid in cluster_ids:
        df_c = df_cluster[df_cluster["cluster"] == cid]
        top5 = [w for w, _ in kata_dominan.get(cid, [])[:5]]
        with st.expander(f"Cluster {cid}: {tema_map[cid]} ({len(df_c)} postingan)", expanded=True):
            st.write(f"**Jumlah:** {len(df_c):,} postingan ({len(df_c) / total * 100:.1f}%).")
            st.write("**Lima kata dominan:** " + (", ".join(top5) if top5 else "Belum tersedia"))
            st.write("**Tiga contoh postingan:**")
            contoh = df_c["text"].dropna().astype(str).head(3)
            for nomor, teks in enumerate(contoh, start=1):
                st.text_area(f"Contoh {nomor}", teks, height=130,
                             key=f"contoh_cluster_{cid}_{nomor}", disabled=True)

    # 10. Perbedaan cluster dan tumpang tindih (dihitung dari hasil, bukan teks tetap)
    st.markdown("---")
    st.subheader("💡 Perbedaan Cluster dan Tumpang Tindih")

    st.markdown("**Apa yang membedakan isi ketiga cluster?**")
    for cid in cluster_ids:
        top5 = [w for w, _ in kata_dominan.get(cid, [])[:5]]
        st.markdown(f"- **Cluster {cid} ({tema_map[cid]}):** kata dominan {', '.join(top5)}.")
    st.caption("Tema diturunkan dari kata dominan dan perlu dikonfirmasi dengan membaca contoh postingan di atas.")

    st.markdown("**Apakah masih terdapat pembahasan yang tumpang tindih?**")
    top15 = {cid: {w for w, _ in kata_dominan.get(cid, [])} for cid in cluster_ids}
    baris_irisan = []
    for i, a in enumerate(cluster_ids):
        for b in cluster_ids[i + 1:]:
            sama = sorted(top15[a] & top15[b])
            baris_irisan.append(
                f"- Cluster {a} & {b}: " + (", ".join(sama) if sama else "tidak ada kata yang sama")
            )
    st.markdown("Kata yang sama pada 15 kata dominan teratas antar cluster:\n" + "\n".join(baris_irisan))
    st.markdown(
        f"Sebanyak **{hasil['n_ambigu']:,} dari {total:,} postingan "
        f"({hasil['n_ambigu'] / total * 100:.1f}%)** hampir sama dekatnya ke dua centroid cluster."
    )

    persen_terbesar = cluster_counts.max() / total * 100
    if hasil["silhouette"] is not None and hasil["silhouette"] < 0.1:
        st.warning(
            f"Silhouette score hanya {hasil['silhouette']:.3f}, sehingga pemisahan antar-cluster lemah "
            "dan tumpang tindih topik masih jelas terlihat. Hasil ini dilaporkan apa adanya."
        )
    if persen_terbesar > 60:
        st.warning(
            f"Cluster {int(cluster_counts.idxmax())} menampung {persen_terbesar:.1f}% postingan, "
            "sehingga isinya kemungkinan campuran dan temanya belum sepenuhnya jelas. "
            "Periksa contoh postingannya sebelum menyimpulkan tema."
        )
    st.caption(
        "K-Means mengelompokkan teks berdasarkan kemiripan fitur TF-IDF, bukan tema yang sudah diketahui. "
        "Jumlah cluster tidak dioptimasi (sesuai ketentuan tugas)."
    )

    # 11. Temuan utama dan keterbatasan (dibuat dari hasil yang sedang dihitung)
    st.markdown("---")
    st.subheader("Temuan Utama Clustering")
    c_besar = int(cluster_counts.idxmax())
    c_kecil = int(cluster_counts.idxmin())
    kata_besar = ", ".join(w for w, _ in kata_dominan.get(c_besar, [])[:5])
    kata_kecil = ", ".join(w for w, _ in kata_dominan.get(c_kecil, [])[:5])
    st.markdown(
        f"1. **Cluster terbesar:** Cluster {c_besar} ({tema_map[c_besar]}) berisi "
        f"{int(cluster_counts.max()):,} postingan ({persen_terbesar:.1f}%), dengan kata dominan {kata_besar}."
    )
    st.markdown(
        f"2. **Cluster terkecil:** Cluster {c_kecil} ({tema_map[c_kecil]}) berisi "
        f"{int(cluster_counts.min()):,} postingan ({cluster_counts.min() / total * 100:.1f}%), "
        f"dengan kata dominan {kata_kecil}. Data tidak terbagi rata antar cluster."
    )
    if hasil["silhouette"] is not None:
        st.markdown(
            f"3. **Kejelasan pemisahan:** silhouette score {hasil['silhouette']:.3f} dan "
            f"{hasil['n_ambigu'] / total * 100:.1f}% postingan ambigu, sehingga batas antar topik "
            "belum tegas dan perlu dibaca hati-hati."
        )

    st.subheader("Keterbatasan Clustering")
    st.markdown(
        "1. **Keterbatasan data:** analisis hanya mewakili postingan yang terkumpul lewat kata kunci crawling. "
        "Postingan dengan Cleaned_Text kosong atau teks duplikat dikeluarkan, sehingga hasil belum tentu "
        "mewakili seluruh percakapan di X tentang dunia kerja."
    )
    st.markdown(
        "2. **Keterbatasan metode:** TF-IDF dan K-Means hanya melihat kemiripan kata, bukan konteks makna. "
        "Jumlah cluster ditetapkan 3 tanpa dioptimasi, dan nama tema bersifat interpretasi yang perlu "
        "diperiksa manual."
    )

    with st.expander("Lihat data clustering yang digunakan"):
        st.dataframe(df_cluster[["text", "Cleaned_Text", "jumlah_kata", "cluster"]], width="stretch", hide_index=True)


with tab3:
    tampilkan_clustering(df)

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