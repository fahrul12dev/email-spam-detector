
import streamlit as st
import joblib

# Memuat model dan TF-IDF dari Google Colab
model = joblib.load("model_spam.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Pengaturan halaman
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Judul aplikasi
st.title("📧 Email Spam Detector")

st.write(
    "Sistem klasifikasi email untuk mendeteksi "
    "apakah sebuah email termasuk Spam atau Ham."
)

st.caption("Menggunakan metode TF-IDF dan algoritma Decision Tree.")

st.divider()

# Form input email
st.subheader("Masukkan Email")

with st.form("form_email"):

    subject = st.text_input(
        "Subject Email",
        placeholder="Masukkan subject email..."
    )

    message = st.text_area(
        "Isi Email",
        placeholder="Masukkan isi email di sini...",
        height=180
    )

    tombol_deteksi = st.form_submit_button(
        "🔍 Deteksi Email",
        use_container_width=True
    )

# Proses klasifikasi
if tombol_deteksi:

    if subject.strip() == "" and message.strip() == "":
        st.warning("Silakan masukkan subject atau isi email terlebih dahulu.")

    else:
        # Menggabungkan subject dan message
        teks_email = subject + " " + message

        # Preprocessing teks
        teks_email = teks_email.lower().strip()

        # Mengubah teks menjadi fitur TF-IDF
        email_tfidf = tfidf.transform([teks_email])

        # Prediksi menggunakan Decision Tree
        hasil = model.predict(email_tfidf)[0]

        st.divider()
        st.subheader("Hasil Klasifikasi")

        if hasil == 1:
            st.error(
                "🚨 EMAIL TERDETEKSI SEBAGAI SPAM",
                icon="🚨"
            )

            st.write(
                "Email ini dikategorikan sebagai pesan spam."
            )

        else:
            st.success(
                "✅ EMAIL TERDETEKSI SEBAGAI HAM",
                icon="✅"
            )

            st.write(
                "Email ini dikategorikan sebagai pesan normal."
            )

# Informasi aplikasi
st.divider()

st.caption(
    "Project Machine Learning - Kelompok 2"
)
