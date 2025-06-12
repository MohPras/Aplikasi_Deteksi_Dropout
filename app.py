import streamlit as st
import pandas as pd
import joblib
import io

# --- Konfigurasi Awal ---
st.set_page_config(layout="wide", page_title="Prediksi Mahasiswa DO")

# --- Memuat Model (Dilakukan sekali saat aplikasi dimulai) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_rf.joblib')
        return model
    except FileNotFoundError:
        st.error("File model 'random_forest_model.joblib' tidak ditemukan. Pastikan file model berada di direktori yang sama dengan app.py.")
        st.stop()

model = load_model()


# --- DEFINISI FITUR UNTUK INPUT STREAMLIT (Digunakan juga sebagai referensi kolom) ---
features_definition = {
    "Marital_status": {
        "type": "selectbox",
        "options": {
            "Single": 1, "Married": 2, "Widower": 3, "Divorced": 4, "Legally Separated": 5
        },
        "default": 1,
        "help": "Status pernikahan mahasiswa."
    },
    "Application_mode": {
        "type": "selectbox",
        "options": {
            "1 - 1st Phase - General Contingent": 1, "2 - Ordinance No. 612/93": 2,
            "5 - 1st Phase - Special Contingent (Azores Island)": 5, "7 - Holders of Other Higher Courses": 7,
            "10 - Ordinance No. 854-B/99": 10, "15 - International Student (Bachelor)": 15,
            "16 - 1st phase - Special Contingent (Madeira Island)": 16, "17 - 2nd phase - General Contingent": 17,
            "18 - 3rd phase - General Contingent": 18, "26 - Ordinance No. 533-A/99, Item B2 (Different Plan)": 26,
            "27 - Ordinance No. 533-A/99, Item B3 (Other Institution)": 27, "39 - Over 23 Years Old": 39,
            "42 - Transfer": 42, "43 - Change of Course": 43, "44 - Technological Specialization Diploma Holders": 44,
            "51 - Change of Institution/Course": 51, "53 - Short Cycle Diploma Holders": 53,
            "57 - Change of Institution/Course (International)": 57
        },
        "default": 1,
        "help": "Mode aplikasi pendaftaran mahasiswa."
    },
    "Previous_qualification_grade": {
        "type": "number_input", "min_value": 0.0, "max_value": 200.0, "default": 140.0, "step": 0.1,
        "help": "Nilai rata-rata kualifikasi sebelumnya (misal: nilai UN/raport SMA)."
    },
    "Admission_grade": {
        "type": "number_input", "min_value": 0.0, "max_value": 200.0, "default": 130.0, "step": 0.1,
        "help": "Nilai saat masuk universitas (misal: nilai tes masuk)."
    },
    "Displaced": {
        "type": "selectbox", "options": {"Tidak": 0, "Ya": 1}, "default": 0,
        "help": "Apakah mahasiswa pindahan dari kota/negara lain?"
    },
    "Debtor": {
        "type": "selectbox", "options": {"Tidak": 0, "Ya": 1}, "default": 0,
        "help": "Apakah mahasiswa memiliki tunggakan (misal: biaya kuliah)?"
    },
    "Tuition_fees_up_to_date": {
        "type": "selectbox", "options": {"Tidak": 0, "Ya": 1}, "default": 1,
        "help": "Apakah pembayaran uang kuliah tepat waktu?"
    },
    "Gender": {
        "type": "selectbox", "options": {"Perempuan": 0, "Laki-laki": 1}, "default": 1,
        "help": "Jenis kelamin mahasiswa."
    },
    "Scholarship_holder": {
        "type": "selectbox", "options": {"Tidak": 0, "Ya": 1}, "default": 0,
        "help": "Apakah mahasiswa penerima beasiswa?"
    },
    "Age_at_enrollment": {
        "type": "number_input", "min_value": 15, "max_value": 90, "default": 18, "step": 1,
        "help": "Usia mahasiswa saat pertama kali mendaftar."
    },
    "Curricular_units_1st_sem_enrolled": {
        "type": "number_input", "min_value": 0, "max_value": 100, "default": 6, "step": 1,
        "help": "Jumlah unit kurikuler yang didaftarkan pada semester 1."
    },
    "Curricular_units_1st_sem_approved": {
        "type": "number_input", "min_value": 0, "max_value": 100, "default": 6, "step": 1,
        "help": "Jumlah unit kurikuler yang disetujui pada semester 1."
    },
    "Curricular_units_1st_sem_grade": {
        "type": "number_input", "min_value": 0.0, "max_value": 20.0, "default": 12.0, "step": 0.1,
        "help": "Nilai rata-rata unit kurikuler pada semester 1."
    },
    "Curricular_units_2nd_sem_enrolled": {
        "type": "number_input", "min_value": 0, "max_value": 100, "default": 6, "step": 1,
        "help": "Jumlah unit kurikuler yang didaftarkan pada semester 2."
    },
    "Curricular_units_2nd_sem_evaluations": {
        "type": "number_input", "min_value": 0, "max_value": 20, "default": 6, "step": 1,
        "help": "Jumlah evaluasi yang dilakukan pada unit kurikuler semester 2."
    },
    "Curricular_units_2nd_sem_approved": {
        "type": "number_input", "min_value": 0, "max_value": 100, "default": 6, "step": 1,
        "help": "Jumlah unit kurikuler yang disetujui pada semester 2."
    },
    "Curricular_units_2nd_sem_grade": {
        "type": "number_input", "min_value": 0.0, "max_value": 20.0, "default": 12.0, "step": 0.1,
        "help": "Nilai rata-rata unit kurikuler pada semester 2."
    },
    "Curricular_units_2nd_sem_without_evaluations": {
        "type": "number_input", "min_value": 0, "max_value": 100, "default": 0, "step": 1,
        "help": "Jumlah unit kurikuler tanpa evaluasi pada semester 2."
    },
}

# Urutan kolom harus sama persis dengan urutan fitur saat model dilatih
feature_columns_order = list(features_definition.keys())


# --- Fungsi untuk Halaman Prediksi Single Data ---
def single_prediction_page():
    st.title("Prediksi Mahasiswa DO (Single Data)")
    st.markdown("Aplikasi ini membantu memprediksi kemungkinan seorang mahasiswa akan mengalami *dropout* berdasarkan data akademik dan demografi.")
    st.markdown("---")
    
    st.header("Masukkan Data Mahasiswa")

    input_data = {}
    cols = st.columns(2)
    col_idx = 0

    for feature_name, details in features_definition.items():
        with cols[col_idx % 2]:
            st.subheader(f"{feature_name.replace('_', ' ').title()}")
            if details["type"] == "number_input":
                input_data[feature_name] = st.number_input(
                    label=f"Masukkan {feature_name.replace('_', ' ').title()}",
                    min_value=details["min_value"],
                    max_value=details["max_value"],
                    value=details["default"],
                    step=details.get("step", 1),
                    help=details.get("help", "")
                )
            elif details["type"] == "selectbox":
                display_options = list(details["options"].keys())
                
                default_label = next(
                    (key for key, value in details["options"].items() if value == details["default"]),
                    None
                )
                
                if default_label is None and display_options:
                    default_index = 0
                elif default_label is not None:
                    default_index = display_options.index(default_label)
                else:
                    default_index = 0
                
                selected_display = st.selectbox(
                    label=f"Pilih {feature_name.replace('_', ' ').title()}",
                    options=display_options,
                    index=default_index,
                    help=details.get("help", "")
                )
                input_data[feature_name] = details["options"][selected_display]
        col_idx += 1

    st.markdown("---")
    if st.button("Prediksi Kemungkinan DO"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns_order]

        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.subheader("Hasil Prediksi:")
            if prediction[0] == 0: # 0 = DO
                st.error(f"⚠️ Mahasiswa ini **DIPREDIKSI AKAN MENGALAMI DROPOUT (DO)**.")
                st.write(f"Probabilitas Dropout: **{prediction_proba[0][0]:.2f}**")
            else: # 1 = TIDAK DO
                st.success(f"✅ Mahasiswa ini **DIPREDIKSI TIDAK AKAN MENGALAMI DROPOUT (DO)**.")
                st.write(f"Probabilitas Tidak Dropout: **{prediction_proba[0][1]:.2f}**")
            
            st.markdown("---")
            st.subheader("Detail Input Data:")
            st.dataframe(input_df)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat prediksi: {e}")
            st.warning(f"Pastikan input data Anda sesuai dengan format yang diharapkan oleh model. Error detail: {e}")

    st.markdown("---")
    st.caption("Dibuat dengan Streamlit dan Model Machine Learning Random Forest")


# --- Fungsi untuk Halaman Prediksi Multiple Data ---
def multiple_prediction_page():
    st.title("Prediksi Multi-Mahasiswa DO (Unggah File CSV)")
    st.markdown("Unggah file CSV Anda yang berisi data banyak mahasiswa untuk diprediksi kemungkinan *dropout* secara bersamaan.")
    st.markdown("---")

    st.subheader("1. Unduh Contoh File CSV")
    st.info("Gunakan file contoh ini sebagai template untuk memastikan format kolom Anda benar.")

    # Membuat contoh DataFrame
    sample_data = {col: [details["default"]] * 2 for col, details in features_definition.items()}
    if "Age_at_enrollment" in sample_data:
        sample_data["Age_at_enrollment"][1] = 20
    if "Curricular_units_1st_sem_grade" in sample_data:
        sample_data["Curricular_units_1st_sem_grade"][1] = 10.5
    if "Gender" in sample_data:
        sample_data["Gender"][1] = 0
    if "Scholarship_holder" in sample_data:
        sample_data["Scholarship_holder"][1] = 1

    df_sample = pd.DataFrame(sample_data)
    df_sample = df_sample[feature_columns_order] 

    csv_sample_buffer = io.StringIO()
    df_sample.to_csv(csv_sample_buffer, index=False)
    csv_sample_data = csv_sample_buffer.getvalue().encode('utf-8')

    st.download_button(
        label="Unduh Contoh CSV",
        data=csv_sample_data,
        file_name="contoh_data_mahasiswa_input.csv",
        mime="text/csv",
        help="Klik untuk mengunduh file CSV contoh dengan format kolom yang benar."
    )
    
    st.markdown("---")

    st.subheader("2. Unggah File CSV Anda")
    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        help="Pastikan kolom di CSV Anda sesuai dengan yang diharapkan model: " + ", ".join(feature_columns_order)
    )

    df_results = pd.DataFrame()

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success("File CSV berhasil diunggah!")
            st.write("Pratinjau Data Anda:")
            st.dataframe(df_input.head())

            missing_cols = [col for col in feature_columns_order if col not in df_input.columns]
            extra_cols = [col for col in df_input.columns if col not in feature_columns_order]

            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan di file CSV Anda: {', '.join(missing_cols)}. Harap periksa format file atau gunakan file contoh.")
                return
            
            if extra_cols:
                st.warning(f"Kolom berikut di file CSV Anda tidak digunakan oleh model dan akan diabaikan: {', '.join(extra_cols)}.")
            
            st.subheader("3. Melakukan Prediksi")
            with st.spinner("Sedang memproses prediksi..."):
                df_to_predict = df_input[feature_columns_order]
                
                predictions = model.predict(df_to_predict)
                probabilities = model.predict_proba(df_to_predict)

                df_results = df_input.copy()
                df_results['Predicted_Dropout_Status'] = ['DO' if p == 0 else 'Tidak DO' for p in predictions]
                df_results['Dropout_Probability'] = [proba[0] for proba in probabilities]

            st.success("Prediksi berhasil!")
            st.subheader("4. Hasil Prediksi")
            st.dataframe(df_results)

            st.subheader("5. Unduh Hasil Prediksi")
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode('utf-8')

            st.download_button(
                label="Unduh Hasil Prediksi (CSV)",
                data=csv_data,
                file_name="prediksi_mahasiswa_do_results.csv",
                mime="text/csv"
            )

        except pd.errors.EmptyDataError:
            st.error("File CSV kosong. Harap unggah file dengan data.")
        except pd.errors.ParserError:
            st.error("Gagal membaca file CSV. Pastikan format CSV valid. Pastikan juga data numerik tidak mengandung teks atau spasi.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.warning("Pastikan file CSV Anda memiliki format yang benar dan semua kolom fitur yang diperlukan ada. Error detail: " + str(e))


# --- Sidebar Judul ---
st.sidebar.markdown("# 🎓 Aplikasi Deteksi Dropout Student")
st.sidebar.markdown("Selamat datang! Silakan pilih halaman di bawah ini untuk memulai prediksi:")

# Navigasi Halaman
page_selection = st.sidebar.radio(
    "🔍 Pilih Mode Prediksi:",
    ("📄 Prediksi Multi-Data", "👤 Prediksi Single Data")
)

# Menampilkan Halaman Sesuai Pilihan
if page_selection == "👤 Prediksi Single Data":
    single_prediction_page()
elif page_selection == "📄 Prediksi Multi-Data":
    multiple_prediction_page()

# Garis pembatas & Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🛠️ Dibuat dengan **Streamlit**")
st.sidebar.caption("© Mohammad Nurdin Prastya Hermansah 2025")


