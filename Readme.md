# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech Jaya Jaya Institut

----------------------------------

![image](https://github.com/user-attachments/assets/49b6b94d-abfc-4e90-a7bb-066c440e4cc7)


## Business Understanding

Jaya Jaya Institute adalah lembaga pendidikan tinggi yang berdiri sejak tahun 2000 dan telah meluluskan banyak mahasiswa dengan reputasi baik. Namun, institusi ini menghadapi tantangan tingginya angka mahasiswa dropout. Masalah ini berdampak pada reputasi, perencanaan, dan kualitas layanan pendidikan.

Untuk mengatasi hal tersebut, Jaya Jaya Institute berupaya membangun sistem prediksi guna mendeteksi mahasiswa yang berisiko dropout sejak dini. Dengan sistem ini, institusi dapat memberikan intervensi tepat waktu, seperti bimbingan akademik atau konseling, guna meningkatkan retensi dan keberhasilan mahasiswa.

----------------------------------

### Permasalahan Bisnis
Permasalahan Bisnis yang Akan Diselesaikan:
1. Tingginya angka siswa yang dropout menjadi tantangan besar bagi Jaya Jaya Institut.
2. Tidak adanya sistem prediktif untuk mengidentifikasi siswa berisiko dropout sejak dini.
3. Diperlukan model prediksi yang akurat berdasarkan data performa siswa.
4. Dibutuhkan dashboard visual untuk membantu institusi memantau dan memahami kondisi siswa secara lebih efisien.

----------------------------------

### Cakupan Proyek
Proyek ini bertujuan untuk membantu Jaya Jaya Institut dalam mengidentifikasi siswa yang berisiko tinggi mengalami dropout. Cakupan proyek mencakup hal-hal berikut:

1. **Eksplorasi dan Pembersihan Data**
   Melakukan eksplorasi awal terhadap dataset performa siswa serta membersihkan data dari nilai yang hilang, duplikat, atau tidak konsisten.

2. **Analisis Data dan Visualisasi**
   Menganalisis pola dan hubungan antar fitur yang memengaruhi potensi dropout serta menyajikannya dalam bentuk visualisasi yang mudah dipahami.

3. **Pembangunan Model Prediksi Dropout**
   Menggunakan algoritma machine learning untuk membangun model prediktif yang dapat mengklasifikasikan siswa berdasarkan tingkat risiko dropout.

4. **Evaluasi Model**
   Menilai performa model dengan metrik evaluasi yang sesuai (seperti akurasi, precision, recall, dan F1-score), untuk memastikan prediksi yang andal.

5. **Pembuatan Dashboard Interaktif**
   Membangun dashboard visual yang menampilkan performa siswa, prediksi risiko dropout, dan insight penting lainnya guna membantu pengambilan keputusan.

6. **Rekomendasi Intervensi**
   Memberikan rekomendasi awal terkait strategi intervensi atau tindakan lanjut yang dapat diambil oleh pihak institusi berdasarkan hasil analisis.

----------------------------------

### Persiapan

Sumber data:
- [Students' Performance data](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance 'Dicoding GitHub - Students Performance data')

----------------------------------

### Setup Environment:

**A. Clone Repository**

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di lingkungan lokal Anda:

1. **Clone Repository**

   ```bash
   git clone <URL-repo>
   ```

2. **Buat dan Aktifkan Virtual Environment**

   * **Untuk Windows:**

     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   * **Untuk macOS/Linux:**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

###### Catatan:

* Pastikan Anda sudah menginstal **Python 3.8 atau lebih tinggi**.
* Gunakan `python` atau `python3` sesuai dengan OS dan pengaturan sistem Anda.
* Jika menggunakan **Anaconda**, Anda bisa membuat environment dengan:

  ```bash
  conda create --name dropout-env python=3.9
  conda activate dropout-env
  pip install -r requirements.txt
  ```

**B. Akses Dataset**

###### Install dan import library
```python
# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import joblib
import math

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

from google.colab import files
```

###### Dapatkan data
```python
# baca data
dataset_url = 'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv'

# Read dataset
df = pd.read_csv(dataset_url, delimiter=';')
df.head(10)
```

**C. Akses Dashboard**

- email “mohammadprastya2023@gmail.com” dan password “123Nurdin”

###### Tutorial Singkat: Membuka Dashboard Metabase dari File Backup

**Prasyarat:**
Pengguna telah memiliki Java (JRE 8 atau lebih baru) terinstal dan telah mengunduh file `metabase.jar`.

**Langkah-langkah:**

1.  **Hentikan Layanan Metabase (Jika Sedang Berjalan):**
    * Apabila *instance* Metabase saat ini sedang aktif, harap hentikan prosesnya. Jika dijalankan melalui Command Prompt/Terminal, dapat dihentikan dengan menekan `Ctrl + C`.

2.  **Identifikasi Lokasi Folder Database Metabase:**
    * Temukan folder `.metabase` yang biasanya terletak di direktori *home* pengguna:
        * Untuk Windows: `C:\Users\NamaPengguna\.metabase`
        * Untuk macOS/Linux: `/home/NamaPengguna/.metabase`
    * (Apabila folder ini belum terbentuk, Anda dapat menjalankan `java -jar metabase.jar` satu kali, lalu hentikan prosesnya, dan folder tersebut akan secara otomatis dibuat.)

3.  **Ganti File Database:**
    * **Penting:** Harap **memindahkan atau mengganti nama** file `metabase.db.mv.db` dan `metabase.db.trace.db` yang ada di dalam folder `.metabase` tersebut (sebagai langkah pencadangan).
    * **Salin** kedua file *backup* Anda (`metabase.db.mv.db` dan `metabase.db.trace.db`) ke dalam folder `.metabase` yang telah diidentifikasi. Pastikan nama file tetap sama persis.

4.  **Jalankan Metabase:**
    * Buka Command Prompt atau Terminal.
    * Navigasikan ke direktori tempat file `metabase.jar` berada (contoh: `cd C:\Metabase`).
    * Eksekusi Metabase menggunakan perintah berikut: `java -jar metabase.jar`

5.  **Akses Dashboard:**
    * Setelah Metabase selesai memuat sepenuhnya, buka peramban web (browser) dan kunjungi alamat: `http://localhost:3000`
    * Dashboard Metabase Anda seharusnya kini dapat diakses dan menampilkan data dari file *backup* yang telah disalin.

----------------------------------


## Business Dashboard
Dashboard Monitoring student Jaya jaya university adalah bussines dashboard yang didesain sefektif mungkin untuk memberikan insight kepada universitas terkait dropout rate yang cukup tinggi. Berikut dibawah ini adalah cuplikan dari sebagian dahboard bussines ini untuk melihat fullnya bisa kunjungi link yaitube ini ( https://www.youtube.com/watch?v=_67WphBXaGM ).

<img width="962" alt="image" src="https://github.com/user-attachments/assets/a90e9b42-9a17-4e26-91b5-9cb0fb809c22" />

### **Halaman Pertama: Visualisasi Monitoring Student secara Keseluruhan**

Halaman pertama dari dashboard dirancang untuk memberikan **gambaran umum** mengenai profil dan karakteristik mahasiswa di Jaya Jaya Institut. Tujuan utama dari halaman ini adalah untuk memberikan informasi ringkas namun komprehensif kepada pihak manajemen agar dapat memantau kondisi mahasiswa secara real-time. Adapun beberapa visualisasi utama yang ditampilkan pada halaman ini meliputi:

1. **Visualisasi Total Status Student**
   Menampilkan jumlah total mahasiswa berdasarkan status mereka, seperti *enroled*, *graduation*, *dropout*. Visualisasi ini berguna untuk memantau perkembangan populasi mahasiswa dan mengidentifikasi proporsi siswa yang telah berhenti belajar.

2. **Rata-Rata Usia Student Berdasarkan Gender**
   Menyajikan perbandingan rata-rata usia antara mahasiswa laki-laki dan perempuan. Hal ini dapat memberikan wawasan mengenai karakteristik demografis populasi mahasiswa dan membantu dalam penyusunan kebijakan berbasis data.

3. **Distribusi Student Berdasarkan Attendance Type**
   Visualisasi ini menampilkan pembagian mahasiswa berdasarkan jenis kehadiran, misalnya *full-time* atau *part-time*. Informasi ini penting untuk mengetahui pola belajar mahasiswa serta keterkaitannya dengan potensi dropout.

4. **Distribusi Status Pernikahan (Marital Status)**
   Menunjukkan komposisi status pernikahan mahasiswa, seperti *single*, *married*, atau *divorced* dst. Data ini dapat digunakan untuk memahami tanggung jawab sosial mahasiswa yang mungkin memengaruhi performa akademik mereka.

5. **Distribusi Penerima Beasiswa (Scholarship Holder)**
   Memberikan informasi mengenai jumlah dan persentase mahasiswa yang menerima beasiswa. Ini dapat membantu institusi menilai efektivitas program beasiswa dan kaitannya dengan kelangsungan studi mahasiswa.

6. **Distribusi Kewarganegaraan (Nationality Student)**
   Menampilkan persebaran kewarganegaraan mahasiswa, baik mahasiswa lokal maupun internasional. Visualisasi ini penting untuk memahami keragaman dan internasionalisasi institusi.

7. **Distribusi Course Berdasarkan Status Kelas Internasional**
   Visualisasi ini menunjukkan jumlah mahasiswa dalam setiap program studi, dibedakan berdasarkan apakah program tersebut merupakan bagian dari kelas internasional atau bukan. Hal ini memberikan insight terkait preferensi mahasiswa terhadap kelas bertaraf internasional.

8. **Distribusi Course Berdasarkan Total Mahasiswa**
   Menyajikan jumlah mahasiswa di tiap program studi secara keseluruhan, yang dapat digunakan untuk evaluasi popularitas program dan alokasi sumber daya.

Halaman ini berperan sebagai **overview dashboard** dan menjadi dasar pemantauan strategis. Melalui visualisasi-visualisasi ini, pihak institusi dapat dengan mudah mengenali tren, ketimpangan, serta potensi masalah yang perlu ditindaklanjuti lebih lanjut.

<img width="982" alt="image" src="https://github.com/user-attachments/assets/381fcc67-7503-47c3-ab76-6d96cc6fdc41" />

### **Halaman Kedua: Analisis Status Student dengan Fokus pada Dropout**

Halaman kedua dari dashboard difokuskan secara khusus untuk menganalisis faktor-faktor yang berpotensi berhubungan dengan **tingginya angka dropout mahasiswa** di Jaya Jaya Institut. Visualisasi pada halaman ini dirancang untuk membantu pihak institusi memahami pola, hubungan, serta kemungkinan penyebab dari mahasiswa yang tidak menyelesaikan studinya. Beberapa aspek penting yang divisualisasikan meliputi:

1. **Gender vs Student Status**
   Visualisasi ini membandingkan status mahasiswa (aktif, dropout, lulus.) berdasarkan jenis kelamin. Tujuannya adalah untuk melihat apakah terdapat perbedaan signifikan antara mahasiswa laki-laki dan perempuan dalam hal kemungkinan dropout.

2. **Application Mode vs Student Status**
   Menunjukkan hubungan antara jalur masuk mahasiswa (misalnya reguler, beasiswa, transfer, dll.) dengan status mereka saat ini. Data ini dapat membantu mengidentifikasi apakah mode penerimaan tertentu memiliki risiko dropout yang lebih tinggi.

3. **Debtor vs Student Status**
   Menggambarkan status keuangan mahasiswa (apakah memiliki tunggakan pembayaran atau tidak) dan kaitannya dengan status akademik. Visualisasi ini berguna untuk mengetahui sejauh mana masalah finansial memengaruhi keberlangsungan studi mahasiswa.

4. **Age at Enrollment vs Student Status**
   Menampilkan hubungan antara usia saat pertama kali mendaftar dengan status terkini mahasiswa. Visualisasi ini dapat membantu mengidentifikasi apakah mahasiswa dengan usia tertentu lebih rentan terhadap dropout.

5. **Tuition Fees Status vs Student Status**
   Menganalisis dampak dari status pembayaran biaya kuliah terhadap kelangsungan studi mahasiswa. Data ini penting untuk memahami apakah keterlambatan atau ketidakmampuan membayar biaya kuliah berkorelasi dengan angka dropout.

6. **Scholarship Holder vs Student Status**
   Menunjukkan perbandingan status akademik antara mahasiswa penerima beasiswa dan yang tidak. Hal ini dapat digunakan untuk mengevaluasi efektivitas program beasiswa dalam membantu mahasiswa menyelesaikan studi.

7. **Academic Performance vs Student Status**
   Visualisasi ini menghubungkan performa akademik mahasiswa (misalnya IPK atau indeks prestasi lainnya) dengan status mereka. Tujuannya adalah untuk melihat apakah performa yang buruk berkontribusi langsung terhadap kemungkinan dropout.

Melalui visualisasi-visualisasi pada halaman ini, diharapkan pihak manajemen Jaya Jaya Institut dapat memperoleh **wawasan berbasis data** yang lebih mendalam tentang faktor-faktor yang memengaruhi dropout. Informasi ini akan sangat berguna dalam merancang strategi intervensi yang tepat sasaran, seperti bimbingan akademik, dukungan finansial, atau pendekatan personal bagi mahasiswa yang berisiko tinggi.

----------------------------------

## Machine Learning Prediction System

Untuk dapat membantu institusi dalam memprediksi kemungkinan jika siswanya akan dropout dan mencegah hal tersebut lebih dini, dapat menggunakan sistem prediksi yang telah dibangun. Sistem dibangun menggunakan Streamlit dan untuk menjalankan sistem tersebut secara local, dapat menjalankan kode berikut pada Terminal,

```bash
streamlit run app.py
```

Dan untuk menghentikan program aplikasi Streamlit dapat melalui `ctrl + c`.

Sistem prediksi tersebut juga dapat diakses secara langsung yang sudah di-deploy ke Streamlit Cloud melalui tautan.

----------------------------------

## Conclusion
Dari hasil proyek yang sudah dikerjakan dapat di simpulkan beberapa point sebagai berikut:
1.  Berdasarkan visualisasi jumlah mahasiswa perempuan yang lulus (1.661) jauh lebih tinggi dibandingkan dengan mahasiswa laki-laki yang lulus (548), menunjukkan tingkat kelulusan yang lebih dominan pada mahasiswa perempuan.
2.  Mahasiswa yang memiliki tunggakan cenderung lebih berisiko mengalami dropout dibandingkan yang tidak, sehingga status keuangan dapat menjadi indikator penting dalam upaya pencegahan dropout.
3.  Mahasiswa dengan usia masuk lebih muda (17–20 tahun) memiliki tingkat kelulusan yang tinggi, sedangkan dropout lebih banyak terjadi pada kelompok usia lebih tua (25 tahun ke atas), yang menunjukkan bahwa usia saat mendaftar berpengaruh terhadap keberhasilan studi.
4.  Mahasiswa yang tidak membayar uang kuliah memiliki kemungkinan lebih tinggi untuk dropout, menunjukkan bahwa ketidakmampuan membayar biaya pendidikan merupakan faktor signifikan terhadap putus studi.
5.  Mahasiswa penerima beasiswa cenderung memiliki tingkat kelulusan yang lebih tinggi dan angka putus studi yang lebih rendah dibandingkan dengan yang tidak menerima beasiswa.
6.  Mahasiswa yang lulus memiliki nilai rata-rata total dan jumlah mata kuliah yang disetujui (lulus) paling tinggi dibandingkan mahasiswa yang masih terdaftar maupun yang putus studi. Hal ini menunjukkan bahwa kinerja akademik yang baik berkontribusi besar terhadap kelulusan.

----------------------------------

### Rekomendasi Action Items
Berdasarkan kesimpulan dari hasil proyek analisis *dropout* di Jaya Jaya Institut, dan tujuan utama untuk mendeteksi serta memberikan bimbingan khusus kepada siswa yang berpotensi *dropout*, berikut adalah rekomendasi *action items* yang komprehensif:

**Rekomendasi Action Items untuk Jaya Jaya Institut:**

* **Action Item 1: Implementasi Sistem Deteksi Dini Risiko *Dropout* Berbasis Indikator Utama:**
    * **Deskripsi:** Kembangkan dan implementasikan sistem peringatan dini (early warning system) yang secara otomatis mengidentifikasi mahasiswa dengan risiko *dropout* tinggi berdasarkan kombinasi indikator yang telah diidentifikasi: tunggakan pembayaran, tidak membayar uang kuliah, usia masuk yang lebih tua (25 tahun ke atas), dan potensi penurunan kinerja akademik (misalnya, nilai rata-rata mata kuliah yang lulus di bawah standar tertentu).
    * **Tujuan:** Memungkinkan staf atau konselor untuk dengan cepat mengidentifikasi mahasiswa yang membutuhkan perhatian khusus sebelum mereka benar-benar *dropout*.

* **Action Item 2: Optimalisasi Program Dukungan Keuangan dan Beasiswa yang Bertarget:**
    * **Deskripsi:** Perluas dan promosikan program beasiswa atau bantuan keuangan khusus bagi mahasiswa yang menunjukkan kesulitan finansial atau memiliki tunggakan pembayaran, terutama bagi mereka yang sebelumnya tidak menerima beasiswa. Pertimbangkan untuk menyediakan skema cicilan yang lebih fleksibel atau bantuan dana darurat bagi kasus-kasus tertentu.
    * **Tujuan:** Mengurangi *dropout* yang disebabkan oleh kendala finansial dan meningkatkan retensi mahasiswa berprestasi yang terkendala biaya.

* **Action Item 3: Pengembangan Program Orientasi dan Dukungan Terpersonalisasi untuk Mahasiswa Usia Lanjut dan Non-Tradisional:**
    * **Deskripsi:** Rancang program orientasi dan dukungan akademik serta non-akademik yang spesifik untuk mahasiswa dengan usia masuk lebih tua (25 tahun ke atas). Program ini bisa mencakup sesi adaptasi perkuliahan, manajemen waktu, bimbingan karir yang relevan dengan pengalaman mereka, serta *peer support group* untuk membantu mereka berintegrasi dan mengatasi tantangan unik yang mungkin dihadapi.
    * **Tujuan:** Meningkatkan tingkat kelulusan pada kelompok usia yang rentan *dropout* ini dengan memberikan dukungan yang relevan dengan kebutuhan mereka.

* **Action Item 4: Penguatan Bimbingan Akademik dan Konseling Berbasis Kinerja:**
    * **Deskripsi:** Tingkatkan intensitas dan proaktivitas program bimbingan akademik, terutama bagi mahasiswa yang menunjukkan tanda-tanda penurunan nilai rata-rata atau jumlah mata kuliah yang disetujui. Perkenalkan sistem mentoring oleh dosen atau mahasiswa senior yang berhasil, dan dorong pemanfaatan layanan konseling untuk mengatasi tekanan akademik atau masalah pribadi yang memengaruhi studi.
    * **Tujuan:** Memastikan mahasiswa memiliki dukungan yang diperlukan untuk mempertahankan kinerja akademik yang baik, yang terbukti berkorelasi dengan kelulusan.

* **Action Item 5: Peningkatan Inisiatif Inklusi dan Dukungan Gender (khususnya untuk Pria):**
    * **Deskripsi:** Lakukan studi kualitatif (misalnya, wawancara atau focus group discussion) untuk memahami lebih dalam faktor-faktor yang menyebabkan rendahnya tingkat kelulusan di kalangan mahasiswa laki-laki dibandingkan perempuan. Berdasarkan temuan, kembangkan inisiatif yang ditargetkan, seperti kelompok studi khusus pria, program *mentoring* yang lebih berfokus pada tantangan yang dihadapi pria dalam lingkungan akademik, atau kampanye kesadaran akan pentingnya dukungan akademik dan psikologis bagi semua gender.
    * **Tujuan:** Mengurangi kesenjangan tingkat kelulusan antara gender dan meningkatkan retensi mahasiswa laki-laki.
