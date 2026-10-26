# 📘 Panduan Lengkap Setup & Eksekusi ETS di Google Colab (+ Integrasi VS Code Lokal)

Paket ini sudah disesuaikan 100% dengan **materi kuliah Bapak Djoko Purwanto (Teknik Elektro ITS)** dan struktur asli dataset **[Indoor Object Detection Dataset — Mendeley v1](https://data.mendeley.com/datasets/3ggxwf2vpr/1)**.

---

## 📂 Daftar File yang Sudah Disiapkan untuk Anda

1. **`ETS_Object_Detection_Indoor_Colab.ipynb`** *(File Utama untuk Google Colab)*
   Notebook lengkap dari A–Z: instalasi Ultralytics, *auto-download* dataset dari server Mendeley (~20–40 detik), konversi otomatis Pascal VOC `.xml` $\rightarrow$ YOLO `.txt`, pembuatan `data.yaml`, plot distribusi kelas, *transfer learning* (`yolo12n.pt` / `yolo11n.pt`), evaluasi (`results.png`, `confusion_matrix.png`, perbandingan validasi), uji gambar *test*, tangkap *webcam* di browser Colab, hingga *auto-download* ZIP seluruh hasil ujian.
2. **`scripts_lokal/`** *(Cadangan jika diminta kumpulkan file `.py` atau demo Webcam Real-Time di Laptop)*
   * `1_setup_dataset_mendeley.py` — Konverter dataset VOC `.xml` ke YOLO `.txt` di PC lokal.
   * `2_train_yolo.py` — Script training persis format **Slide 31**.
   * `3_predict_image.py` — Script uji deteksi gambar statis persis format **Slide 36**.
   * `4_indoor_security_webcam.py` — Aplikasi *Smart Indoor Security Camera & Object Counter* via Webcam (gabungan **Slide 29, 37, & 38**: hitung objek *real-time*, simpan `detection_result.jpg`, dan rekam video `.mp4` otomatis ketika objek target terdeteksi).

---

## 🚀 Langkah Demi Langkah Setup di Google Colab (Dari Nol)

### Langkah 1: Buka Google Colab & Upload Notebook
1. Unduh file **`ETS_Object_Detection_Indoor_Colab.ipynb`** dari workspace ini ke laptop Anda.
2. Buka **[https://colab.research.google.com](https://colab.research.google.com)** dan login dengan akun Google Anda.
3. Pada jendela awal yang muncul, pilih tab **Upload** $\rightarrow$ pilih file **`ETS_Object_Detection_Indoor_Colab.ipynb`** *(atau klik menu **File $\rightarrow$ Upload notebook**).*

### Langkah 2: WAJIB Aktifkan Akselerator GPU T4!
1. Di bagian atas Google Colab, klik menu **Runtime $\rightarrow$ Change runtime type** *(atau **Waktu proses $\rightarrow$ Ubah jenis waktu proses**)*.
2. Pada bagian **Hardware accelerator**, pilih **`T4 GPU`**.
3. Klik **Save**.
4. Klik tombol **Connect** di pojok kanan atas sampai muncul tanda centang hijau `RAM / Disk (T4)`.

### Langkah 3: Sesuaikan "Panel Kontrol Soal ETS" (Bagian 3 & Bagian 4 Notebook)
Sebelum menekan *Run*, baca soal ETS Anda terlebih dahulu dan cek 3 variabel berikut:

| Skenario Soal Saat ETS | Apa yang Perlu Diubah di Notebook? |
| :--- | :--- |
| **Diminta semua 7 kelas objek indoor** | Biarkan `SELECTED_CLASSES` apa adanya (`['bin', 'chair', 'door', 'fire extinguisher', 'human', 'shelf', 'table']`). |
| **Hanya diminta beberapa objek tertentu** *(misal: hanya manusia, pintu, dan APAR)* | Di **Bagian 3**, ubah menjadi:<br>`SELECTED_CLASSES = ['human', 'door', 'fire extinguisher']`<br>dan ubah `SKIP_EMPTY_IMAGES_WHEN_SUBSETTING = True`. Script otomatis mengatur ulang ID kelas (`0, 1, 2`) dan `nc: 3` di `data.yaml`! |
| **Waktu ETS *live* di kelas terbatas (misal 60–90 menit)** | Di **Bagian 3**, isi `MAX_TRAIN_IMAGES = 2000` (atau `1500`).<br>Di **Bagian 4**, pasang `EPOCHS = 30`, `BATCH_SIZE = 32`.<br>⏱️ *Estimasi selesai training di GPU T4 hanya **~5–7 menit**!* |
| **Waktu ETS longgar / *Take-Home*** | Di **Bagian 3**, biarkan `MAX_TRAIN_IMAGES = None` (memakai seluruh 12.012 gambar).<br>Di **Bagian 4**, pasang `EPOCHS = 25` s/d `50` (~25–50 menit di GPU T4). |
| **Diminta pakai YOLOv11 (`yolo11n.pt`) atau YOLOv12 (`yolo12n.pt`)** | Di **Bagian 4**, ubah `PRETRAINED_MODEL = "yolo11n.pt"` (Slide 25) atau `"yolo12n.pt"` (Slide 31). |

### Langkah 4: Jalankan Sel Berurutan & Unduh Hasil
1. Tekan **Shift + Enter** mulai dari **Bagian 1** hingga **Bagian 8** (atau klik **Runtime $\rightarrow$ Run all** jika tidak perlu upload foto manual di Bagian 7).
2. Pada **Bagian 8**, browser Anda akan otomatis mengunduh file **`Hasil_ETS_ObjectDetection.zip`** yang sudah berisi:
   * `MyYOLOModel.pt` (bobot model terbaik hasil training GPU Colab)
   * `data.yaml`
   * `results.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `PR_curve.png`, `F1_curve.png`
   * `val_batch0_labels.jpg`, `val_batch0_pred.jpg`, `val_comparison_slide35.png`
   * `class_distribution.png`, `prediction_result.jpg`, `comparison_coco_vs_custom.png`

---

## 💻 Cara Menjalankan Aplikasi Webcam Real-Time di Laptop (Jika Diminta seperti Slide 29, 37, atau 38)

Karena server cloud Google Colab tidak bisa melakukan *streaming video webcam* OpenCV `cv2.imshow()` secara *real-time* 30 FPS, trik paling cepat & ringan adalah **melatih model di GPU Colab, lalu menjalankan inferensi webcam di laptop**:

1. Buka terminal / Command Prompt di laptop (VS Code), lalu instal library:
   ```bash
   pip install ultralytics opencv-python
   ```
2. Buat satu folder proyek (misal `NamaAnda_NRP/`), lalu letakkan di dalamnya:
   * File **`MyYOLOModel.pt`** (hasil ekstrak dari `Hasil_ETS_ObjectDetection.zip` yang diunduh dari Colab).
   * File **`4_indoor_security_webcam.py`** dari folder `scripts_lokal/` (jangan lupa isi **Nama & NRP** pada baris komentar paling atas!).
3. Sesuaikan `TRIGGER_CLASSES = {"human", "fire extinguisher"}` di dalam `4_indoor_security_webcam.py` sesuai objek yang diminta dosen.
4. Jalankan di terminal VS Code:
   ```bash
   python 4_indoor_security_webcam.py
   ```
   * Tekan tombol **`s`** pada jendela kamera untuk menyimpan *screenshot* `detection_result.jpg`.
   * Tekan tombol **`q`** untuk keluar.
   * Jika objek target terdeteksi, program otomatis menyimpan bukti foto & rekaman video `.mp4` di folder `captures_ets/`!
5. Kompres folder proyek menjadi **`NamaAnda_NRP.zip`** untuk diunggah ke **MyITS Classroom**.

---

## 📝 Contekan Poin Analisis Evaluasi (Siap Pakai untuk Laporan / Jawaban Soal Teori)

Jika dosen meminta analisis tertulis terhadap dataset dan hasil training (mengacu **Slide 21, 26, 32–35**):

1. **Evaluasi Kualitas Dataset terhadap Kriteria Slide 21:**
   * **Jumlah Gambar:** Sangat memadai (**12.747 gambar** total, melampaui batas minimum 1.000–5.000 gambar untuk objek beragam).
   * **Pre-processing & Augmentasi:** Seluruh gambar telah diseragamkan ke resolusi `640×640` piksel dengan augmentasi komprehensif (flip horizontal/vertikal, rotasi 90° dan acak ±15°, *shear* ±10°, serta penyesuaian kecerahan ±15%) sehingga meningkatkan kemampuan generalisasi model pada pencahayaan/sudut kamera dalam ruangan.
   * **Catatan Kritis (*Split* & *Class Imbalance*):** Pembagian dataset asli adalah **94% Train (12.012) : 4% Valid (490) : 2% Test (245)**, sedikit berbeda dari rasio umum Slide 21 (70–80% : 10–15% : 10–15%). Selain itu terdapat ketidakseimbangan kelas (*class imbalance*), di mana kelas `chair` dan `door` sangat dominan sedangkan `shelf` dan `fire extinguisher` memiliki jumlah *instance* jauh lebih sedikit.
2. **Cara Membaca Metrik Evaluasi (Slide 26 & 32):**
   * **IoU (*Intersection over Union*):** Rasio luas irisan terhadap gabungan antara *bounding box* prediksi dan *ground truth*.
   * **Precision (`Box(P)`):** Tingkat ketepatan prediksi positif model ($\frac{TP}{TP + FP}$). Nilai tinggi berarti sedikit *false alarm*.
   * **Recall (`R`):** Kemampuan model menemukan seluruh objek asli yang ada di gambar ($\frac{TP}{TP + FN}$). Kelas dengan data sedikit seperti `shelf` cenderung memiliki *Recall* lebih rendah dibanding `chair` atau `door`.
   * **mAP50 vs mAP50-95:** `mAP50` menghitung rata-rata *Average Precision* seluruh kelas pada ambang batas $\text{IoU} \ge 0.50$, sedangkan `mAP50-95` merata-ratakan AP dari $\text{IoU} = 0.50$ hingga $0.95$ (kelipatan $0.05$) sehingga menguji kepresisian posisi *bounding box* secara jauh lebih ketat.
3. **Manfaat *Transfer Learning* dibanding Model Pre-trained MS COCO:**
   * Model *pre-trained* bawaan (`yolo11n.pt` / `yolo12n.pt`) dilatih pada 80 kelas MS COCO yang hanya mencakup `person`, `chair`, dan `dining table`, namun **sama sekali tidak memiliki kelas `door`, `bin`, `fire extinguisher`, dan `shelf`**. Dengan *transfer learning*, fitur ekstraktor awal tetap dimanfaatkan sehingga konvergensi cepat, sekaligus kepala detektor (*detection head*) mampu mengenali ke-7 objek spesifik dalam ruangan tersebut.
