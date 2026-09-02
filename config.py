# ==============================================================================
# FILE: config.py
# FUNGSI: Pusat Pengaturan (Configuration File) untuk Sistem AI Camera Absensi
# PENJELASAN: 
# File ini berfungsi menyimpan semua konstanta, path folder, dan nilai ambang batas
# (threshold) di satu tempat. Dengan begitu, jika kita ingin mengganti kamera,
# mengubah folder, atau mengatur sensitivitas AI, kita cukup mengedit file ini saja
# tanpa perlu mengubah logika kode di file-file lainnya.
# ==============================================================================

import os                          # Library standar Python untuk berinteraksi dengan sistem operasi
from pathlib import Path           # Library modern Python untuk memanipulasi path direktori/file secara rapi

# ------------------------------------------------------------------------------
# 1. PENGATURAN STRUKTUR DIREKTORI (FOLDER PATHS)
# ------------------------------------------------------------------------------
# __file__ adalah variabel bawaan Python yang berisi path dari file config.py saat ini.
# .resolve() menghasilkan path absolut yang utuh (lengkap dari root sistem).
# .parent mengambil direktori induknya (yaitu folder AbsensiYolo).
BASE_DIR = Path(__file__).resolve().parent

# Mendefinisikan lokasi folder untuk menyimpan data sistem
DATA_DIR = BASE_DIR / "data"                       # Folder induk untuk seluruh data runtime
FACES_DIR = DATA_DIR / "faces"                     # Tempat menyimpan foto wajah referensi user
SNAPSHOTS_DIR = DATA_DIR / "snapshots"             # Tempat menyimpan foto bukti saat absensi berhasil
MODELS_DIR = BASE_DIR / "models"                   # Tempat menyimpan file bobot model AI (.onnx)
DB_PATH = BASE_DIR / "database" / "attendance.db"  # Lokasi file database SQLite

# Memastikan folder-folder di atas otomatis dibuat jika belum ada di komputer
# parents=True  -> membuat folder induknya sekaligus jika belum ada
# exist_ok=True -> jangan error jika folder tersebut sudah ada sebelumnya
FACES_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "database").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. PENGATURAN KAMERA (CAMERA SETTINGS)
# ------------------------------------------------------------------------------
# CAMERA_SOURCE:
# - Angka 0: Menggunakan webcam bawaan laptop atau webcam USB pertama.
# - Angka 1, 2, dst: Jika ada beberapa webcam USB terpasang.
# - String RTSP (contoh: "rtsp://admin:pass@192.168.1.100:554/stream"):
#   Jika ingin menghubungkan langsung ke CCTV / IP Camera di kantor/sekolah.
CAMERA_SOURCE = 0

# Resolusi ideal penangkapan frame video (lebar x tinggi dalam pixel)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ------------------------------------------------------------------------------
# 3. PENGATURAN KECERDASAN BUATAN (AI FACE RECOGNITION THRESHOLD)
# ------------------------------------------------------------------------------
# SIMILARITY_THRESHOLD (Nilai Ambang Batas Kemiripan Vektor Wajah):
# Model AI mengubah wajah menjadi array 128 angka (vektor embedding).
# Kita mengukur kemiripan dua wajah dengan 'Cosine Similarity' (rentang 0.0 s/d 1.0).
# - 1.0 artinya wajah identik persis (100% sama).
# - 0.0 artinya tidak ada kemiripan sama sekali.
# Nilai 0.55 adalah standar optimal untuk model SFace:
#   * Jika nilai dinaikkan (misal 0.65): AI sangat ketat, mengurangi salah kenal (false positive),
#     tapi mungkin agak susah mengenali jika pencahayaan kurang.
#   * Jika nilai diturunkan (misal 0.45): AI lebih mudah mengenali, tapi rawan tertukar dengan orang lain.
SIMILARITY_THRESHOLD = 0.55

# ------------------------------------------------------------------------------
# 4. LOGIKA BISNIS ABSENSI (ATTENDANCE LOGIC)
# ------------------------------------------------------------------------------
# ATTENDANCE_COOLDOWN_SECONDS (Durasi Jeda Anti-Spam):
# Kamera menangkap sekitar 30 frame per detik (30 FPS). Jika seseorang berdiri di depan
# kamera selama 5 detik, tanpa jeda sistem akan mencatat 150 kali absensi!
# Dengan cooldown 300 detik (5 menit), setelah "Budi" tercatat absen, wajah Budi tidak akan
# disimpan ulang ke database selama 5 menit berikutnya.
ATTENDANCE_COOLDOWN_SECONDS = 300  # 5 Menit (dalam satuan detik)

# ATTENDANCE_MODE:
# - "AUTO" : Logika pintar -> Jika di hari ini user belum absen, maka otomatis dicatat "MASUK".
#            Jika sudah pernah absen "MASUK", scan berikutnya dicatat "PULANG".
# - "IN"   : Mengunci sistem hanya untuk absensi masuk (misal di pintu gerbang pagi hari).
# - "OUT"  : Mengunci sistem hanya untuk absensi pulang (misal di sore hari).
ATTENDANCE_MODE = "AUTO"
