# ==============================================================================
# FILE: register_user.py
# FUNGSI: Skrip Pendaftaran (Enrollment) Wajah Pengguna Baru
# PENJELASAN TEORI AI:
# Agar AI bisa mengenali seseorang, sistem harus memiliki "data referensi" orang tersebut.
# Skrip ini memiliki 2 opsi:
# 1. Melalui Webcam Langsung (Rekomendasi):
#    Mengambil 5 foto dari beberapa sudut berbeda saat pengguna menekan tombol [SPACE].
#    Kelima vektor ciri (embedding) tersebut dihitung RATA-RATANYA (Averaging) lalu
#    di-normalisasi (L2 Normalization). Hasil rata-rata ini jauh lebih tahan terhadap
#    perubahan ekspresi (tersenyum/diam) dan sedikit kemiringan kepala!
# 2. Melalui Upload File Gambar:
#    Mendeteksi wajah pada file foto .jpg/.png yang sudah ada, lalu mengekstrak embedding-nya.
# ==============================================================================

import cv2                         # Library OpenCV untuk menangani kamera dan manipulasi gambar
import sys                         # Library sistem untuk fungsi keluar program (sys.exit)
import time                        # Library waktu
import numpy as np                 # Library manipulasi vektor dan matriks numerik
from pathlib import Path           # Library pengelolaan path folder
from config import FACES_DIR, SIMILARITY_THRESHOLD # Konfigurasi path dan threshold
from database.db import init_db, save_user, get_all_users # Modul database SQLite
from modules.detector import FaceDetector       # Modul detektor wajah YuNet
from modules.recognizer import FaceRecognizer   # Modul ekstraktor ciri SFace

def register_from_webcam(user_code: str, name: str, department: str, cam_index: int = 0):
    """
    LOGIKA PENDAFTARAN VIA WEBCAM:
    Membuka jendela kamera interaktif, mendeteksi keberadaan wajah secara langsung,
    dan memandu pengguna mengambil 7 sampel foto dengan variasi arahan pose wajah
    (Lurus, Tengok Kanan, Tengok Kiri, Atas, Bawah, Tilt, & Tersenyum).
    """
    print("\n[INFO] Membuka kamera webcam...")
    # Optimasi DirectShow untuk Windows 11
    if isinstance(cam_index, int) or str(cam_index).isdigit():
        cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        if isinstance(cam_index, int) or str(cam_index).isdigit():
            cap = cv2.VideoCapture(int(cam_index))
            
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera pada index {cam_index}!")
        print("Tips: Pastikan webcam terpasang dan tidak sedang dipakai aplikasi lain (Zoom, browser, dll).")
        return False

    # Inisialisasi model deteksi dan pengenal wajah
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Daftar petunjuk pose wajah secara berurutan agar dataset embedding kaya & berkualitas
    POSE_INSTRUCTIONS = [
        "1/7: Lurus ke Depan (Wajah Netral / Normal)",
        "2/7: Tengok Sedikit ke KANAN",
        "3/7: Tengok Sedikit ke KIRI",
        "4/7: Tengadahkan Kepala Sedikit ke ATAS",
        "5/7: Tundukkan Kepala Sedikit ke BAWAH",
        "6/7: Miringkan Kepala (Tilt Kanan / Kiri)",
        "7/7: Tersenyum / Ekspresi Alami (Depan)"
    ]
    
    collected_embeddings = []      # List wadah penampung 7 vektor embedding wajah
    sample_count_needed = len(POSE_INSTRUCTIONS) # Jumlah sampel foto (7 pose)
    captured_frame = None          # Variabel untuk menyimpan frame foto profil terakhir
    
    print(f"\n=== REGISTRASI WAJAH: {name} ({user_code}) ===")
    print(f"Petunjuk: Diperlukan {sample_count_needed} sampel pose wajah berbeda.")
    for pose in POSE_INSTRUCTIONS:
        print(f"  - {pose}")
    print("\nPosisikan wajah sesuai arahan di layar lalu tekan [SPACE] untuk setiap pose (Tekan [Q] untuk Batal).\n")
    
    # Perulangan berjalan terus selama sampel yang terkumpul masih kurang dari jumlah yang dibutuhkan
    while len(collected_embeddings) < sample_count_needed:
        # cap.read(): Membaca 1 frame gambar dari kamera
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari webcam!")
            break

        # Duplikasi frame untuk keperluan visual display (menggambar kotak & teks)
        display_frame = frame.copy()
        
        # Jalankan AI Detektor Wajah pada frame saat ini
        faces = detector.detect(frame)
        
        # ----------------------------------------------------------------------
        # MENGGAMBAR ANTARMUKA PANDUAN (UI OVERLAY)
        # ----------------------------------------------------------------------
        h, w = frame.shape[:2]     # Mengambil tinggi dan lebar gambar
        
        # Gambar kotak hitam transparan di bagian atas sebagai latar belakang teks petunjuk (tinggi 85px)
        cv2.rectangle(display_frame, (0, 0), (w, 85), (20, 20, 20), -1)
        
        current_pose = POSE_INSTRUCTIONS[len(collected_embeddings)]
        
        # Teks progress sampel
        cv2.putText(
            display_frame,
            f"Registrasi: {name} ({user_code}) | Progress: {len(collected_embeddings)}/{sample_count_needed}",
            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )
        # Teks petunjuk arahan pose (Mencolok dengan warna Cyan / Kuning)
        cv2.putText(
            display_frame,
            f"ARAHAN POSE: {current_pose}",
            (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )
        # Teks instruksi tombol aksi
        cv2.putText(
            display_frame,
            "Posisikan wajah sesuai arahan & tekan [SPACE] | [Q] Batal",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1
        )

        current_face_raw = None    # Variabel penyimpan data wajah yang siap diambil
        
        # Kondisi 1: Tepat 1 wajah terdeteksi (IDEAL)
        if len(faces) == 1:
            box = faces[0]["box"]
            current_face_raw = faces[0]["raw"]
            # Gambar kotak hijau di sekitar wajah
            cv2.rectangle(display_frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, "Wajah Terdeteksi (Siap)", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        # Kondisi 2: Terlalu banyak wajah di kamera (PERINGATAN)
        elif len(faces) > 1:
            cv2.putText(display_frame, "PERINGATAN: Lebih dari 1 wajah terdeteksi!", (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
        # Kondisi 3: Tidak ada wajah yang terlihat
        else:
            cv2.putText(display_frame, "Mencari wajah...", (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # cv2.imshow: Memunculkan jendela grafis ke layar komputer pengguna
        cv2.imshow("Registrasi Wajah - Absensi AI", display_frame)
        
        # cv2.waitKey(1): Menunggu input keyboard selama 1 milidetik
        key = cv2.waitKey(1) & 0xFF
        
        # Jika pengguna menekan tombol 'q' -> Batalkan pendaftaran
        if key == ord('q'):
            print("[INFO] Registrasi dibatalkan oleh pengguna.")
            cap.release()
            cv2.destroyAllWindows()
            return False
            
        # Jika pengguna menekan tombol SPASI (Kode ASCII 32)
        elif key == 32:
            if current_face_raw is not None:
                # Ekstrak 128 fitur embedding wajah menggunakan SFace
                feat = recognizer.extract_feature(frame, current_face_raw)
                if feat is not None:
                    curr_pose_desc = POSE_INSTRUCTIONS[len(collected_embeddings)]
                    collected_embeddings.append(feat) # Simpan vektor ke list
                    captured_frame = frame.copy()     # Simpan foto untuk avatar profil
                    print(f" -> [OK] Sampel [{len(collected_embeddings)}/{sample_count_needed}] Selesai: {curr_pose_desc}")
                    
                    # Efek Kilat Kamera (Visual Flash putih sejenak selama 80ms)
                    white_frame = np.ones_like(frame) * 255
                    cv2.imshow("Registrasi Wajah - Absensi AI", white_frame)
                    cv2.waitKey(80)
            else:
                print(" [!] Pastikan tepat 1 wajah terlihat jelas di kamera sebelum menekan SPACE.")

    # Lepaskan perangkat kamera dan tutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()
    
    # --------------------------------------------------------------------------
    # TAHAP AKHIR: PENGHITUNGAN RATA-RATA VEKTOR & PENYIMPANAN KE DATABASE
    # --------------------------------------------------------------------------
    if len(collected_embeddings) == sample_count_needed:
        # np.mean(axis=0): Menghitung rata-rata dari ke-7 vektor embedding
        avg_embedding = np.mean(collected_embeddings, axis=0)
        
        # L2-Normalization (Normalisasi Panjang Vektor = 1):
        # Membagi vektor dengan nilai normanya agar panjang vektor bernilai tepat 1.0.
        # Ini adalah syarat mutlak agar rumus Cosine Similarity berjalan akurat 100%.
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-6)
        
        # Simpan satu foto profil ke folder data/faces/{user_code}/profile.jpg
        user_folder = FACES_DIR / user_code
        user_folder.mkdir(parents=True, exist_ok=True)
        photo_path = user_folder / "profile.jpg"
        if captured_frame is not None:
            cv2.imwrite(str(photo_path), captured_frame)
            
        # Simpan seluruh data ke tabel 'users' di SQLite
        save_user(
            user_code=user_code,
            name=name,
            department=department,
            embedding=avg_embedding,
            photo_path=str(photo_path)
        )
        print(f"\n[SUKSES] Pengguna '{name}' ({user_code}) berhasil didaftarkan ke Database!")
        return True
        
    return False

def register_from_image(user_code: str, name: str, department: str, image_path: str):
    """
    LOGIKA PENDAFTARAN DARI FILE GAMBAR LOKAL:
    Membaca file foto dari harddisk (misal: pasfoto.jpg) tanpa menggunakan webcam.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"[ERROR] File foto '{image_path}' tidak ditemukan!")
        return False

    # cv2.imread: Membaca file gambar dari harddisk menjadi numpy array
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[ERROR] Gagal membaca file gambar '{image_path}'! Pastikan formatnya benar (JPG/PNG).")
        return False

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Deteksi wajah pada foto
    faces = detector.detect(frame)
    if len(faces) == 0:
        print("[ERROR] Tidak ada wajah manusia yang terdeteksi pada gambar ini!")
        return False
    elif len(faces) > 1:
        print("[ERROR] Terdapat lebih dari 1 wajah pada gambar. Gunakan foto dengan 1 wajah saja!")
        return False
        
    # Ekstraksi fitur embedding dari wajah tunggal tersebut
    feat = recognizer.extract_feature(frame, faces[0]["raw"])
    if feat is None:
        print("[ERROR] Gagal mengekstrak fitur wajah!")
        return False
        
    # Normalisasi vektor L2
    feat = feat / (np.linalg.norm(feat) + 1e-6)
    
    # Salin foto ke folder data/faces/
    user_folder = FACES_DIR / user_code
    user_folder.mkdir(parents=True, exist_ok=True)
    save_photo = user_folder / "profile.jpg"
    cv2.imwrite(str(save_photo), frame)
    
    # Simpan ke database SQLite
    save_user(
        user_code=user_code,
        name=name,
        department=department,
        embedding=feat,
        photo_path=str(save_photo)
    )
    print(f"\n[SUKSES] Pengguna '{name}' ({user_code}) berhasil didaftarkan dari file foto!")
    return True

# ------------------------------------------------------------------------------
# BAGIAN UTAMA (PROGRAM ENTRY POINT)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()                      # Pastikan tabel database sudah terbentuk
    print("========================================")
    print("       FORM REGISTRASI PENGGUNA AI      ")
    print("========================================")
    
    # Meminta input data dari keyboard terminal
    code = input("Masukkan NIP / NIM / ID Karyawan : ").strip()
    if not code:
        print("[ERROR] ID Karyawan tidak boleh kosong!")
        sys.exit(1)
        
    nama = input("Masukkan Nama Lengkap            : ").strip()
    if not nama:
        print("[ERROR] Nama tidak boleh kosong!")
        sys.exit(1)
        
    dept = input("Masukkan Divisi / Departemen      : ").strip() or "Umum"
    
    print("\nPilih Metode Pendaftaran Foto:")
    print("1. Ambil langsung via Webcam (Rekomendasi - Multi Sample)")
    print("2. Upload dari file gambar (JPG/PNG)")
    choice = input("Pilihan (1/2): ").strip()
    
    if choice == "1":
        register_from_webcam(code, nama, dept)
    elif choice == "2":
        img_p = input("Masukkan path lengkap file foto: ").strip()
        register_from_image(code, nama, dept, img_p)
    else:
        print("[ERROR] Pilihan tidak valid.")
