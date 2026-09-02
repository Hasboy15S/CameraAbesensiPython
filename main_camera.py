# ==============================================================================
# FILE: main_camera.py
# FUNGSI: Skrip Utama Pemantauan Kamera & Pencatatan Absensi Otomatis Realtime
# PENJELASAN ALUR KERJA (PIPELINE):
# 1. Buka aliran video dari webcam / RTSP IP Camera.
# 2. Ambil frame demi frame (sekitar 30 frame per detik).
# 3. Jalankan deteksi wajah (FaceDetector - YuNet) untuk mencari semua wajah di frame.
# 4. Untuk setiap wajah:
#    a. Ekstrak ciri vektor wajah (FaceRecognizer - SFace).
#    b. Cocokkan dengan database pengguna di RAM (Cosine Similarity).
#    c. Jika dikenal (skor >= 0.55):
#       - Gambar kotak HIJAU dan Nama.
#       - Kirim ke AttendanceEngine (cek cooldown, tentukan status Masuk/Pulang, simpan snapshot, catat ke DB).
#       - Tampilkan banner pop-up notifikasi hijau di layar video.
#    d. Jika tidak dikenal:
#       - Gambar kotak MERAH dan label "UNKNOWN".
# 5. Tampilkan info statistik (FPS, Jumlah Karyawan, Jam) di bagian atas layar (HUD).
# 6. Tekan 'q' untuk keluar, tekan 'r' untuk reload database.
# ==============================================================================

import cv2                         # Library Computer Vision untuk menampilkan jendela video
import time                        # Library waktu untuk mengukur durasi dan FPS
import numpy as np                 # Library komputasi matriks gambar
from datetime import datetime, date# Library format jam dan tanggal
from typing import Optional
from config import CAMERA_SOURCE, SIMILARITY_THRESHOLD # Konfigurasi sumber kamera & threshold AI
from database.db import init_db, get_today_attendance_summary # Modul database
from modules.detector import FaceDetector       # Modul detektor wajah
from modules.recognizer import FaceRecognizer   # Modul pengenal wajah
from modules.attendance_engine import AttendanceEngine # Modul mesin absensi & cooldown

def draw_hud(frame: np.ndarray, fps: float, user_count: int, today_att_count: int, notification: Optional[dict], current_mode: str = "AUTO"):
    """
    LOGIKA GRAFIS ANTARMUKA (HEAD-UP DISPLAY / HUD):
    Menggambar panel informasi transparan di bagian atas dan banner notifikasi di bagian bawah.
    """
    h, w = frame.shape[:2]         # Mendapatkan tinggi (h) dan lebar (w) frame video
    
    # --------------------------------------------------------------------------
    # 1. MEMBUAT LAYER TRANSPARAN (OVERLAY)
    # --------------------------------------------------------------------------
    overlay = frame.copy()         # Buat salinan kanvas frame
    
    # Gambar balok hitam di bagian paling atas (tinggi 55 pixel)
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    
    # --------------------------------------------------------------------------
    # 2. BANNER NOTIFIKASI EVENT ABSENSI (MUNCUL SELAMA 4 DETIK)
    # --------------------------------------------------------------------------
    if notification and (time.time() - notification.get("time", 0) < 4.0):
        # Warna Hijau terang jika "IN" (Masuk), Warna Oranye terang jika "OUT" (Pulang)
        banner_color = (34, 139, 34) if notification["status"] == "IN" else (0, 140, 255)
        # Gambar balok notifikasi di bagian bawah layar
        cv2.rectangle(overlay, (0, h - 50), (w, h), banner_color, -1)
        
    # Gabungkan layer overlay transparan ke frame asli (80% overlay + 20% gambar asli)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    
    # --------------------------------------------------------------------------
    # 3. MENULISKAN TEKS INFORMASI (HEADER TEXT)
    # --------------------------------------------------------------------------
    now_str = datetime.now().strftime("%H:%M:%S")
    # Judul Sistem di pojok kiri atas
    cv2.putText(frame, "AI CAMERA ABSENSI", (15, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2)
    
    # Mode badge warna
    mode_color = (0, 255, 255) if current_mode == "AUTO" else ((0, 255, 0) if current_mode == "IN" else (0, 140, 255))
    
    # Statistik real-time di sebelah kanan judul
    stats_text = f"MODE: [{current_mode}] | FPS: {fps:.1f} | Terdaftar: {user_count} | Absen Hari Ini: {today_att_count} | {now_str}"
    cv2.putText(frame, stats_text, (250, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # --------------------------------------------------------------------------
    # 4. MENULISKAN PESAN PADA BANNER NOTIFIKASI
    # --------------------------------------------------------------------------
    if notification and (time.time() - notification.get("time", 0) < 4.0):
        status_label = "ABSEN MASUK" if notification["status"] == "IN" else "ABSEN PULANG"
        notif_msg = f"[SUCCESS] {status_label} BERHASIL: {notification['name']} ({notification['user_code']}) - {notification['timestamp']}"
        cv2.putText(frame, notif_msg, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

def run_camera():
    """
    FUNGSI UTAMA (REALTIME LOOP):
    Menghubungkan kamera, menjalankan inferensi AI terus menerus,
    dan menampilkan jendela pemantauan interaktif dengan status yang jelas.
    """
    init_db()                      # Pastikan database dan tabel sudah siap
    print("==================================================")
    print("       MEMULAI AI CAMERA ABSENSI REALTIME         ")
    print("==================================================")
    
    # Optimasi thread OpenCV & OpenCL hardware acceleration untuk Windows 11
    cv2.setNumThreads(4)
    cv2.ocl.setUseOpenCL(True)

    print(f"[INFO] Menghubungkan ke sumber kamera: {CAMERA_SOURCE}")
    
    # Deteksi jika sumber kamera adalah webcam lokal (0, 1, dst.)
    if isinstance(CAMERA_SOURCE, int) or str(CAMERA_SOURCE).isdigit():
        cam_id = int(CAMERA_SOURCE)
        # DirectShow (CAP_DSHOW) menghilangkan delay buffering & memberikan FPS tinggi di Windows 11
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Format MJPG ultra-fast
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                           # Minimalkan lag buffer frame
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        # Fallback coba tanpa CAP_DSHOW jika driver webcam legacy
        if isinstance(CAMERA_SOURCE, int) or str(CAMERA_SOURCE).isdigit():
            cap = cv2.VideoCapture(int(CAMERA_SOURCE))
            
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka kamera ({CAMERA_SOURCE})!")
        print("Solusi: Cek apakah webcam laptop terpasang atau ganti CAMERA_SOURCE di config.py.")
        return

    # Inisialisasi komponen modular
    detector = FaceDetector()      # Deteksi wajah YuNet
    recognizer = FaceRecognizer()  # Pengenal wajah SFace
    engine = AttendanceEngine()    # Logika bisnis absensi & cooldown
    
    # Daftar opsi mode absensi
    MODE_OPTIONS = ["AUTO", "IN", "OUT"]
    current_mode_idx = 0
    
    # Hitung data awal absensi yang sudah terjadi hari ini
    today_records = get_today_attendance_summary()
    today_count = len(today_records)
    
    # Variabel penghitung FPS (Frames Per Second)
    prev_time = time.time()
    fps = 0.0
    active_notification = None     # Wadah data notifikasi terbaru yang sedang aktif
    
    print("\n[INFO] Kamera Aktif & Siap Beroperasi!")
    print("Petunjuk Kontrol Keyboard:")
    print(" - Tekan [M] untuk Ganti Mode Absensi (AUTO -> IN ONLY -> OUT ONLY)")
    print(" - Tekan [R] untuk Reload data pengguna dari database")
    print(" - Tekan [Q] untuk Keluar\n")

    # ==========================================================================
    # WHILE LOOP UTAMA (BERJALAN SEPANJANG WAKTU / REALTIME)
    # ==========================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Gagal membaca frame kamera, mencoba menyambung kembali...")
            time.sleep(0.1)
            continue

        # ----------------------------------------------------------------------
        # 1. PENGHITUNGAN FPS SECARA HALUS
        # ----------------------------------------------------------------------
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (cur_time - prev_time + 1e-6))
        prev_time = cur_time
        
        current_mode = MODE_OPTIONS[current_mode_idx]

        # ----------------------------------------------------------------------
        # 2. TAHAP DETEKSI WAJAH (YUNET)
        # ----------------------------------------------------------------------
        faces = detector.detect(frame)
        
        # ----------------------------------------------------------------------
        # 3. TAHAP PENGENALAN TIAP WAJAH (SFACE) & ATURAN ABSENSI
        # ----------------------------------------------------------------------
        for face_info in faces:
            bbox = face_info["box"]    # Koordinat kotak (x, y, lebar, tinggi)
            face_raw = face_info["raw"]# Data mentah 15 elemen untuk alignment
            x, y, w, h = bbox
            
            # Ekstraksi 128-vektor embedding dari wajah
            feat = recognizer.extract_feature(frame, face_raw)
            
            # Cocokkan vektor terhadap pengguna terdaftar
            user, score = recognizer.identify(feat, engine.known_users, threshold=SIMILARITY_THRESHOLD)
            
            # ------------------------------------------------------------------
            # PENENTUAN WARNA KOTAK & INFORMASI TEKS DI ATAS WAJAH
            # ------------------------------------------------------------------
            if user:
                # Memproses data wajah ke AttendanceEngine
                att_res = engine.process_face(user, score, frame, manual_mode=current_mode)
                
                # KONDISI A: BARU BERHASIL DICATAT KE DATABASE
                if att_res["is_recorded"]:
                    today_count += 1
                    status_lbl = "MASUK" if att_res["status"] == "IN" else "PULANG"
                    color = (0, 255, 0) if att_res["status"] == "IN" else (0, 140, 255) # Hijau (IN) / Oranye (OUT)
                    name_label = f"[✓ ABSEN {status_lbl}] {user['name']} ({score*100:.0f}%)"
                    thickness = 3
                    
                    active_notification = {
                        "name": user["name"],
                        "user_code": user["user_code"],
                        "status": att_res["status"],
                        "timestamp": att_res["event_data"]["timestamp"],
                        "time": time.time()
                    }
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {att_res['message']}")
                    
                # KONDISI B: DALAM PERIODE COOLDOWN (SUDAH ABSEN SEBELUMNYA)
                elif att_res["reason"] == "COOLDOWN":
                    last_lbl = "MASUK" if att_res["last_status"] == "IN" else ("PULANG" if att_res["last_status"] == "OUT" else "SUDAH ABSEN")
                    color = (255, 200, 0) # Warna Cyan/Kuning (Tanda sudah absen)
                    name_label = f"{user['name']} | [SUDAH ABSEN {last_lbl}] (Jeda {att_res['cooldown_str']})"
                    thickness = 2
                    
                else:
                    color = (0, 255, 0)
                    name_label = f"{user['name']} ({score*100:.0f}%)"
                    thickness = 2
                    
            # KONDISI C: WAJAH TIDAK DIKENALI (BELUM TERDAFTAR)
            else:
                color = (0, 0, 255)    # Warna Merah
                name_label = f"[UNKNOWN] Belum Terdaftar ({score*100:.0f}%)"
                thickness = 2

            # ------------------------------------------------------------------
            # MENGGAMBAR BOUNDING BOX & LABEL NAMA DI ATAS WAJAH
            # ------------------------------------------------------------------
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Hitung ukuran piksel teks label
            label_size, _ = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            label_w, label_h = label_size
            
            # Gambar balok warna solid di atas kepala sebagai latar belakang teks
            cv2.rectangle(frame, (x, max(0, y - label_h - 12)), (x + label_w + 10, y), color, -1)
            # Tulis teks nama & status (Warna hitam agar kontras dan mudah dibaca)
            cv2.putText(frame, name_label, (x + 5, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

        # ----------------------------------------------------------------------
        # 4. TAMPILKAN PANEL STATISTIK (HUD) & NOTIFIKASI
        # ----------------------------------------------------------------------
        draw_hud(frame, fps, len(engine.known_users), today_count, active_notification, current_mode)

        # Tampilkan hasil olahan gambar ke jendela GUI
        cv2.imshow("AI Camera Absensi - Realtime", frame)
        
        # ----------------------------------------------------------------------
        # 5. PENANGANAN TOMBOL KEYBOARD
        # ----------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Menghentikan sistem kamera...")
            break
        elif key == ord('r'):
            engine.reload_users()
            print("[INFO] Data pengguna berhasil dimuat ulang dari database.")
        elif key == ord('m') or key == ord('M'):
            current_mode_idx = (current_mode_idx + 1) % len(MODE_OPTIONS)
            print(f"[INFO] Mode Absensi diubah ke: [{MODE_OPTIONS[current_mode_idx]}]")

    # Bersihkan memori dan lepaskan kamera
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai dengan aman.")

if __name__ == "__main__":
    run_camera()
