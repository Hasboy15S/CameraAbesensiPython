# ==============================================================================
# FILE: modules/attendance_engine.py
# FUNGSI: Mesin Logika Bisnis Absensi (Attendance Engine & Anti-Spam Logic)
# PENJELASAN:
# Modul ini menjembatani output dari AI (siapa yang terdeteksi) dengan sistem
# pencatatan database absensi sesungguhnya.
# 
# Mengapa kita butuh modul ini terpisah?
# Karena AI hanya bertugas mendeteksi: "Ini Budi dengan akurasi 88%".
# Tetapi sistem nyata butuh aturan bisnis:
# 1. Apakah Budi baru saja absen 3 detik yang lalu? (Jangan spam database!).
# 2. Apakah ini absensi Masuk atau Pulang? (Cek riwayat hari ini).
# 3. Ambil foto kamera saat Budi absen sebagai bukti fisik (Snapshot audit).
# 4. Simpan ke database SQLite.
# ==============================================================================

import cv2                         # Library OpenCV untuk memproses dan menyimpan gambar bukti
import time                        # Library waktu untuk menghitung detik cooldown
from datetime import datetime, date# Library untuk memformat tanggal dan jam
import sys
from pathlib import Path           # Library manipulasi path file
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Optional, Tuple, List
import threading
import requests
from config import SNAPSHOTS_DIR, ATTENDANCE_COOLDOWN_SECONDS, ATTENDANCE_MODE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from database.db import get_all_users, get_last_attendance, record_attendance

class AttendanceEngine:
    """
    Mesin pengatur alur absensi:
    - Mengelola cache daftar karyawan/user di memori RAM
    - Mencegah absensi berulang kali dalam jeda singkat (Cooldown / Debounce)
    - Menentukan status absensi (IN / OUT) secara otomatis atau manual
    - Mengambil dan menyimpan file foto snapshot kehadiran
    - Mencatat data ke tabel SQLite
    """
    def __init__(self, cooldown_seconds: int = ATTENDANCE_COOLDOWN_SECONDS):
        """
        KONSTRUKTOR:
        - cooldown_seconds: Jeda waktu minimal (dalam detik) sebelum orang yang sama
          bisa absen lagi (misal: 300 detik = 5 menit).
        - _last_attended: Dictionary di RAM yang memetakan {user_id: waktu_terakhir_absen_unix}.
        """
        self.cooldown_seconds = cooldown_seconds
        self.known_users: List[Dict] = []
        self._last_attended: Dict[int, float] = {}  # Cache waktu absensi terakhir per user
        self.reload_users()

    def reload_users(self):
        """
        LOGIKA CACHING:
        Mengambil seluruh user dari database dan menyimpannya di list self.known_users.
        Dengan cara ini, saat AI memproses 30 frame/detik, AI tidak menyentuh harddisk sama sekali
        karena seluruh data pembanding sudah siap di RAM.
        """
        self.known_users = get_all_users()
        print(f"[AttendanceEngine] Memuat {len(self.known_users)} pengguna terdaftar dari database.")

    def _send_telegram_notification(self, event_data: dict):
        """
        Kirim pesan ke Telegram secara asinkron agar tidak memblokir laju frame kamera.
        """
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return

        def send_task():
            try:
                status_label = "✅ MASUK" if event_data["status"] == "IN" else "👋 PULANG"
                caption = (
                    f"📸 [ABSEN {status_label}]\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"👤 Nama   : {event_data['name']}\n"
                    f"🔢 Kode   : {event_data['user_code']}\n"
                    f"🏢 Divisi : {event_data['department']}\n"
                    f"🕐 Waktu  : {event_data['timestamp']}\n"
                    f"📊 Akurasi: {event_data['confidence']*100:.1f}%\n"
                    f"━━━━━━━━━━━━━━━━"
                )
                
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
                
                snapshot_path = SNAPSHOTS_DIR / event_data["snapshot"]
                if snapshot_path.exists():
                    with open(snapshot_path, "rb") as photo:
                        files = {"photo": photo}
                        requests.post(url, data=data, files=files, timeout=10)
                else:
                    # Fallback ke pesan teks jika foto gagal dimuat
                    text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                    requests.post(text_url, data={"chat_id": TELEGRAM_CHAT_ID, "text": caption}, timeout=10)
            except Exception as e:
                print(f"[ERROR] Gagal mengirim notifikasi Telegram: {e}")

        # Jalankan di background thread
        threading.Thread(target=send_task, daemon=True).start()

    def process_face(self, user: Dict, confidence: float, frame=None, manual_mode: Optional[str] = None) -> Dict:
        """
        LOGIKA PEMROSESAN ABSENSI TERSTRUKTUR:
        Dipanggil setiap kali AI mengenali wajah seseorang di depan kamera.
        
        Parameter:
            user: Dictionary data profil pengguna dari recognizer.
            confidence: Skor kemiripan wajah (misal: 0.82 = 82%).
            frame: Gambar kamera saat ini (untuk disimpan sebagai foto snapshot).
            manual_mode: Mode absensi opsi ("AUTO", "IN", atau "OUT").
            
        Returns:
            Dictionary status lengkap:
            {
                "is_recorded": bool,
                "reason": "SUCCESS" | "COOLDOWN",
                "status": "IN" | "OUT",
                "last_status": "IN" | "OUT" | None,
                "cooldown_remaining": int,
                "cooldown_str": str,
                "message": str,
                "event_data": dict | None
            }
        """
        user_id = user["id"]
        user_code = user["user_code"]
        user_name = user["name"]
        now_ts = time.time()       # Waktu saat ini dalam detik Unix Epoch
        today_str = date.today().strftime("%Y-%m-%d")
        
        # Ambil catatan absensi terakhir user di hari ini
        last_log = get_last_attendance(user_id, today_str)
        last_status = last_log["status"] if last_log else None
        
        # ----------------------------------------------------------------------
        # TAHAP 1: FILTER ANTI-SPAM (COOLDOWN CHECK)
        # ----------------------------------------------------------------------
        last_time = self._last_attended.get(user_id, 0)
        selisih_waktu = now_ts - last_time
        sisa_cooldown = int(self.cooldown_seconds - selisih_waktu)
        
        # Jika sisa cooldown masih ada (> 0), kembalikan status COOLDOWN dengan info lengkap
        if sisa_cooldown > 0:
            mins, secs = divmod(sisa_cooldown, 60)
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            status_label = "MASUK" if last_status == "IN" else ("PULANG" if last_status == "OUT" else "SUDAH ABSEN")
            msg = f"Cooldown: {user_name} sudah absen [{status_label}] (Sisa jeda: {time_str})"
            
            return {
                "is_recorded": False,
                "reason": "COOLDOWN",
                "status": last_status or "IN",
                "last_status": last_status,
                "cooldown_remaining": sisa_cooldown,
                "cooldown_str": time_str,
                "message": msg,
                "event_data": None
            }

        # ----------------------------------------------------------------------
        # TAHAP 2: MENENTUKAN STATUS ABSENSI (IN / OUT)
        # ----------------------------------------------------------------------
        active_mode = manual_mode or ATTENDANCE_MODE
        if active_mode == "IN":
            status = "IN"
        elif active_mode == "OUT":
            status = "OUT"
        else: # Mode "AUTO"
            if last_log is None:
                # Belum absen hari ini -> MASUK
                status = "IN"
            elif last_log["status"] == "IN":
                # Absen terakhir MASUK -> PULANG
                status = "OUT"
            else:
                # Absen terakhir PULANG -> MASUK lagi
                status = "IN"

        # ----------------------------------------------------------------------
        # TAHAP 3: MENYIMPAN FOTO BUKTI (SNAPSHOT)
        # ----------------------------------------------------------------------
        snapshot_filename = ""
        if frame is not None:
            now_dt = datetime.now()
            time_tag = now_dt.strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"{time_tag}_{user_code}_{status}.jpg"
            snapshot_path = SNAPSHOTS_DIR / snapshot_filename
            cv2.imwrite(str(snapshot_path), frame)

        # ----------------------------------------------------------------------
        # TAHAP 4: MENYIMPAN KE DATABASE SQLITE
        # ----------------------------------------------------------------------
        log_id = record_attendance(
            user_id=user_id,
            status=status,
            confidence=confidence,
            snapshot_path=snapshot_filename
        )

        # ----------------------------------------------------------------------
        # TAHAP 5: UPDATE CACHE COOLDOWN DI RAM
        # ----------------------------------------------------------------------
        self._last_attended[user_id] = now_ts
        
        event_data = {
            "log_id": log_id,
            "user_id": user_id,
            "user_code": user_code,
            "name": user_name,
            "department": user.get("department", "Umum"),
            "status": status,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot": snapshot_filename
        }
        
        status_label = "MASUK" if status == "IN" else "PULANG"
        msg = f"Berhasil Absen [{status_label}]: {user_name} (Akurasi: {confidence*100:.1f}%)"
        
        # Kirim notifikasi Telegram
        self._send_telegram_notification(event_data)
        
        return {
            "is_recorded": True,
            "reason": "SUCCESS",
            "status": status,
            "last_status": status,
            "cooldown_remaining": self.cooldown_seconds,
            "cooldown_str": f"{self.cooldown_seconds // 60}m",
            "message": msg,
            "event_data": event_data
        }
