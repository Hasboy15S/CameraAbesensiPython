# ==============================================================================
# FILE: database/db.py
# FUNGSI: Pengelola Database SQLite untuk Sistem Absensi AI
# PENJELASAN:
# File ini menangani seluruh operasi Create, Read, Update ke file SQLite database.
# Fitur spesial: Kita menyimpan 'Vektor Wajah' (Numpy Array) langsung ke dalam
# database menggunakan tipe data BLOB (Binary Large Object), sehingga tidak perlu
# database terpisah untuk menyimpan vektor AI.
# ==============================================================================

import sqlite3                     # Modul bawaan Python untuk database SQL lokal tanpa server tambahan
import numpy as np                 # Library komputasi numerik, digunakan untuk mengolah matriks & vektor wajah
import io                          # Library bawaan untuk membaca/menulis data biner di memori RAM (Buffer)
from datetime import datetime, date# Modul untuk manipulasi tanggal dan waktu
from typing import List, Dict, Optional, Tuple # Type hint untuk memudahkan pembacaan tipe data fungsi
import sys
from pathlib import Path           # Library path direktori
# Memastikan root project masuk ke sys.path agar import config selalu berhasil
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH         # Mengambil path lokasi database dari file config.py

# ------------------------------------------------------------------------------
# 1. SERIALISASI & DESERIALISASI NUMPY ARRAY KE DATABASE BLOB
# ------------------------------------------------------------------------------
def adapt_array(arr: np.ndarray) -> bytes:
    """
    LOGIKA:
    SQLite tidak bisa langsung menyimpan tipe data 'np.ndarray' (vektor AI).
    Oleh karena itu, fungsi ini mengubah array angka menjadi aliran bytes biner murni.
    - io.BytesIO(): Membuat wadah memori RAM sementara seperti file virtual.
    - np.save(): Menyimpan array ke wadah RAM tersebut dalam format biner berekstensi .npy.
    - sqlite3.Binary(): Membungkus bytes tersebut agar dikenali sebagai kolom BLOB oleh SQLite.
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)                    # Kembalikan kursor pembacaan ke awal file RAM
    return sqlite3.Binary(out.read())

def convert_array(text: bytes) -> np.ndarray:
    """
    LOGIKA:
    Kebalikan dari adapt_array. Saat kita membaca kolom BLOB dari database SQLite,
    fungsi ini mengubah kembali bytes mentah menjadi array angka numpy (np.ndarray)
    sehingga bisa langsung dihitung oleh AI (misal untuk Cosine Similarity).
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# ------------------------------------------------------------------------------
# 2. MANAJEMEN KONEKSI DATABASE
# ------------------------------------------------------------------------------
def get_connection():
    """
    LOGIKA:
    Membuka koneksi ke file database SQLite di DB_PATH.
    - conn.row_factory = sqlite3.Row:
      Secara default, SQLite mengembalikan data dalam tuple angka, misal (1, "Hasboy").
      Dengan 'sqlite3.Row', hasil query bisa diakses dengan nama kolom seperti dictionary,
      contoh: row["name"] atau row["user_code"], sehingga kode jauh lebih mudah dibaca.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ------------------------------------------------------------------------------
# 3. PEMBUATAN TABEL (INITIALIZATION)
# ------------------------------------------------------------------------------
def init_db():
    """
    LOGIKA:
    Membuat 2 tabel utama jika belum ada:
    1. Tabel 'users':
       - id: Nomor urut unik otomatis (Primary Key).
       - user_code: NIP/NIM/ID unik (tidak boleh ada yang kembar / UNIQUE).
       - name: Nama lengkap orang yang didaftarkan.
       - department: Divisi / Kelas.
       - embedding: Fitur vektor wajah 128-dimensi yang disimpan sebagai BLOB.
       - photo_path: Lokasi file foto profil di harddisk.
    2. Tabel 'attendance_logs':
       - id: Nomor urut log absensi.
       - user_id: Menghubungkan log ke pengguna tertentu (Foreign Key ke users.id).
       - date: Tanggal absensi format YYYY-MM-DD (untuk pencarian cepat per hari).
       - timestamp: Waktu lengkap absensi format YYYY-MM-DD HH:MM:SS.
       - status: "IN" (Masuk) atau "OUT" (Pulang).
       - confidence: Tingkat keyakinan kemiripan wajah saat AI mendeteksi (misal: 0.85 atau 85%).
       - snapshot_path: Nama file foto kamera saat orang tersebut absen.
    """
    with get_connection() as conn: # 'with' otomatis menutup koneksi setelah blok selesai (aman dari memory leak)
        cursor = conn.cursor()
        
        # Tabel 1: Data Pengguna / Karyawan
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT DEFAULT 'Umum',
                embedding BLOB NOT NULL,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabel 2: Riwayat Kehadiran (Log Absensi)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                snapshot_path TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # INDEXING:
        # Menambahkan Index pada kolom 'user_id' dan 'date' agar pencarian data
        # "apakah user X sudah absen hari ini" berjalan secepat kilat (dalam hitungan milidetik)
        # meskipun datanya sudah mencapai ratusan ribu baris.
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_user_date ON attendance_logs (user_id, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_logs (date)")
        conn.commit()              # Menyimpan perubahan secara permanen ke file database

# ------------------------------------------------------------------------------
# 4. OPERASI PADA TABEL USERS (PENDAFTARAN & PENGAMBILAN DATA)
# ------------------------------------------------------------------------------
def save_user(user_code: str, name: str, department: str, embedding: np.ndarray, photo_path: str = "") -> int:
    """
    LOGIKA:
    Menyimpan data user baru ke database.
    - 'ON CONFLICT(user_code) DO UPDATE':
      Jika NIP/NIM yang dimasukkan sudah pernah ada di database sebelumnya,
      sistem tidak akan crash atau error, melainkan otomatis memperbarui (update)
      nama dan data vektor wajah terbarunya.
    """
    embedding_blob = adapt_array(embedding) # Ubah array numpy ke format BLOB biner
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (user_code, name, department, embedding, photo_path)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_code) DO UPDATE SET
                name=excluded.name,
                department=excluded.department,
                embedding=excluded.embedding,
                photo_path=excluded.photo_path
        """, (user_code, name, department, embedding_blob, photo_path))
        conn.commit()
        return cursor.lastrowid    # Mengembalikan ID angka user yang baru dibuat/diupdate

def get_all_users() -> List[Dict]:
    """
    LOGIKA:
    Mengambil SELURUH data user dari database ke dalam memori RAM saat program kamera dinyalakan.
    Ini penting agar saat kamera membaca wajah, AI tidak perlu query database berulang kali
    tiap frame (karena membaca disk lambat). AI mencocokkan wajah langsung dari memori RAM.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, user_code, name, department, embedding, photo_path FROM users")
        rows = cursor.fetchall()
        users = []
        for r in rows:
            users.append({
                "id": r["id"],
                "user_code": r["user_code"],
                "name": r["name"],
                "department": r["department"],
                "embedding": convert_array(r["embedding"]), # Ubah BLOB kembali jadi array numpy
                "photo_path": r["photo_path"]
            })
        return users

# ------------------------------------------------------------------------------
# 5. OPERASI PADA TABEL ATTENDANCE_LOGS (LOGIKA PENCATATAN KEHADIRAN)
# ------------------------------------------------------------------------------
def get_last_attendance(user_id: int, target_date: str) -> Optional[Dict]:
    """
    LOGIKA:
    Mencari catatan absensi terakhir dari seorang user pada tanggal tertentu.
    Digunakan untuk menentukan status absensi berikutnya:
    - Jika belum pernah absen hari ini -> Status berikutnya adalah "MASUK" (IN).
    - Jika terakhir absen adalah "IN" -> Status berikutnya adalah "PULANG" (OUT).
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, user_id, date, timestamp, status, confidence, snapshot_path
            FROM attendance_logs
            WHERE user_id = ? AND date = ?
            ORDER BY id DESC LIMIT 1
        """, (user_id, target_date))
        row = cursor.fetchone()
        if row:
            return dict(row)       # Mengembalikan baris data dalam format dictionary Python
        return None                # Mengembalikan None jika user belum pernah absen di hari ini

def record_attendance(user_id: int, status: str, confidence: float, snapshot_path: str = "") -> int:
    """
    LOGIKA:
    Mencatat transaksi absensi baru ke dalam database secara permanen.
    Merekam waktu presisi (jam, menit, detik) dan skor akurasi deteksi AI.
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")               # Contoh: "2026-09-03"
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")  # Contoh: "2026-09-03 08:15:30"
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO attendance_logs (user_id, date, timestamp, status, confidence, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, today_str, timestamp_str, status, round(confidence, 4), snapshot_path))
        conn.commit()
        return cursor.lastrowid

def get_today_attendance_summary() -> List[Dict]:
    """
    LOGIKA:
    Mengambil seluruh rekap absensi hari ini yang digabungkan (JOIN) dengan data profil
    dari tabel 'users'. Dengan SQL JOIN, kita bisa langsung menampilkan Nama dan Divisi
    karyawan di layar kamera tanpa perlu 2 kali query terpisah.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id, a.user_id, u.user_code, u.name, u.department, 
                   a.timestamp, a.status, a.confidence, a.snapshot_path
            FROM attendance_logs a
            JOIN users u ON a.user_id = u.id
            WHERE a.date = ?
            ORDER BY a.id DESC
        """, (today_str,))
        return [dict(r) for r in cursor.fetchall()]

# ------------------------------------------------------------------------------
# BAGIAN TESTING MANDIRI (EXECUTABLE BLOCK)
# ------------------------------------------------------------------------------
# Blok di bawah hanya akan dijalankan jika file ini dieksekusi langsung
# dengan perintah: python3 database/db.py (tidak dijalankan saat di-import oleh file lain)
if __name__ == "__main__":
    init_db()
    print("==================================================")
    print("Database SQLite berhasil diinisialisasi!")
    print("Lokasi file:", DB_PATH)
    print("Tabel yang aktif: 'users' & 'attendance_logs'")
    print("==================================================")
