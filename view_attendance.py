# ==============================================================================
# FILE: view_attendance.py
# FUNGSI: Skrip Penampil Rekapitulasi Kehadiran & Ekspor Data ke Format CSV (Excel)
# PENJELASAN:
# File ini berfungsi sebagai laporan sederhana bagi administrator:
# 1. Menampilkan tabel daftar siapa saja yang sudah absen hari ini langsung di terminal.
# 2. Menyediakan fitur ekspor seluruh riwayat absensi ke file .csv yang bisa langsung
#    dibuka di Microsoft Excel atau Google Sheets.
# ==============================================================================

import csv                         # Library standar Python untuk membaca dan menulis file CSV (Comma Separated Values)
from datetime import datetime, date# Library manipulasi waktu dan tanggal
from pathlib import Path           # Library pengelolaan path file
from database.db import get_today_attendance_summary, get_connection, init_db # Modul database

def show_today_attendance():
    """
    LOGIKA PENAMPILAN REKAP HARI INI:
    Mengambil data absensi hari ini dari SQLite dan mencetaknya ke layar terminal
    dalam bentuk tabel bergaris yang rapi.
    
    Teknik formatting string:
    - {variabel:<lebar_kolom} : Rata kiri (Left-align) dengan batas lebar tertentu.
    """
    init_db()                      # Pastikan tabel database ada
    logs = get_today_attendance_summary()
    today_str = date.today().strftime("%Y-%m-%d") # Ambil tanggal hari ini (contoh: "2026-09-03")
    
    print(f"\n{'='*75}")
    print(f"             REKAP ABSENSI HARI INI ({today_str})             ")
    print(f"{'='*75}")
    
    # Jika belum ada seorang pun yang absen hari ini
    if not logs:
        print("  Belum ada catatan absensi untuk hari ini.")
        print(f"{'='*75}\n")
        return
        
    # Header Tabel
    print(f"{'ID':<4} | {'KODE':<10} | {'NAMA':<20} | {'DIVISI':<12} | {'STATUS':<6} | {'WAKTU':<19}")
    print(f"{'-'*75}")
    
    # Cetak setiap baris data kehadiran
    for log in logs:
        status_str = "MASUK" if log["status"] == "IN" else "PULANG"
        print(f"{log['id']:<4} | {log['user_code']:<10} | {log['name'][:18]:<20} | {log['department'][:10]:<12} | {status_str:<6} | {log['timestamp']}")
        
    print(f"{'='*75}\n")

def export_to_csv(filename: str = ""):
    """
    LOGIKA EKSPOR KE CSV:
    Mengambil SELURUH riwayat absensi (dari hari-hari sebelumnya juga) lalu
    menuliskannya ke dalam file CSV agar bisa dibuka di Excel.
    """
    init_db()
    # Jika nama file tidak ditentukan pengguna, buat nama otomatis berdasarkan tanggal hari ini
    if not filename:
        today_str = date.today().strftime("%Y%m%d")
        filename = f"rekap_absensi_{today_str}.csv"
        
    # Ambil data lengkap menggunakan SQL JOIN antara tabel attendance_logs dan users
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id, u.user_code, u.name, u.department, 
                   a.date, a.timestamp, a.status, a.confidence, a.snapshot_path
            FROM attendance_logs a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.id DESC
        """)
        rows = cursor.fetchall()
        
    # Validasi jika database masih kosong
    if not rows:
        print("[INFO] Tidak ada data absensi di database untuk diekspor.")
        return

    # Daftar nama kolom (Header) file CSV
    fieldnames = [
        "ID", 
        "Kode Pengguna", 
        "Nama", 
        "Divisi", 
        "Tanggal", 
        "Waktu", 
        "Status", 
        "Akurasi (%)", 
        "File Foto Snapshot"
    ]
    
    # Buka file baru untuk ditulisi (mode="w") dengan encoding UTF-8 agar nama Indonesia/karakter khusus tidak rusak
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames) # Tulis baris header judul kolom
        
        # Tulis baris demi baris data
        for r in rows:
            writer.writerow([
                r["id"],
                r["user_code"],
                r["name"],
                r["department"],
                r["date"],
                r["timestamp"],
                "MASUK" if r["status"] == "IN" else "PULANG",
                f"{r['confidence']*100:.1f}", # Format akurasi jadi persentase (contoh: 88.5)
                r["snapshot_path"]
            ])
            
    print(f"[SUKSES] Data absensi ({len(rows)} baris) berhasil diekspor ke:")
    print(f" -> {Path(filename).resolve()}")

# ------------------------------------------------------------------------------
# PROGRAM ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Tampilkan rekap ke terminal terlebih dahulu
    show_today_attendance()
    
    # Tanyakan kepada pengguna apakah ingin mengekspor ke file Excel/CSV
    ans = input("Ekspor riwayat ke file CSV? (y/n): ").strip().lower()
    if ans == "y":
        export_to_csv()
