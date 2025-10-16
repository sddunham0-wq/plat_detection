#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SETUP DATABASE SQLite - Sistem Parkir SMK

Penjelasan SMK: Script ini seperti "installer database"
Buat database file (.db) dan isi dengan tabel + data sample
"""

import sqlite3
import os
from datetime import datetime

DB_FILE = 'sistem_parkir_smk.db'

def setup_database():
    """
    Penjelasan SMK: Fungsi utama untuk setup database
    Seperti "format harddisk" lalu install sistem operasi
    """

    # Hapus database lama kalau ada (fresh install)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"🗑️  Database lama dihapus: {DB_FILE}")

    # Buat database baru
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    print(f"✅ Database baru dibuat: {DB_FILE}")

    # =====================================================
    # TABEL 1: KENDARAAN_TERDAFTAR
    # =====================================================
    print("\n📋 Membuat tabel kendaraan_terdaftar...")
    cursor.execute("""
        CREATE TABLE kendaraan_terdaftar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nomor_plat TEXT UNIQUE NOT NULL,
            nama_pemilik TEXT NOT NULL,
            jenis_kendaraan TEXT CHECK(jenis_kendaraan IN ('mobil', 'motor', 'truk')) DEFAULT 'mobil',
            status TEXT CHECK(status IN ('aktif', 'nonaktif')) DEFAULT 'aktif',
            nomor_hp TEXT,
            tanggal_daftar TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tanggal_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✅ Tabel kendaraan_terdaftar berhasil dibuat")

    # =====================================================
    # TABEL 2: LOG_AKSES_MASUK
    # =====================================================
    print("\n📋 Membuat tabel log_akses_masuk...")
    cursor.execute("""
        CREATE TABLE log_akses_masuk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plat_terdeteksi TEXT NOT NULL,
            tingkat_yakin REAL DEFAULT 0.0,
            status_akses TEXT CHECK(status_akses IN ('boleh_masuk', 'ditolak', 'manual_override')) NOT NULL,
            aksi_palang TEXT CHECK(aksi_palang IN ('opened', 'closed', 'manual')) NOT NULL,
            path_foto TEXT,
            waktu_deteksi TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            catatan TEXT
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX idx_plat ON log_akses_masuk(plat_terdeteksi)")
    cursor.execute("CREATE INDEX idx_waktu ON log_akses_masuk(waktu_deteksi)")
    cursor.execute("CREATE INDEX idx_status ON log_akses_masuk(status_akses)")
    print("✅ Tabel log_akses_masuk berhasil dibuat (dengan 3 index)")

    # =====================================================
    # DATA SAMPLE - Kendaraan Terdaftar
    # =====================================================
    print("\n📥 Memasukkan data sample kendaraan...")

    kendaraan_data = [
        # Kendaraan Guru dan Staff SMK (Jakarta - B series)
        ('B1234ABC', 'Pak Budi - Guru TKJ', 'mobil', 'aktif', '08123456789'),
        ('B5678XYZ', 'Bu Siti - Guru RPL', 'mobil', 'aktif', '08198765432'),
        ('B8888CCC', 'Pak Joko - Kepala Sekolah', 'mobil', 'aktif', '08111222333'),
        ('B9876EEE', 'Pak Doni - Wakasek', 'mobil', 'aktif', '08234567890'),
        ('B4444GGG', 'Pak Indra - Guru MM', 'mobil', 'aktif', '08345678901'),
        ('B2468ACE', 'Pak Rudi - Staff TU', 'truk', 'aktif', '08456789012'),
        ('B1357BEG', 'Pak Andi - Guru TBSM', 'mobil', 'aktif', '08567890123'),

        # Motor Guru dan Staff
        ('B3333MMM', 'Bu Maya - Guru Bahasa', 'motor', 'aktif', '08789012345'),
        ('B7777NNN', 'Bu Rina - Guru BK', 'motor', 'aktif', '08890123456'),

        # Siswa dari daerah lain (D series - Bandung)
        ('D9999AAA', 'Ahmad - Siswa TKJ 3', 'motor', 'aktif', '08712345678'),
        ('D3579BDF', 'Maya - Siswa RPL 2', 'motor', 'aktif', '08723456789'),
        ('D4680CFH', 'Rina - Siswa MM 1', 'motor', 'aktif', '08734567890'),
        ('D7531EHJ', 'Nita - Siswa TBSM 2', 'motor', 'aktif', '08745678901'),

        # Kendaraan dari Bogor (F series)
        ('F1111BBB', 'Bu Dewi - Guru Matematika', 'mobil', 'aktif', '08756789012'),
        ('F1234ABC', 'Bu Sri - Guru Agama', 'mobil', 'aktif', '08767890123'),
        ('F5555FFF', 'Bu Lina - Perpustakaan', 'mobil', 'aktif', '08778901234'),

        # Alumni yang sudah tidak aktif
        ('B0000XXX', 'Alumni Tidak Aktif', 'mobil', 'nonaktif', '08789012345'),
        ('D0000YYY', 'Siswa Pindah Sekolah', 'motor', 'nonaktif', '08790123456'),
    ]

    cursor.executemany("""
        INSERT INTO kendaraan_terdaftar
        (nomor_plat, nama_pemilik, jenis_kendaraan, status, nomor_hp)
        VALUES (?, ?, ?, ?, ?)
    """, kendaraan_data)

    print(f"✅ {len(kendaraan_data)} kendaraan berhasil dimasukkan")

    # =====================================================
    # DATA SAMPLE - Log Akses
    # =====================================================
    print("\n📥 Memasukkan data sample log akses...")

    log_data = [
        # Log masuk hari ini - boleh masuk
        ('B1234ABC', 0.95, 'boleh_masuk', 'opened', None, 'Pak Budi Guru TKJ - akses diberikan'),
        ('B5678XYZ', 0.88, 'boleh_masuk', 'opened', None, 'Bu Siti Guru RPL - akses diberikan'),
        ('D9999AAA', 0.92, 'boleh_masuk', 'opened', None, 'Ahmad siswa TKJ 3 - akses diberikan'),
        ('F1111BBB', 0.89, 'boleh_masuk', 'opened', None, 'Bu Dewi Guru Matematika - akses diberikan'),

        # Log ditolak (kendaraan tidak terdaftar)
        ('B9999ZZZ', 0.76, 'ditolak', 'closed', None, 'Plat tidak terdaftar di sistem sekolah'),
        ('D8888WWW', 0.82, 'ditolak', 'closed', None, 'Bukan kendaraan warga sekolah'),
        ('X0000XXX', 0.45, 'ditolak', 'closed', None, 'Plat tidak jelas - kamera kotor'),

        # Log manual satpam
        ('MANUAL', 1.0, 'manual_override', 'manual', None, 'Satpam buka manual - tamu dari Dinas Pendidikan'),
    ]

    cursor.executemany("""
        INSERT INTO log_akses_masuk
        (plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, path_foto, catatan)
        VALUES (?, ?, ?, ?, ?, ?)
    """, log_data)

    print(f"✅ {len(log_data)} log akses berhasil dimasukkan")

    # =====================================================
    # VERIFICATION
    # =====================================================
    print("\n" + "="*60)
    print("📊 VERIFIKASI DATABASE")
    print("="*60)

    # Count kendaraan
    cursor.execute("SELECT COUNT(*) FROM kendaraan_terdaftar WHERE status = 'aktif'")
    count_aktif = cursor.fetchone()[0]
    print(f"✅ Kendaraan Aktif: {count_aktif}")

    # Count by type
    cursor.execute("""
        SELECT jenis_kendaraan, COUNT(*)
        FROM kendaraan_terdaftar
        WHERE status = 'aktif'
        GROUP BY jenis_kendaraan
    """)
    for jenis, count in cursor.fetchall():
        print(f"   - {jenis.capitalize()}: {count}")

    # Count logs
    cursor.execute("SELECT COUNT(*) FROM log_akses_masuk")
    count_logs = cursor.fetchone()[0]
    print(f"✅ Total Log Akses: {count_logs}")

    cursor.execute("""
        SELECT status_akses, COUNT(*)
        FROM log_akses_masuk
        GROUP BY status_akses
    """)
    for status, count in cursor.fetchall():
        print(f"   - {status}: {count}")

    # Commit and close
    conn.commit()
    conn.close()

    print("\n" + "="*60)
    print("🎉 DATABASE SETUP BERHASIL!")
    print(f"📁 File database: {DB_FILE}")
    print(f"📊 Ukuran: {os.path.getsize(DB_FILE) / 1024:.2f} KB")
    print("="*60)

if __name__ == '__main__':
    print("🚀 SETUP DATABASE SQLite - Sistem Parkir SMK")
    print("="*60)
    setup_database()
