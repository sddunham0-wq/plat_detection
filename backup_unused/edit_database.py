#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDIT DATABASE - Script Mudah untuk Edit Database Plat

Penjelasan SMK: Script ini untuk tambah/edit/hapus kendaraan
tanpa perlu buka web browser
"""

import sqlite3
from datetime import datetime

DB_FILE = 'sistem_parkir_smk.db'

def show_menu():
    """Tampilkan menu pilihan"""
    print("\n" + "="*60)
    print("📋 MENU EDIT DATABASE PLAT NOMOR")
    print("="*60)
    print("1. Lihat semua kendaraan")
    print("2. Tambah kendaraan baru")
    print("3. Edit kendaraan")
    print("4. Hapus kendaraan")
    print("5. Cari kendaraan")
    print("0. Keluar")
    print("="*60)

def lihat_semua():
    """Tampilkan semua kendaraan"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM kendaraan_terdaftar ORDER BY id")
    vehicles = cursor.fetchall()

    print("\n" + "="*100)
    print(f"{'ID':<5} {'Plat':<12} {'Pemilik':<30} {'Jenis':<10} {'Status':<10} {'HP':<15}")
    print("="*100)

    for v in vehicles:
        print(f"{v['id']:<5} {v['nomor_plat']:<12} {v['nama_pemilik']:<30} {v['jenis_kendaraan']:<10} {v['status']:<10} {v['nomor_hp'] or '-':<15}")

    print("="*100)
    print(f"Total: {len(vehicles)} kendaraan")

    conn.close()

def tambah_kendaraan():
    """Tambah kendaraan baru"""
    print("\n📝 TAMBAH KENDARAAN BARU")
    print("-" * 60)

    nomor_plat = input("Nomor Plat (contoh: B1234ABC): ").strip().upper().replace(' ', '')
    nama_pemilik = input("Nama Pemilik: ").strip()

    print("\nJenis Kendaraan:")
    print("1. Mobil")
    print("2. Motor")
    print("3. Truk")
    jenis = input("Pilih (1/2/3): ").strip()

    jenis_map = {'1': 'mobil', '2': 'motor', '3': 'truk'}
    jenis_kendaraan = jenis_map.get(jenis, 'mobil')

    nomor_hp = input("Nomor HP (optional): ").strip()

    # Insert ke database
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO kendaraan_terdaftar (nomor_plat, nama_pemilik, jenis_kendaraan, nomor_hp, status)
            VALUES (?, ?, ?, ?, 'aktif')
        """, (nomor_plat, nama_pemilik, jenis_kendaraan, nomor_hp))

        conn.commit()
        conn.close()

        print(f"\n✅ Kendaraan {nomor_plat} berhasil ditambahkan!")

    except sqlite3.IntegrityError:
        print(f"\n❌ Error: Plat {nomor_plat} sudah terdaftar!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def edit_kendaraan():
    """Edit data kendaraan"""
    print("\n✏️ EDIT KENDARAAN")
    print("-" * 60)

    # Tampilkan daftar
    lihat_semua()

    vehicle_id = input("\nMasukkan ID kendaraan yang mau diedit: ").strip()

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Cek apakah ID ada
        cursor.execute("SELECT * FROM kendaraan_terdaftar WHERE id = ?", (vehicle_id,))
        vehicle = cursor.fetchone()

        if not vehicle:
            print(f"❌ Kendaraan dengan ID {vehicle_id} tidak ditemukan!")
            conn.close()
            return

        # Tampilkan data lama
        print(f"\nData Lama:")
        print(f"Plat: {vehicle['nomor_plat']}")
        print(f"Pemilik: {vehicle['nama_pemilik']}")
        print(f"Jenis: {vehicle['jenis_kendaraan']}")
        print(f"Status: {vehicle['status']}")
        print(f"HP: {vehicle['nomor_hp'] or '-'}")

        # Input data baru (tekan Enter untuk skip)
        print("\nData Baru (tekan Enter untuk tidak ubah):")
        nama_pemilik = input(f"Nama Pemilik [{vehicle['nama_pemilik']}]: ").strip() or vehicle['nama_pemilik']

        print("\nJenis: 1=Mobil, 2=Motor, 3=Truk")
        jenis = input(f"Jenis [{vehicle['jenis_kendaraan']}]: ").strip()
        jenis_map = {'1': 'mobil', '2': 'motor', '3': 'truk'}
        jenis_kendaraan = jenis_map.get(jenis, vehicle['jenis_kendaraan'])

        print("\nStatus: 1=Aktif, 2=Nonaktif")
        status = input(f"Status [{vehicle['status']}]: ").strip()
        status_map = {'1': 'aktif', '2': 'nonaktif'}
        status_value = status_map.get(status, vehicle['status'])

        nomor_hp = input(f"Nomor HP [{vehicle['nomor_hp'] or '-'}]: ").strip() or vehicle['nomor_hp']

        # Update database
        cursor.execute("""
            UPDATE kendaraan_terdaftar
            SET nama_pemilik = ?, jenis_kendaraan = ?, status = ?, nomor_hp = ?
            WHERE id = ?
        """, (nama_pemilik, jenis_kendaraan, status_value, nomor_hp, vehicle_id))

        conn.commit()
        conn.close()

        print(f"\n✅ Data kendaraan ID {vehicle_id} berhasil diupdate!")

    except Exception as e:
        print(f"\n❌ Error: {e}")

def hapus_kendaraan():
    """Hapus kendaraan"""
    print("\n🗑️ HAPUS KENDARAAN")
    print("-" * 60)

    # Tampilkan daftar
    lihat_semua()

    vehicle_id = input("\nMasukkan ID kendaraan yang mau dihapus: ").strip()

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Cek apakah ID ada
        cursor.execute("SELECT * FROM kendaraan_terdaftar WHERE id = ?", (vehicle_id,))
        vehicle = cursor.fetchone()

        if not vehicle:
            print(f"❌ Kendaraan dengan ID {vehicle_id} tidak ditemukan!")
            conn.close()
            return

        # Konfirmasi
        print(f"\nAnda yakin mau hapus kendaraan ini?")
        print(f"Plat: {vehicle['nomor_plat']}")
        print(f"Pemilik: {vehicle['nama_pemilik']}")

        confirm = input("\nKetik 'YA' untuk hapus: ").strip().upper()

        if confirm == 'YA':
            cursor.execute("DELETE FROM kendaraan_terdaftar WHERE id = ?", (vehicle_id,))
            conn.commit()
            print(f"\n✅ Kendaraan {vehicle['nomor_plat']} berhasil dihapus!")
        else:
            print("\n❌ Batal hapus.")

        conn.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")

def cari_kendaraan():
    """Cari kendaraan berdasarkan plat atau nama"""
    print("\n🔍 CARI KENDARAAN")
    print("-" * 60)

    keyword = input("Masukkan plat nomor atau nama pemilik: ").strip().upper()

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM kendaraan_terdaftar
            WHERE UPPER(nomor_plat) LIKE ? OR UPPER(nama_pemilik) LIKE ?
        """, (f'%{keyword}%', f'%{keyword}%'))

        vehicles = cursor.fetchall()

        if vehicles:
            print("\n" + "="*100)
            print(f"{'ID':<5} {'Plat':<12} {'Pemilik':<30} {'Jenis':<10} {'Status':<10} {'HP':<15}")
            print("="*100)

            for v in vehicles:
                print(f"{v['id']:<5} {v['nomor_plat']:<12} {v['nama_pemilik']:<30} {v['jenis_kendaraan']:<10} {v['status']:<10} {v['nomor_hp'] or '-':<15}")

            print("="*100)
            print(f"Ditemukan: {len(vehicles)} kendaraan")
        else:
            print(f"\n❌ Tidak ditemukan kendaraan dengan keyword '{keyword}'")

        conn.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    """Main program"""
    print("\n🚗 SISTEM EDIT DATABASE PLAT NOMOR")
    print("File database: " + DB_FILE)

    while True:
        show_menu()
        choice = input("\nPilih menu (0-5): ").strip()

        if choice == '1':
            lihat_semua()
        elif choice == '2':
            tambah_kendaraan()
        elif choice == '3':
            edit_kendaraan()
        elif choice == '4':
            hapus_kendaraan()
        elif choice == '5':
            cari_kendaraan()
        elif choice == '0':
            print("\n👋 Terima kasih! Sampai jumpa.")
            break
        else:
            print("\n❌ Pilihan tidak valid!")

        input("\nTekan Enter untuk lanjut...")

if __name__ == '__main__':
    main()
