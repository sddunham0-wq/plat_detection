-- =====================================================
-- DATABASE SETUP UNTUK SISTEM AKSES KENDARAAN
-- =====================================================
-- Penjelasan SMK: Seperti bikin "pondasi rumah" digital
-- Tempat simpan semua data kendaraan dan log akses

-- 1. Buat database baru
DROP DATABASE IF EXISTS sistem_parkir_smk;
CREATE DATABASE sistem_parkir_smk;
USE sistem_parkir_smk;

-- =====================================================
-- TABEL 1: KENDARAAN_TERDAFTAR (Daftar Kendaraan yang Boleh Masuk)
-- =====================================================
-- Penjelasan SMK: Seperti "daftar member" yang boleh masuk ke sekolah
-- Mirip seperti kartu member di gym atau daftar tamu VIP

CREATE TABLE kendaraan_terdaftar (
    id_kendaraan INT AUTO_INCREMENT PRIMARY KEY,
    nomor_plat VARCHAR(20) UNIQUE NOT NULL,      -- Plat nomor (B1234ABC)
    nama_pemilik VARCHAR(100) NOT NULL,          -- Nama pemilik (Guru/Siswa/Staff)
    jenis_kendaraan ENUM('mobil', 'motor', 'truk') DEFAULT 'mobil',  -- Jenis kendaraan
    status ENUM('aktif', 'nonaktif') DEFAULT 'aktif',                -- Status aktif/tidak
    nomor_hp VARCHAR(15),                        -- Nomor HP pemilik
    tanggal_daftar TIMESTAMP DEFAULT CURRENT_TIMESTAMP,             -- Kapan didaftarkan
    tanggal_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- =====================================================
-- TABEL 2: LOG_AKSES_MASUK (Log Akses Masuk/Keluar Kendaraan)
-- =====================================================
-- Penjelasan SMK: Seperti "buku tamu digital" di sekolah
-- Catat siapa masuk kapan, boleh atau tidak, seperti satpam catat

CREATE TABLE log_akses_masuk (
    id_log INT AUTO_INCREMENT PRIMARY KEY,
    plat_terdeteksi VARCHAR(20) NOT NULL,        -- Plat yang terdeteksi kamera
    tingkat_yakin FLOAT DEFAULT 0.0,            -- Seberapa yakin sistem (0-1)
    status_akses ENUM('boleh_masuk', 'ditolak', 'manual_override', 'error') NOT NULL,  -- Status akses (match dengan app.py)
    aksi_palang ENUM('opened', 'closed', 'manual') NOT NULL,                -- Aksi palang (match dengan app.py)
    path_foto VARCHAR(255),                      -- Path foto kendaraan
    waktu_deteksi TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                      -- Kapan kejadian
    catatan TEXT,                                -- Catatan tambahan
    INDEX idx_plat (plat_terdeteksi),           -- Index untuk pencarian cepat
    INDEX idx_waktu (waktu_deteksi),            -- Index untuk filter tanggal
    INDEX idx_status (status_akses)             -- Index untuk filter status
);

-- =====================================================
-- DATA DUMMY UNTUK TESTING
-- =====================================================
-- Penjelasan SMK: Seperti "contoh data" untuk testing sistem
-- Data kendaraan yang sudah terdaftar (boleh masuk)

INSERT INTO kendaraan_terdaftar (nomor_plat, nama_pemilik, jenis_kendaraan, status, nomor_hp) VALUES
-- Kendaraan Guru dan Staff SMK (Jakarta - B series)
('B1234ABC', 'Pak Budi - Guru TKJ', 'mobil', 'aktif', '08123456789'),
('B5678XYZ', 'Bu Siti - Guru RPL', 'mobil', 'aktif', '08198765432'),
('B8888CCC', 'Pak Joko - Kepala Sekolah', 'mobil', 'aktif', '08111222333'),
('B9876EEE', 'Pak Doni - Wakasek', 'mobil', 'aktif', '08234567890'),
('B4444GGG', 'Pak Indra - Guru MM', 'mobil', 'aktif', '08345678901'),
('B2468ACE', 'Pak Rudi - Staff TU', 'truk', 'aktif', '08456789012'),
('B1357BEG', 'Pak Andi - Guru TBSM', 'mobil', 'aktif', '08567890123'),

-- Motor Guru dan Staff (Jakarta - B series)
('B3333MMM', 'Bu Maya - Guru Bahasa', 'motor', 'aktif', '08789012345'),
('B7777NNN', 'Bu Rina - Guru BK', 'motor', 'aktif', '08890123456'),

-- Siswa dari daerah lain (D series - Bandung)
('D9999AAA', 'Ahmad - Siswa TKJ 3', 'motor', 'aktif', '08712345678'),
('D3579BDF', 'Maya - Siswa RPL 2', 'motor', 'aktif', '08723456789'),
('D4680CFH', 'Rina - Siswa MM 1', 'motor', 'aktif', '08734567890'),
('D7531EHJ', 'Nita - Siswa TBSM 2', 'motor', 'aktif', '08745678901'),

-- Kendaraan dari Bogor (F series)
('F1111BBB', 'Bu Dewi - Guru Matematika', 'mobil', 'aktif', '08756789012'),
('F2222DDD', 'Bu Sri - Guru Agama', 'mobil', 'aktif', '08767890123'),
('F5555FFF', 'Bu Lina - Perpustakaan', 'mobil', 'aktif', '08778901234'),
('F 1818 HG', 'Pak Ahmad - Staff IT', 'motor', 'aktif', '08123456789'),

-- Alumni yang sudah tidak aktif (untuk testing)
('B0000XXX', 'Alumni Tidak Aktif', 'mobil', 'nonaktif', '08789012345'),
('D0000YYY', 'Siswa Pindah Sekolah', 'motor', 'nonaktif', '08790123456');

-- =====================================================
-- SAMPLE ACCESS LOGS UNTUK DEMO
-- =====================================================
-- Penjelasan SMK: Contoh data log untuk demo dashboard
-- Seperti "history" akses sebelumnya

INSERT INTO log_akses_masuk (plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, catatan) VALUES
-- Log masuk hari ini - boleh masuk
('B1234ABC', 0.95, 'boleh_masuk', 'opened', 'Pak Budi Guru TKJ - akses diberikan'),
('B5678XYZ', 0.88, 'boleh_masuk', 'opened', 'Bu Siti Guru RPL - akses diberikan'),
('D9999AAA', 0.92, 'boleh_masuk', 'opened', 'Ahmad siswa TKJ 3 - akses diberikan'),
('F1111BBB', 0.89, 'boleh_masuk', 'opened', 'Bu Dewi Guru Matematika - akses diberikan'),

-- Log ditolak (kendaraan tidak terdaftar)
('B9999ZZZ', 0.76, 'ditolak', 'closed', 'Plat tidak terdaftar di sistem sekolah'),
('D8888WWW', 0.82, 'ditolak', 'closed', 'Bukan kendaraan warga sekolah'),
('X0000XXX', 0.45, 'ditolak', 'closed', 'Plat tidak jelas - kamera kotor'),

-- Log manual satpam
('MANUAL01', 1.0, 'manual_override', 'manual', 'Satpam buka manual - tamu dari Dinas Pendidikan'),
('EMERGENCY', 1.0, 'manual_override', 'manual', 'Emergency - ambulans siswa sakit');

-- =====================================================
-- VIEWS UNTUK REPORTING (OPTIONAL)
-- =====================================================
-- Penjelasan SMK: "Shortcut" untuk query yang sering dipakai
-- Seperti "bookmark" di browser

-- View untuk statistik harian
CREATE VIEW statistik_akses_harian AS
SELECT
    DATE(waktu_deteksi) as tanggal_akses,
    COUNT(*) as total_akses,
    SUM(CASE WHEN status_akses = 'boleh_masuk' THEN 1 ELSE 0 END) as jumlah_masuk,
    SUM(CASE WHEN status_akses = 'ditolak' THEN 1 ELSE 0 END) as jumlah_ditolak,
    SUM(CASE WHEN status_akses = 'manual_override' THEN 1 ELSE 0 END) as manual_override
FROM log_akses_masuk
GROUP BY DATE(waktu_deteksi)
ORDER BY tanggal_akses DESC;

-- View untuk kendaraan paling aktif
CREATE VIEW kendaraan_paling_aktif AS
SELECT
    al.plat_terdeteksi,
    v.nama_pemilik,
    v.jenis_kendaraan,
    COUNT(*) as jumlah_akses,
    MAX(al.waktu_deteksi) as akses_terakhir
FROM log_akses_masuk al
LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
WHERE al.status_akses = 'boleh_masuk'
GROUP BY al.plat_terdeteksi, v.nama_pemilik, v.jenis_kendaraan
ORDER BY jumlah_akses DESC;

-- =====================================================
-- STORED PROCEDURES UNTUK OPERATIONS
-- =====================================================
-- Penjelasan SMK: "Function" yang sudah disimpan di database
-- Seperti "macro" di Excel untuk operasi berulang

DELIMITER //

-- Procedure untuk cek akses kendaraan (dipakai sistem)
CREATE PROCEDURE CekAksesKendaraan(
    IN p_nomor_plat VARCHAR(20),
    OUT p_status_akses VARCHAR(20),
    OUT p_nama_pemilik VARCHAR(100),
    OUT p_jenis_kendaraan VARCHAR(20)
)
BEGIN
    DECLARE v_count INT DEFAULT 0;

    -- Cek apakah kendaraan terdaftar dan aktif
    SELECT COUNT(*), nama_pemilik, jenis_kendaraan
    INTO v_count, p_nama_pemilik, p_jenis_kendaraan
    FROM kendaraan_terdaftar
    WHERE nomor_plat = p_nomor_plat AND status = 'aktif'
    GROUP BY nama_pemilik, jenis_kendaraan;

    -- Tentukan status
    IF v_count > 0 THEN
        SET p_status_akses = 'boleh_masuk';
    ELSE
        SET p_status_akses = 'ditolak';
        SET p_nama_pemilik = 'Tidak Dikenal';
        SET p_jenis_kendaraan = 'Tidak Dikenal';
    END IF;
END //

-- Procedure untuk catat log akses
CREATE PROCEDURE CatatLogAkses(
    IN p_plat_terdeteksi VARCHAR(20),
    IN p_tingkat_yakin FLOAT,
    IN p_status_akses VARCHAR(20),
    IN p_aksi_palang VARCHAR(20),
    IN p_path_foto VARCHAR(255),
    IN p_catatan TEXT
)
BEGIN
    INSERT INTO log_akses_masuk
    (plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, path_foto, catatan)
    VALUES
    (p_plat_terdeteksi, p_tingkat_yakin, p_status_akses, p_aksi_palang, p_path_foto, p_catatan);
END //

DELIMITER ;

-- =====================================================
-- PERMISSIONS & SECURITY (PRODUCTION READY)
-- =====================================================
-- Penjelasan SMK: Setup keamanan database
-- Seperti "password" dan "permission" di Windows

-- Buat user khusus untuk aplikasi (tidak pakai root)
-- CREATE USER 'aplikasi_parkir_smk'@'localhost' IDENTIFIED BY 'password_aman_123';
-- GRANT SELECT, INSERT, UPDATE ON sistem_parkir_smk.* TO 'aplikasi_parkir_smk'@'localhost';
-- FLUSH PRIVILEGES;

-- =====================================================
-- VERIFICATION QUERIES
-- =====================================================
-- Penjelasan SMK: Query untuk cek apakah setup berhasil
-- Seperti "test" untuk memastikan semua berjalan

SELECT '=== DATABASE SETUP VERIFICATION ===' as status;

SELECT 'Tabel Kendaraan Terdaftar' as nama_tabel, COUNT(*) as jumlah_record FROM kendaraan_terdaftar;
SELECT 'Tabel Log Akses Masuk' as nama_tabel, COUNT(*) as jumlah_record FROM log_akses_masuk;

SELECT 'Kendaraan Aktif per Jenis' as info, jenis_kendaraan, COUNT(*) as jumlah
FROM kendaraan_terdaftar WHERE status = 'aktif' GROUP BY jenis_kendaraan;

SELECT 'Log Akses Terbaru' as info, status_akses, COUNT(*) as jumlah
FROM log_akses_masuk GROUP BY status_akses;

SELECT '=== SETUP COMPLETED SUCCESSFULLY ===' as status;