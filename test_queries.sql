-- =====================================================
-- TEST QUERIES - Plat F 1818 HG (Motor Bogor)
-- =====================================================
-- Test INSERT dan SELECT untuk verifikasi database
-- Plat dari image2.png: F 1818 HG

USE sistem_parkir_smk;

-- =====================================================
-- 1. INSERT: Tambah Plat F 1818 HG
-- =====================================================

INSERT INTO kendaraan_terdaftar
(nomor_plat, nama_pemilik, jenis_kendaraan, status, nomor_hp)
VALUES
('F 1818 HG', 'Pak Ahmad - Staff IT', 'motor', 'aktif', '08123456789');

-- =====================================================
-- 2. SELECT: Verifikasi Data Berhasil Masuk
-- =====================================================

SELECT
    id_kendaraan,
    nomor_plat,
    nama_pemilik,
    jenis_kendaraan,
    status,
    nomor_hp,
    tanggal_daftar
FROM kendaraan_terdaftar
WHERE nomor_plat = 'F 1818 HG';

-- =====================================================
-- 3. SELECT: Semua Kendaraan dari Bogor (F Series)
-- =====================================================

SELECT
    nomor_plat,
    nama_pemilik,
    jenis_kendaraan,
    status
FROM kendaraan_terdaftar
WHERE nomor_plat LIKE 'F%'
ORDER BY nomor_plat;

-- =====================================================
-- 4. SELECT: Count Total per Jenis Kendaraan
-- =====================================================

SELECT
    jenis_kendaraan,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'aktif' THEN 1 ELSE 0 END) as aktif,
    SUM(CASE WHEN status = 'nonaktif' THEN 1 ELSE 0 END) as nonaktif
FROM kendaraan_terdaftar
GROUP BY jenis_kendaraan
ORDER BY jenis_kendaraan;

-- =====================================================
-- 5. INSERT: Test Log Akses untuk F 1818 HG
-- =====================================================

INSERT INTO log_akses_masuk
(plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, catatan)
VALUES
('F 1818 HG', 0.92, 'boleh_masuk', 'opened', 'Test deteksi motor Bogor - dari image2.png');

-- =====================================================
-- 6. SELECT: Verifikasi Log Akses
-- =====================================================

SELECT
    id_log,
    plat_terdeteksi,
    tingkat_yakin,
    status_akses,
    aksi_palang,
    waktu_deteksi,
    catatan
FROM log_akses_masuk
WHERE plat_terdeteksi = 'F 1818 HG'
ORDER BY waktu_deteksi DESC;

-- =====================================================
-- 7. SELECT: Join - Info Lengkap Akses dengan Pemilik
-- =====================================================

SELECT
    l.id_log,
    l.plat_terdeteksi,
    k.nama_pemilik,
    k.jenis_kendaraan,
    l.tingkat_yakin,
    l.status_akses,
    l.waktu_deteksi
FROM log_akses_masuk l
LEFT JOIN kendaraan_terdaftar k ON l.plat_terdeteksi = k.nomor_plat
WHERE l.plat_terdeteksi = 'F 1818 HG'
ORDER BY l.waktu_deteksi DESC;

-- =====================================================
-- 8. SELECT: Summary Statistics
-- =====================================================

-- Total kendaraan terdaftar
SELECT 'Total Kendaraan Terdaftar' as info, COUNT(*) as jumlah
FROM kendaraan_terdaftar
WHERE status = 'aktif';

-- Total motor dari Bogor (F series)
SELECT 'Motor Bogor (F Series)' as info, COUNT(*) as jumlah
FROM kendaraan_terdaftar
WHERE nomor_plat LIKE 'F%' AND jenis_kendaraan = 'motor';

-- Total log akses F 1818 HG
SELECT 'Log Akses F 1818 HG' as info, COUNT(*) as jumlah
FROM log_akses_masuk
WHERE plat_terdeteksi = 'F 1818 HG';

-- =====================================================
-- 9. SELECT: Recent Access Logs (Last 10)
-- =====================================================

SELECT
    l.plat_terdeteksi,
    k.nama_pemilik,
    l.status_akses,
    l.waktu_deteksi
FROM log_akses_masuk l
LEFT JOIN kendaraan_terdaftar k ON l.plat_terdeteksi = k.nomor_plat
ORDER BY l.waktu_deteksi DESC
LIMIT 10;

-- =====================================================
-- 10. DELETE: Cleanup Test Data (Optional)
-- =====================================================
-- Uncomment untuk hapus test data

-- DELETE FROM log_akses_masuk WHERE plat_terdeteksi = 'F 1818 HG' AND catatan LIKE '%Test%';
-- DELETE FROM kendaraan_terdaftar WHERE nomor_plat = 'F 1818 HG' AND nama_pemilik = 'Pak Ahmad - Staff IT';

-- =====================================================
-- VERIFICATION CHECKLIST
-- =====================================================

SELECT '=== VERIFICATION CHECKLIST ===' as status;

-- Check F 1818 HG exists in kendaraan_terdaftar
SELECT
    CASE
        WHEN COUNT(*) > 0 THEN '✅ F 1818 HG terdaftar'
        ELSE '❌ F 1818 HG belum terdaftar'
    END as status
FROM kendaraan_terdaftar
WHERE nomor_plat = 'F 1818 HG';

-- Check F 1818 HG has logs
SELECT
    CASE
        WHEN COUNT(*) > 0 THEN CONCAT('✅ F 1818 HG punya ', COUNT(*), ' log akses')
        ELSE '❌ F 1818 HG belum ada log'
    END as status
FROM log_akses_masuk
WHERE plat_terdeteksi = 'F 1818 HG';

-- Check path_foto column exists
SELECT
    CASE
        WHEN COUNT(*) > 0 THEN '✅ Kolom path_foto ada'
        ELSE '❌ Kolom path_foto tidak ada'
    END as status
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'sistem_parkir_smk'
AND TABLE_NAME = 'log_akses_masuk'
AND COLUMN_NAME = 'path_foto';

SELECT '=== TEST COMPLETED ===' as status;
