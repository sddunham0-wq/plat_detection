-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 25 Okt 2025 pada 05.16
-- Versi server: 9.4.0
-- Versi PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `plat_detection`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `access_log`
--

CREATE TABLE `access_log` (
  `id` int NOT NULL,
  `vehicle_id` int DEFAULT NULL,
  `plate_number` varchar(10) DEFAULT NULL,
  `acces_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `status` varchar(10) DEFAULT NULL,
  `image_url` varchar(255) DEFAULT NULL,
  `manual_override` tinyint(1) DEFAULT '0',
  `override_reason` varchar(255) DEFAULT NULL,
  `reviewed_by` varchar(100) DEFAULT NULL,
  `ocr_confidence` decimal(5,2) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data untuk tabel `access_log`
--

INSERT INTO `access_log` (`id`, `vehicle_id`, `plate_number`, `acces_time`, `status`, `image_url`, `manual_override`, `override_reason`, `reviewed_by`, `ocr_confidence`) VALUES
(1, 1, 'F1818HG', '2025-10-16 09:36:14', 'masuk', 'test_image.jpg', 0, NULL, NULL, NULL),
(2, NULL, 'TEST9999', '2025-10-16 09:36:14', 'ditolak', 'test_denied.jpg', 0, NULL, NULL, NULL),
(3, 1, 'F1818HG', '2025-10-16 09:36:41', 'masuk', 'test_image.jpg', 0, NULL, NULL, NULL),
(4, NULL, 'TEST9999', '2025-10-16 09:36:41', 'ditolak', 'test_denied.jpg', 0, NULL, NULL, NULL),
(5, 1, 'F1818HG', '2025-10-16 09:42:24', 'masuk', 'test_image.jpg', 0, NULL, NULL, NULL),
(6, NULL, 'TEST9999', '2025-10-16 09:42:24', 'ditolak', 'test_denied.jpg', 0, NULL, NULL, NULL),
(7, 1, 'F1818HG', '2025-10-16 09:42:33', 'masuk', 'test_image.jpg', 0, NULL, NULL, NULL),
(8, NULL, 'TEST9999', '2025-10-16 09:42:33', 'ditolak', 'test_denied.jpg', 0, NULL, NULL, NULL),
(32, NULL, '', '2025-10-20 04:08:57', 'ditolak', NULL, 0, NULL, NULL, NULL),
(224, NULL, 'B1203EZU', '2025-10-20 08:26:05', 'ditolak', NULL, 0, NULL, NULL, NULL),
(340, 6, 'B1263EZU', '2025-10-20 08:55:26', 'masuk', NULL, 0, NULL, NULL, NULL),
(830, 6, 'B1263EZU', '2025-10-24 07:10:44', 'masuk', 'detected_vehicles/B1263EZU_20251024_141044.jpg', 0, NULL, NULL, NULL),
(849, NULL, '', '2025-10-25 01:56:49', 'ditolak', 'detected_vehicles/_20251025_085649.jpg', 0, NULL, NULL, NULL),
(850, NULL, '', '2025-10-25 01:57:26', 'ditolak', 'detected_vehicles/_20251025_085726.jpg', 0, NULL, NULL, NULL);

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `access_log`
--
ALTER TABLE `access_log`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_accesslog_vehicle` (`vehicle_id`),
  ADD KEY `idx_plate_number` (`plate_number`),
  ADD KEY `idx_acces_time` (`acces_time`),
  ADD KEY `idx_status` (`status`),
  ADD KEY `idx_manual_override` (`manual_override`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `access_log`
--
ALTER TABLE `access_log`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=851;

--
-- Ketidakleluasaan untuk tabel pelimpahan (Dumped Tables)
--

--
-- Ketidakleluasaan untuk tabel `access_log`
--
ALTER TABLE `access_log`
  ADD CONSTRAINT `fk_accesslog_vehicle` FOREIGN KEY (`vehicle_id`) REFERENCES `vehicles` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
