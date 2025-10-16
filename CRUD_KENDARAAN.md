# ✅ CRUD KENDARAAN - Sistem Manajemen Kendaraan Terdaftar

**Tanggal**: 2025-10-16
**Status**: ✅ **READY TO USE**

---

## 🎯 **Fitur CRUD Lengkap**

CRUD (Create, Read, Update, Delete) untuk mengelola kendaraan yang boleh masuk area parkir.

### **Fitur yang Tersedia**:

1. ✅ **Create** - Tambah kendaraan baru
2. ✅ **Read** - Lihat daftar semua kendaraan
3. ✅ **Update** - Edit data kendaraan
4. ✅ **Delete** - Hapus kendaraan
5. ✅ **Search** - Cari berdasarkan plat/nama
6. ✅ **Statistics** - Ringkasan data kendaraan

---

## 📊 **Struktur Database**

### **Tabel: kendaraan_terdaftar**

```sql
CREATE TABLE kendaraan_terdaftar (
    id_kendaraan INT AUTO_INCREMENT PRIMARY KEY,
    nomor_plat VARCHAR(20) UNIQUE NOT NULL,
    nama_pemilik VARCHAR(100) NOT NULL,
    jenis_kendaraan ENUM('mobil', 'motor', 'truk') DEFAULT 'mobil',
    status ENUM('aktif', 'nonaktif') DEFAULT 'aktif',
    nomor_hp VARCHAR(15),
    tanggal_daftar TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tanggal_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### **Field Explanation**:

| Field | Tipe | Keterangan |
|-------|------|------------|
| `id_kendaraan` | INT | Primary key, auto increment |
| `nomor_plat` | VARCHAR(20) | Plat nomor (unique), contoh: B1234ABC |
| `nama_pemilik` | VARCHAR(100) | Nama pemilik kendaraan |
| `jenis_kendaraan` | ENUM | mobil / motor / truk |
| `status` | ENUM | aktif / nonaktif |
| `nomor_hp` | VARCHAR(15) | Nomor HP pemilik (optional) |
| `tanggal_daftar` | TIMESTAMP | Tanggal registrasi |
| `tanggal_update` | TIMESTAMP | Last update timestamp |

---

## 🚀 **Cara Menggunakan**

### **1. Akses Halaman Kendaraan**

```
http://localhost:5001/vehicles
```

Atau klik menu **"Kendaraan"** di navigation bar.

---

### **2. CREATE - Tambah Kendaraan Baru**

**Langkah**:
1. Klik tombol **"Tambah Kendaraan"** (hijau, kanan atas)
2. Modal form akan terbuka
3. Isi data:
   - **Nomor Plat** (required): Contoh `B1234ABC` (tanpa spasi)
   - **Nama Pemilik** (required): Contoh `Pak Budi - Guru TKJ`
   - **Jenis Kendaraan** (required): Pilih Mobil/Motor/Truk
   - **Nomor HP** (optional): Contoh `08123456789`
4. Klik **"Simpan"**

**Backend Route**: `POST /add_vehicle`

**Validasi**:
- ✅ Nomor plat di-uppercase otomatis
- ✅ Spasi di plat dihapus otomatis
- ✅ Nomor plat harus unique (tidak boleh duplikat)
- ✅ Flash message untuk sukses/error

**Response**:
```
✅ Kendaraan B1234ABC berhasil ditambahkan!
❌ Plat nomor B1234ABC sudah terdaftar!
```

---

### **3. READ - Lihat Daftar Kendaraan**

**Fitur**:

**a) List View**:
- Tampilan card untuk setiap kendaraan
- Menampilkan:
  - 🚗 Icon kendaraan (mobil/motor/truk)
  - Nomor plat (font khusus, monospace)
  - Nama pemilik
  - Badge jenis kendaraan
  - Badge status (Aktif/Nonaktif)
  - Nomor HP
  - Tanggal registrasi
  - Tombol Edit & Hapus

**b) Statistics Dashboard**:
- Total kendaraan
- Total kendaraan aktif
- Total mobil
- Total motor

**c) Search Functionality**:
- Search box untuk filter real-time
- Cari berdasarkan nomor plat atau nama pemilik
- Auto-filter tanpa reload page

**Backend Route**: `GET /vehicles`

**Query**:
```sql
SELECT * FROM kendaraan_terdaftar
ORDER BY tanggal_daftar DESC
```

---

### **4. UPDATE - Edit Data Kendaraan**

**Langkah**:
1. Klik tombol **"Edit"** (kuning) pada kendaraan yang ingin diubah
2. Modal form akan terbuka dengan data terisi
3. Edit data yang ingin diubah:
   - Nama Pemilik (bisa diubah)
   - Jenis Kendaraan (bisa diubah)
   - Nomor HP (bisa diubah)
   - Status (bisa diubah: aktif/nonaktif)
   - **Nomor Plat** (readonly, tidak bisa diubah)
4. Klik **"Update"**

**Backend Route**: `POST /edit_vehicle/<int:vehicle_id>`

**Query**:
```sql
UPDATE kendaraan_terdaftar
SET nama_pemilik = %s, jenis_kendaraan = %s,
    nomor_hp = %s, status = %s
WHERE id_kendaraan = %s
```

**Response**:
```
✅ Data kendaraan berhasil diupdate!
```

**Note**: Nomor plat tidak bisa diubah untuk menjaga integritas data dengan log akses.

---

### **5. DELETE - Hapus Kendaraan**

**Langkah**:
1. Klik tombol **"Hapus"** (merah) pada kendaraan
2. Modal konfirmasi akan muncul
3. Review data kendaraan yang akan dihapus:
   - Plat nomor
   - Nama pemilik
4. Klik **"Ya, Hapus"** untuk konfirmasi

**Backend Route**: `GET /delete_vehicle/<int:vehicle_id>`

**Query**:
```sql
-- Get vehicle info dulu
SELECT nomor_plat, nama_pemilik
FROM kendaraan_terdaftar
WHERE id_kendaraan = %s

-- Delete
DELETE FROM kendaraan_terdaftar
WHERE id_kendaraan = %s
```

**Response**:
```
✅ Kendaraan B1234ABC (Pak Budi) berhasil dihapus!
❌ Kendaraan tidak ditemukan!
```

**Warning**:
- ⚠️ Tindakan ini tidak dapat dibatalkan!
- ⚠️ Log akses historis tidak terhapus (foreign key: nomor_plat)

---

## 🎨 **UI Features**

### **Design System**:

**Color Coding by Vehicle Type**:
- 🏍️ **Motor**: Green border (#28a745)
- 🚗 **Mobil**: Blue border (#007bff)
- 🚚 **Truk**: Yellow border (#ffc107)

**Status Badges**:
- ✅ **Aktif**: Green gradient background
- ❌ **Nonaktif**: Red gradient background

**Interactive Elements**:
- ✨ Hover effect pada card (translateY + shadow)
- 🎯 Auto-dismiss alerts setelah 5 detik
- 🔍 Real-time search filtering
- 📱 Responsive design (mobile-friendly)

**Typography**:
- Nomor plat: `Courier New` (monospace, letter-spacing 3px)
- Heading: `Segoe UI` (clean, modern)

---

## 📋 **File Structure**

```
project/
├── app.py                      # Backend routes CRUD
├── templates/
│   └── vehicles.html          # Frontend UI CRUD ✅ NEW
├── database_setup.sql         # Database schema
└── CRUD_KENDARAAN.md         # Documentation (this file)
```

---

## 🧪 **Testing**

### **Manual Testing Checklist**:

**1. CREATE (Tambah)**:
- ✅ Tambah kendaraan valid → Success
- ✅ Tambah plat duplikat → Error message
- ✅ Field required kosong → Validation error
- ✅ Nomor plat auto uppercase

**2. READ (Lihat)**:
- ✅ Load semua kendaraan
- ✅ Statistics muncul dengan benar
- ✅ Card styling sesuai jenis kendaraan
- ✅ Status badge warna benar

**3. UPDATE (Edit)**:
- ✅ Edit nama pemilik → Success
- ✅ Ubah status aktif/nonaktif → Success
- ✅ Nomor plat readonly (tidak bisa diubah)
- ✅ Update timestamp otomatis

**4. DELETE (Hapus)**:
- ✅ Hapus dengan konfirmasi → Success
- ✅ Modal konfirmasi muncul
- ✅ Cancel tetap di halaman

**5. SEARCH**:
- ✅ Cari by plat nomor → Filter bekerja
- ✅ Cari by nama pemilik → Filter bekerja
- ✅ Case insensitive search

---

## 🔒 **Security Features**

### **Input Validation**:

**1. Server-Side Validation**:
```python
# Normalisasi nomor plat
nomor_plat = request.form['nomor_plat'].replace(' ', '').upper()

# Required fields check (HTML + Backend)
if not nama_pemilik or not jenis_kendaraan:
    return error
```

**2. SQL Injection Prevention**:
```python
# Parameterized queries (NOT string concatenation)
cursor.execute(query, (nomor_plat, nama_pemilik, ...))  # ✅ SAFE
# cursor.execute(f"INSERT ... VALUES ('{nomor_plat}')")  # ❌ DANGEROUS
```

**3. Unique Constraint**:
```sql
nomor_plat VARCHAR(20) UNIQUE NOT NULL
```

**4. CSRF Protection**:
- Flask secret key configured
- Form POST dengan method validation

---

## 📊 **Database Statistics**

Current data (from database_setup.sql):

```sql
-- Total kendaraan: 19
-- Kendaraan aktif: 17
-- Kendaraan nonaktif: 2

-- By type:
-- Mobil: 10
-- Motor: 9
-- Truk: 0

-- By region:
-- Jakarta (B): 13
-- Bandung (D): 4
-- Bogor (F): 2
```

---

## 🔗 **Integration dengan Sistem Deteksi**

### **Flow Integration**:

```
1. Kamera deteksi plat → OCR baca teks
2. Sistem cek database:
   SELECT * FROM kendaraan_terdaftar
   WHERE nomor_plat = ? AND status = 'aktif'
3. Result:
   - Found + aktif → BOLEH MASUK (palang buka)
   - Found + nonaktif → DITOLAK
   - Not found → DITOLAK
4. Log access ke log_akses_masuk
```

**Fungsi backend**: `process_vehicle_access()` di app.py:354

---

## 💡 **Tips & Best Practices**

### **Untuk Administrator**:

1. **Registrasi Kendaraan Baru**:
   - Pastikan format plat benar (contoh: B1234ABC, bukan B 1234 ABC)
   - Nama pemilik jelas (include role: Pak Budi - Guru TKJ)
   - Nomor HP untuk contact emergency

2. **Manajemen Status**:
   - Set `nonaktif` untuk kendaraan tidak boleh masuk sementara
   - **Jangan delete** kendaraan kecuali benar-benar sudah tidak dipakai
   - Update status lebih baik daripada delete (untuk histori)

3. **Pencarian Efisien**:
   - Gunakan search box untuk cari cepat
   - Ketik sebagian plat atau nama sudah cukup

4. **Backup Data**:
   ```bash
   mysqldump -u root sistem_parkir_smk > backup.sql
   ```

### **Untuk Developer**:

1. **Foreign Key Relationship**:
   - `log_akses_masuk.plat_terdeteksi` reference ke `kendaraan_terdaftar.nomor_plat`
   - Soft delete lebih baik (status nonaktif) daripada hard delete

2. **Performance**:
   - Index pada `nomor_plat` untuk query cepat
   - Limit 100 untuk log akses display

3. **Error Handling**:
   - Try-catch semua database operations
   - Flash message untuk user feedback
   - Logger untuk debugging

---

## 🚀 **Future Enhancements** (Optional)

1. **Export/Import**:
   - Export daftar kendaraan ke Excel/CSV
   - Import bulk data dari file

2. **Advanced Filter**:
   - Filter by jenis kendaraan
   - Filter by status
   - Date range filter

3. **Pagination**:
   - Jika data >100, implement pagination

4. **Photo Upload**:
   - Upload foto kendaraan/pemilik
   - Store di folder `vehicle_photos/`

5. **History Log**:
   - Track perubahan data (audit trail)
   - Who changed what when

6. **QR Code**:
   - Generate QR code untuk setiap kendaraan
   - Quick scan untuk manual override

---

## ✅ **Summary**

| Feature | Status | Route | Template |
|---------|--------|-------|----------|
| **Create** | ✅ Working | `POST /add_vehicle` | Modal form |
| **Read** | ✅ Working | `GET /vehicles` | vehicles.html |
| **Update** | ✅ Working | `POST /edit_vehicle/<id>` | Modal form |
| **Delete** | ✅ Working | `GET /delete_vehicle/<id>` | Confirm modal |
| **Search** | ✅ Working | JavaScript | Real-time filter |
| **Stats** | ✅ Working | Jinja2 filters | Dashboard |

---

## 📝 **Sample API Responses**

### **Success Cases**:

```json
// Add vehicle success
{
  "message": "✅ Kendaraan B1234ABC berhasil ditambahkan!",
  "redirect": "/vehicles"
}

// Update success
{
  "message": "✅ Data kendaraan berhasil diupdate!",
  "redirect": "/vehicles"
}

// Delete success
{
  "message": "✅ Kendaraan B1234ABC (Pak Budi) berhasil dihapus!",
  "redirect": "/vehicles"
}
```

### **Error Cases**:

```json
// Duplicate plate
{
  "message": "❌ Plat nomor B1234ABC sudah terdaftar!",
  "redirect": "/vehicles"
}

// Database error
{
  "message": "❌ Error menambah kendaraan: [error details]",
  "redirect": "/vehicles"
}

// Not found
{
  "message": "❌ Kendaraan tidak ditemukan!",
  "redirect": "/vehicles"
}
```

---

**Created**: 2025-10-16
**Status**: ✅ **PRODUCTION READY - CRUD Kendaraan Siap Digunakan!** 🎉

---

## 🎯 **Quick Start**

1. Pastikan MySQL running
2. Database `sistem_parkir_smk` sudah setup (lihat database_setup.sql)
3. Jalankan aplikasi: `python3 app.py`
4. Buka browser: `http://localhost:5001/vehicles`
5. Mulai kelola data kendaraan! 🚗🏍️
