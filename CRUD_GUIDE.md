# 🚗 CCTV Access Control CRUD System - User Guide

Sistem CRUD lengkap untuk manajemen kendaraan dan log akses dengan Flask Web Interface.

---

## 📋 Fitur yang Diimplementasikan

### ✅ 1. **CRUD Kendaraan (Vehicles Management)**
- **Create**: Tambah kendaraan baru ke whitelist
- **Read**: Tampilkan daftar semua kendaraan terdaftar
- **Update**: Edit informasi kendaraan yang sudah ada
- **Delete**: Hapus kendaraan dengan PIN protection (1234)

### ✅ 2. **Daftar Kendaraan di Web**
- Tabel interaktif dengan DataTables.js (search, sort, pagination)
- Filter by status (Hadir/Belum) dan vehicle type
- Real-time statistics cards (Total, Present, Absent, Today's Access)
- Auto-refresh setiap 30 detik

### ✅ 3. **Log Akses + Export CSV**
- Halaman log akses dengan filter advanced:
  - Date range (Today, Yesterday, Last 7 days, Last 30 days, Custom)
  - Status filter (Masuk/Keluar/Ditolak)
  - Plate number search
- Export to CSV dengan Excel compatibility (UTF-8 BOM)
- Image preview untuk gambar plat yang tersimpan

---

## 🚀 Cara Menjalankan Sistem

### **1. Start Flask Server**

```bash
cd /Users/andra/Documents/DWI/project-plat-detection-alfi
python3 headless_stream.py --port 5010
```

### **2. Akses Web Interface**

Buka browser dan kunjungi:
- **Live Stream**: http://localhost:5010/
- **Vehicles Management**: http://localhost:5010/vehicles
- **Access Log**: http://localhost:5010/access-log

---

## 📖 Panduan Penggunaan

### **A. Manajemen Kendaraan (`/vehicles`)**

#### **1. Tambah Kendaraan Baru**
1. Klik tombol **"Add New Vehicle"**
2. Isi form:
   - **Plate Number**: Nomor plat (format Indonesia, e.g., B1234ABC)
   - **Owner Name**: Nama pemilik kendaraan
   - **Vehicle Type**: Pilih tipe (Karyawan/Tamu/Vendor/Management/Other)
   - **Contact Info**: Nomor telepon/email (opsional)
3. Klik **"Add Vehicle"**
4. Kendaraan akan muncul di daftar dan siap di-detect oleh CCTV

#### **2. Edit Kendaraan**
1. Di tabel vehicles, klik tombol **Edit** (icon pensil)
2. Ubah informasi yang diperlukan
3. Klik **"Update Vehicle"**

#### **3. Hapus Kendaraan**
1. Di tabel vehicles, klik tombol **Delete** (icon trash)
2. Masukkan PIN: **1234** (untuk konfirmasi)
3. Klik **"Delete"**

#### **4. Status Kendaraan**
- **Hadir** (hijau): Kendaraan sudah terdeteksi masuk hari ini
- **Belum** (abu-abu): Kendaraan belum terdeteksi hari ini
- Status auto-update ketika kendaraan terdeteksi CCTV

---

### **B. Log Akses (`/access-log`)**

#### **1. Filter Log**
- **Date Range**: Pilih periode (Today/Last 7 days/Last 30 days/Custom)
- **Status**: Filter by Masuk/Keluar/Ditolak
- **Search**: Cari berdasarkan plate number
- Klik **"Apply Filters"**

#### **2. Export CSV**
1. Set filter yang diinginkan (opsional)
2. Klik tombol **"Export to CSV"**
3. File CSV akan download otomatis
4. Buka dengan Excel (auto-detect UTF-8)

#### **3. View Image**
- Klik tombol **"View"** di kolom Image
- Modal akan muncul menampilkan gambar plat yang terdeteksi

---

## 🗂️ Struktur File yang Dibuat

```
project-plat-detection-alfi/
├── headless_stream.py          # ✅ UPDATED: Tambah CRUD routes
├── templates/
│   ├── layout.html             # ✅ NEW: Base template dengan navbar
│   ├── vehicles.html           # ✅ NEW: Daftar kendaraan + CRUD
│   ├── vehicle_form.html       # ✅ NEW: Form add/edit kendaraan
│   ├── access_log.html         # ✅ NEW: Log akses + filter
│   └── stream.html             # EXISTING (live detection)
├── static/
│   ├── css/
│   │   └── custom.css          # ✅ NEW: Custom styling
│   └── js/
│       ├── vehicles.js         # ✅ NEW: CRUD operations (AJAX)
│       └── access_log.js       # ✅ NEW: Log filtering + CSV export
└── mysql_database.py           # EXISTING (sudah lengkap!)
```

---

## 🔗 Routes & Endpoints

### **Web Pages**
| Route | Method | Deskripsi |
|-------|--------|-----------|
| `/` | GET | Live stream detection (existing) |
| `/vehicles` | GET | Daftar kendaraan terdaftar |
| `/vehicles/add` | GET/POST | Form tambah kendaraan baru |
| `/vehicles/edit/<id>` | GET/POST | Form edit kendaraan |
| `/vehicles/delete/<id>` | POST | Hapus kendaraan (PIN: 1234) |
| `/access-log` | GET | Log akses dengan filter |
| `/access-log/export` | GET | Download CSV export |

### **API Endpoints**
| Route | Method | Deskripsi |
|-------|--------|-----------|
| `/api/vehicles/stats` | GET | Statistics kendaraan (AJAX) |
| `/api/access-log/stats` | GET | Statistics log akses (AJAX) |

---

## 💾 Database Schema (MySQL)

### **Table: `vehicles`**
```sql
id              INT (PK)
plate_number    VARCHAR (UNIQUE) - Nomor plat kendaraan
owner_name      VARCHAR - Nama pemilik
vehicle_type    VARCHAR - Jenis kendaraan (Karyawan/Tamu/dll)
contact_info    VARCHAR - Kontak (opsional)
status          ENUM - 'Hadir' atau 'Belum'
created_at      DATETIME
updated_at      DATETIME
```

### **Table: `access_log`**
```sql
id              INT (PK)
vehicle_id      INT (FK) - NULL untuk kendaraan tidak terdaftar
plate_number    VARCHAR - Nomor plat yang terdeteksi
acces_time      DATETIME - Waktu deteksi
status          ENUM - 'masuk', 'keluar', 'ditolak'
image_url       VARCHAR - Path ke gambar plat
```

---

## 🎨 UI/UX Features

### **Bootstrap 5**
- Responsive design (mobile-friendly)
- Modern card-based layout
- Color-coded status badges

### **DataTables.js**
- Interactive table dengan search, sort, pagination
- Default: 25 records per page (vehicles), 50 records per page (access log)
- Real-time filtering

### **AJAX Operations**
- Delete tanpa reload halaman
- Auto-refresh statistics setiap 30 detik
- Toast notifications untuk feedback

### **Form Validation**
- Client-side validation (HTML5 + JavaScript)
- Server-side validation (Flask)
- Auto-uppercase untuk plate number
- Indonesian plate format validation

---

## 🔐 Security Features

### **PIN Protection**
- Delete operations require PIN (default: **1234**)
- Modal confirmation untuk prevent accidental deletion

### **Input Validation**
- Plate number format validation (Indonesian format)
- Required field checking
- XSS protection (automatic by Flask)

### **Database**
- Prepared statements (prevent SQL injection)
- Transaction support (rollback on error)
- Connection pooling (optimized for multi-user)

---

## 📊 Sample Data Flow

### **Skenario 1: Tambah Kendaraan Baru**
```
User → /vehicles/add
     → Fill form: B1234ABC, John Doe, Karyawan
     → Submit
     → MySQL: INSERT INTO vehicles
     → Redirect to /vehicles
     → Success notification
```

### **Skenario 2: CCTV Detect Kendaraan**
```
CCTV → Detect plate: B1234ABC
     → AccessController.process_detection()
     → MySQL: Check whitelist (vehicles table)
     → Found! Status: Authorized
     → MySQL: Log access (access_log table)
     → MySQL: UPDATE vehicles SET status='Hadir'
     → WebSocket: Emit to frontend
     → UI: Show "ACCESS AUTHORIZED - John Doe"
```

### **Skenario 3: Export CSV**
```
User → /access-log
     → Set filter: Last 7 days, Status: Masuk
     → Click "Export CSV"
     → /access-log/export?date_range=last7days&status=masuk
     → MySQL: SELECT with filters
     → Generate CSV in memory
     → Download: access_log_20241022_153000.csv
```

---

## 🐛 Troubleshooting

### **1. MySQL Connection Error**
```bash
# Check MySQL server running
mysql -u root -p

# Check .env file
cat .env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=plat_detection
```

### **2. Templates Not Found**
```bash
# Pastikan folder templates ada
ls -la templates/

# Restart Flask server
pkill -f headless_stream.py
python3 headless_stream.py
```

### **3. Static Files Not Loading**
```bash
# Clear browser cache: Ctrl+Shift+R (Chrome)
# Check static folder
ls -la static/css/ static/js/
```

### **4. DELETE PIN Not Working**
- Default PIN: **1234**
- Ubah di `headless_stream.py` line 882 jika perlu
```python
if pin != '1234':  # Ganti dengan PIN custom
```

---

## 🎯 Tips & Best Practices

### **1. Data Entry**
- Gunakan format plat Indonesia yang benar: **B1234ABC** (tanpa spasi)
- Isi contact info untuk memudahkan komunikasi
- Pilih vehicle type yang sesuai untuk filtering

### **2. Log Management**
- Export CSV secara berkala untuk backup
- Gunakan filter date range untuk performa optimal
- Limit query: 1000 records (untuk mencegah overload)

### **3. Performance**
- Auto-refresh statistics dimatikan saat page inactive
- DataTables caching untuk faster load
- Connection pooling untuk MySQL (max 3 connections)

---

## 📞 Support

Jika ada masalah atau pertanyaan:
1. Check MySQL connection: `python3 test_mysql_connection.py`
2. Check logs: `tail -f logs/*.log`
3. Restart system: `./restart_stream.sh`

---

## ✨ Summary

Sistem CRUD lengkap sudah siap digunakan dengan fitur:
- ✅ Full CRUD operations untuk kendaraan
- ✅ Real-time data dari MySQL database
- ✅ Interactive web interface dengan Bootstrap 5
- ✅ Filter & search functionality
- ✅ CSV export (Excel-compatible)
- ✅ PIN protection untuk delete
- ✅ Auto-refresh statistics
- ✅ Responsive design (mobile-friendly)

**Selamat menggunakan sistem CCTV Access Control!** 🚀
