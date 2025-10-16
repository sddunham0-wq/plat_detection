# 🍎 Setup Guide untuk macOS Server

**Panduan lengkap setup MySQL di macOS untuk project Sistem Parkir SMK**

---

## ⚡ Quick Start (Otomatis)

Jalankan script setup otomatis:

```bash
cd /Users/andra/Documents/DWI/project-plat-detection-dude
./setup_mysql_macos.sh
```

Script ini akan:
1. ✅ Install Homebrew (kalau belum ada)
2. ✅ Install MySQL via Homebrew
3. ✅ Start MySQL service
4. ✅ Create database `sistem_parkir_smk`
5. ✅ Import schema & sample data
6. ✅ Verify installation

**Estimasi waktu:** 5-10 menit (tergantung koneksi internet)

---

## 🔧 Manual Setup (Step-by-Step)

Kalau script otomatis gagal, ikuti langkah manual:

### **Step 1: Install Homebrew**

```bash
# Cek apakah Homebrew sudah installed
brew --version

# Kalau belum, install:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Untuk Apple Silicon (M1/M2/M3), tambahkan ke PATH:
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

---

### **Step 2: Install MySQL**

```bash
# Install MySQL
brew install mysql

# Cek versi
mysql --version
```

**Expected output:**
```
mysql  Ver 8.0.xx for macos13.x on arm64 (Homebrew)
```

---

### **Step 3: Start MySQL Service**

```bash
# Start MySQL (auto-start on boot)
brew services start mysql

# Cek status
brew services list | grep mysql
```

**Expected output:**
```
mysql started andra ~/Library/LaunchAgents/homebrew.mxcl.mysql.plist
```

---

### **Step 4: Test MySQL Connection**

```bash
# Login ke MySQL (password kosong, tekan Enter saja)
mysql -u root

# Di MySQL prompt, cek database:
SHOW DATABASES;

# Exit
EXIT;
```

---

### **Step 5: Create Database**

```bash
# Login ke MySQL
mysql -u root

# Create database
CREATE DATABASE sistem_parkir_smk CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# Use database
USE sistem_parkir_smk;

# Import schema (ganti path sesuai lokasi project)
SOURCE /Users/andra/Documents/DWI/project-plat-detection-dude/database_setup.sql;

# Verify tables
SHOW TABLES;

# Exit
EXIT;
```

**Expected tables:**
- `kendaraan_terdaftar`
- `log_akses_masuk`
- `statistik_akses_harian` (view)
- `kendaraan_paling_aktif` (view)

---

### **Step 6: Test Connection dari Python**

```bash
cd /Users/andra/Documents/DWI/project-plat-detection-dude
python3 test_mysql_connection.py
```

**Expected output:**
```
============================================================
  TEST KONEKSI MYSQL - Sistem Parkir SMK
============================================================

✅ Koneksi ke MySQL Server BERHASIL!
📊 MySQL Version: 8.0.xx

✅ Koneksi ke Database 'sistem_parkir_smk' BERHASIL!

📊 Tabel yang ditemukan:
   - kendaraan_terdaftar: 19 records
   - log_akses_masuk: 9 records

✅ Query test BERHASIL!
📄 Sample data:
   - Plat: B1234ABC
   - Pemilik: Pak Budi - Guru TKJ
   - Status: aktif

============================================================
  ✅ SEMUA TEST BERHASIL!
============================================================
```

---

### **Step 7: Run Application**

```bash
# Start Flask app
python3 app.py

# Buka browser:
# http://localhost:5001
```

---

## 🔍 Troubleshooting

### **Problem 1: Command not found: brew**

**Solution:** Homebrew belum terinstall atau belum di PATH

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Untuk Apple Silicon, add to PATH:
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile
```

---

### **Problem 2: MySQL service won't start**

**Solution:**

```bash
# Stop existing service
brew services stop mysql

# Check if port 3306 is used
lsof -i :3306

# If port is busy, kill process:
kill -9 <PID>

# Start again
brew services start mysql
```

---

### **Problem 3: Access denied for user 'root'**

**Solution:** Reset MySQL root password

```bash
# Stop MySQL
brew services stop mysql

# Start in safe mode
mysqld_safe --skip-grant-tables &

# Reset password (di terminal baru)
mysql -u root
FLUSH PRIVILEGES;
ALTER USER 'root'@'localhost' IDENTIFIED BY '';
EXIT;

# Kill safe mode & restart normal
killall mysqld
brew services start mysql
```

---

### **Problem 4: Database schema import failed**

**Solution:** Import manual step-by-step

```bash
mysql -u root sistem_parkir_smk

# Copy-paste isi database_setup.sql satu per satu
# atau
SOURCE /full/path/to/database_setup.sql;
```

---

## 💡 Useful Commands

### **MySQL Service Management**

```bash
# Start MySQL
brew services start mysql

# Stop MySQL
brew services stop mysql

# Restart MySQL
brew services restart mysql

# Check status
brew services list | grep mysql
```

### **MySQL Client**

```bash
# Login
mysql -u root

# Login to specific database
mysql -u root sistem_parkir_smk

# Run SQL file
mysql -u root sistem_parkir_smk < file.sql

# Export database
mysqldump -u root sistem_parkir_smk > backup.sql
```

### **Database Operations**

```bash
# Inside MySQL:

# Show databases
SHOW DATABASES;

# Use database
USE sistem_parkir_smk;

# Show tables
SHOW TABLES;

# Show table structure
DESCRIBE kendaraan_terdaftar;

# Count records
SELECT COUNT(*) FROM kendaraan_terdaftar;

# View sample data
SELECT * FROM kendaraan_terdaftar LIMIT 5;
```

---

## 🔐 Security Notes

**Default Configuration (Development):**
- User: `root`
- Password: (empty)
- Port: `3306`
- Host: `localhost`

**⚠️ PENTING untuk Production:**

Jalankan security setup:

```bash
mysql_secure_installation
```

Answers:
- Set root password? → **YES** (buat password kuat)
- Remove anonymous users? → **YES**
- Disallow root login remotely? → **YES**
- Remove test database? → **YES**
- Reload privilege tables? → **YES**

Lalu update `.env`:
```env
DB_PASSWORD=your_secure_password_here
```

---

## 📞 Support

Kalau masih ada masalah:

1. Cek log MySQL:
   ```bash
   brew services list
   tail -f /opt/homebrew/var/mysql/*.err
   ```

2. Cek Python dependencies:
   ```bash
   pip3 list | grep mysql
   ```

3. Cek .env configuration:
   ```bash
   cat .env | grep DB_
   ```

---

**Good luck! 🚀**
