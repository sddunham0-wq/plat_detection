# 🔧 MySQL Setup Guide - Fix Connection Issues

## ❌ Problem:
```
Error: (2003, "Can't connect to MySQL server on '127.0.0.1' [Errno 61]")
```

**Root Cause:** MySQL server tidak running!

---

## ✅ Solution: Start MySQL Server

### **Option 1: Homebrew (Mac)**
```bash
# Install MySQL (jika belum)
brew install mysql

# Start MySQL service
brew services start mysql

# Test connection
mysql -u root -p
```

### **Option 2: XAMPP/MAMP**
```bash
# 1. Buka XAMPP/MAMP Control Panel
# 2. Klik "Start" pada MySQL
# 3. Wait hingga status "Running"
```

### **Option 3: MySQL Manual**
```bash
# Start MySQL server
sudo /usr/local/mysql/support-files/mysql.server start

# Check status
sudo /usr/local/mysql/support-files/mysql.server status
```

### **Option 4: System Service**
```bash
# Ubuntu/Debian
sudo systemctl start mysql
sudo systemctl status mysql

# CentOS/RHEL
sudo systemctl start mysqld
sudo systemctl status mysqld
```

---

## 🔍 Verify MySQL is Running

```bash
# Check if MySQL is listening
lsof -i :3306

# Expected output:
# mysqld  1234  user  10u  IPv4  TCP *:mysql (LISTEN)

# Test connection
mysql -h 127.0.0.1 -P 3306 -u root -p
```

---

## 🗄️ Setup Database for CRUD

### **1. Create Database**
```bash
# Login to MySQL
mysql -u root -p

# Create database
CREATE DATABASE IF NOT EXISTS plat_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# Use database
USE plat_detection;

# Verify
SHOW DATABASES;
```

### **2. Create Tables**
```sql
-- Vehicles table (whitelist)
CREATE TABLE IF NOT EXISTS vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL UNIQUE,
    owner_name VARCHAR(100) NOT NULL,
    vehicle_type VARCHAR(50) NOT NULL,
    contact_info VARCHAR(100),
    status ENUM('Hadir', 'Belum') DEFAULT 'Belum',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_plate (plate_number),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Access log table
CREATE TABLE IF NOT EXISTS access_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_id INT NULL,
    plate_number VARCHAR(20) NOT NULL,
    acces_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    status ENUM('masuk', 'keluar', 'ditolak') NOT NULL,
    image_url VARCHAR(255),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE SET NULL,
    INDEX idx_plate (plate_number),
    INDEX idx_time (acces_time),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### **3. Insert Sample Data**
```sql
-- Add sample vehicles
INSERT INTO vehicles (plate_number, owner_name, vehicle_type, contact_info, status) VALUES
('B1234ABC', 'John Doe', 'Karyawan', '081234567890', 'Belum'),
('F5678XYZ', 'Jane Smith', 'Tamu', '082345678901', 'Belum'),
('D9999EFG', 'Bob Johnson', 'Vendor', '083456789012', 'Belum');

-- Add sample access log
INSERT INTO access_log (vehicle_id, plate_number, status, image_url) VALUES
(1, 'B1234ABC', 'masuk', 'detected_plates/plate_001.jpg'),
(2, 'F5678XYZ', 'masuk', 'detected_plates/plate_002.jpg'),
(NULL, 'UNKNOWN1', 'ditolak', 'detected_plates/plate_003.jpg');

-- Verify data
SELECT * FROM vehicles;
SELECT * FROM access_log;
```

---

## ⚙️ Update .env Configuration

```bash
# Edit .env file
nano /Users/andra/Documents/DWI/project-plat-detection-alfi/.env
```

**Make sure these settings:**
```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=plat_detection
USE_MYSQL_DATABASE=True
ENABLE_MYSQL_ACCESS_CONTROL=True
```

**Important:**
- Set `MYSQL_PASSWORD` to your actual MySQL root password
- If no password: leave empty (`MYSQL_PASSWORD=`)
- Port default: `3306` (change if different)

---

## 🧪 Test Connection

### **1. Test dari Terminal**
```bash
cd /Users/andra/Documents/DWI/project-plat-detection-alfi
python3 test_mysql_connection.py
```

**Expected output:**
```
Testing MySQL connection...
✅ Connection successful!

Database Statistics:
  Total Vehicles: 3
  Total Access Logs: 3
  Access Today: 0
```

### **2. Test dari Web**
```bash
# Start server
python3 headless_stream.py --port 5010

# Open browser
http://localhost:5010/vehicles
```

**Expected:**
- ✅ Shows vehicle list from MySQL
- ✅ Add/Edit/Delete buttons work
- ✅ Access log displays properly

---

## 🐛 Troubleshooting

### **Error: Access Denied**
```
ERROR 1045 (28000): Access denied for user 'root'@'localhost'
```

**Fix:**
```bash
# Reset MySQL root password
mysql -u root

# Inside MySQL:
ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
FLUSH PRIVILEGES;

# Update .env with new password
```

### **Error: Database Does Not Exist**
```
ERROR 1049 (42000): Unknown database 'plat_detection'
```

**Fix:**
```bash
mysql -u root -p -e "CREATE DATABASE plat_detection;"
```

### **Error: Can't Connect to Socket**
```
ERROR 2002 (HY000): Can't connect to local MySQL server through socket
```

**Fix:**
```bash
# Use TCP connection instead
mysql -h 127.0.0.1 -P 3306 -u root -p
```

### **Error: Port Already in Use**
```
ERROR 2003: Can't connect (port 3306)
```

**Fix:**
```bash
# Check what's using port 3306
lsof -i :3306

# Kill the process or use different port
# Update .env:
MYSQL_PORT=3307
```

---

## 📋 Quick Commands Reference

```bash
# Start MySQL
brew services start mysql

# Stop MySQL
brew services stop mysql

# Restart MySQL
brew services restart mysql

# Check status
brew services list | grep mysql

# Login to MySQL
mysql -u root -p

# Run setup script
python3 setup_database.py

# Test connection
python3 test_mysql_connection.py
```

---

## ✅ Verification Checklist

- [ ] MySQL server is running (`lsof -i :3306`)
- [ ] Database `plat_detection` exists
- [ ] Tables `vehicles` and `access_log` created
- [ ] `.env` file configured correctly
- [ ] Test connection successful
- [ ] Web interface loads data

---

## 🎯 Summary

**For CRUD to work, you need:**
1. ✅ MySQL server running
2. ✅ Database `plat_detection` created
3. ✅ Tables created (vehicles, access_log)
4. ✅ `.env` configured correctly
5. ✅ Sample data inserted (optional)

**After setup complete:**
```bash
python3 headless_stream.py --port 5010
```

Then access:
- Vehicles: http://localhost:5010/vehicles
- Access Log: http://localhost:5010/access-log

**Good luck!** 🚀
