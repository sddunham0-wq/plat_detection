# MySQL Integration Guide - Access Control System

Panduan lengkap untuk menggunakan MySQL Access Control System dalam project License Plate Detection.

---

## 📋 **Table of Contents**

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Database Setup](#database-setup)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Frontend Integration](#frontend-integration)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 **Overview**

**MySQL Integration** menambahkan **Access Control System** ke project dengan fitur:

- ✅ **Whitelist Management** - Manage kendaraan terdaftar
- ✅ **Access Control** - Grant/Deny akses berdasarkan whitelist
- ✅ **Hybrid Mode** - SQLite (logging lengkap) + MySQL (access control)
- ✅ **Real-time Notifications** - WebSocket events untuk frontend
- ✅ **Access History** - Log semua akses kendaraan
- ✅ **Auto Status Update** - Update status kendaraan otomatis

---

## 📦 **Prerequisites**

### **System Requirements**
- Python 3.8+
- MySQL Server 5.7+ atau MariaDB 10.3+
- Existing project License Plate Detection

### **MySQL Server**
Anda membutuhkan MySQL server yang running. Pilihan:

**Option 1: XAMPP/MAMP (Recommended)**
- Download: [XAMPP](https://www.apachefriends.org/) atau [MAMP](https://www.mamp.info/)
- Easy GUI untuk manage database
- Includes phpMyAdmin

**Option 2: MySQL Standalone**
```bash
# macOS
brew install mysql

# Ubuntu/Debian
sudo apt install mysql-server

# Windows
# Download from: https://dev.mysql.com/downloads/mysql/
```

---

## 🚀 **Installation**

### **Step 1: Install Dependencies**

```bash
pip install pymysql python-dotenv
```

Atau install dari requirements.txt yang sudah diupdate:
```bash
pip install -r requirements.txt
```

### **Step 2: Configure MySQL Credentials**

File `.env` sudah dibuat dengan default configuration:

```env
# MySQL Server Connection
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3307
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=plat_detection

# Feature Flags
USE_MYSQL_DATABASE=True
ENABLE_ACCESS_CONTROL=True
ENABLE_SQLITE_LOGGING=True
```

**⚠️ Important**: Jika MySQL Anda menggunakan port berbeda atau memiliki password, edit file `.env` sesuai konfigurasi Anda.

---

## 🗄️ **Database Setup**

### **Option 1: Automated Setup (Recommended)**

```bash
python setup_database.py
```

Script ini akan:
- ✅ Check MySQL connection
- ✅ Create database `plat_detection` jika belum ada
- ✅ Import schema dari `contoh/plat_detection.sql`
- ✅ Verify tables creation
- ✅ Test connection

### **Option 2: Manual Setup**

**Via phpMyAdmin:**
1. Buka phpMyAdmin (http://localhost/phpmyadmin)
2. Create database: `plat_detection`
3. Select database
4. Import file: `contoh/plat_detection.sql`

**Via Command Line:**
```bash
# Create database
mysql -u root -P 3307 -h 127.0.0.1 -p -e "CREATE DATABASE plat_detection"

# Import schema
mysql -u root -P 3307 -h 127.0.0.1 -p plat_detection < contoh/plat_detection.sql
```

---

## 🧪 **Testing**

### **Test 1: Connection Test**

```bash
python test_mysql_connection.py
```

Expected output:
```
✅ Connection successful!
✅ Table 'vehicles' exists
✅ Table 'access_log' exists
📊 Total Vehicles: 1
📊 Sample vehicle found: F1818HG - bradpitt
```

### **Test 2: Integration Test**

```bash
python test_mysql_integration.py
```

Tests:
- ✅ Access granted (registered vehicle)
- ✅ Access denied (unregistered vehicle)
- ✅ Multiple detections
- ✅ Vehicle registration
- ✅ Access history
- ✅ Database statistics

---

## 🎮 **Usage**

### **Basic Usage - Headless Stream**

```bash
python headless_stream.py
```

MySQL access control **automatically enabled** berdasarkan `.env` configuration.

### **Python API Usage**

#### **Using AccessController Directly**

```python
from access_controller import AccessController
from utils.plate_detector import PlateDetection
import time

# Initialize controller
controller = AccessController()

# Test connection
if controller.test_connection():
    print("✅ MySQL connected!")

# Simulate detection
dummy_detection = PlateDetection(
    text="F1818HG",
    confidence=88.5,
    bbox=(100, 100, 200, 50),
    processed_image=None,
    timestamp=time.time()
)

# Process detection
result = controller.process_detection(dummy_detection)

if result['access'] == 'Authorized':
    print(f"✅ ACCESS AUTHORIZED")
    print(f"   Owner: {result['vehicle']['owner_name']}")
    print(f"   Type: {result['vehicle']['vehicle_type']}")
else:
    print(f"❌ ACCESS DENIED")
    print(f"   Reason: {result['reason']}")
```

#### **Register New Vehicle**

```python
from access_controller import AccessController

controller = AccessController()

# Register vehicle
result = controller.register_vehicle(
    plate_number="B1234XYZ",
    owner_name="John Doe",
    vehicle_type="karyawan",
    contact_info="081234567890"
)

if result['success']:
    print(f"✅ Vehicle registered with ID: {result['vehicle_id']}")
else:
    print(f"❌ Registration failed: {result['message']}")
```

#### **Query Access History**

```python
from access_controller import AccessController

controller = AccessController()

# Get recent access history
history = controller.get_access_history(limit=10)

for record in history:
    print(f"{record['plate_number']} - {record['status']} - {record['acces_time']}")

# Get history for specific vehicle
history = controller.get_access_history(plate_number="F1818HG", limit=50)
```

#### **Get Statistics**

```python
from access_controller import AccessController

controller = AccessController()

stats = controller.get_statistics()

print(f"Total Vehicles: {stats['database']['total_vehicles']}")
print(f"Access Today: {stats['database']['access_today']}")
print(f"Grant Rate: {stats['controller']['grant_rate']:.1f}%")
```

---

## 📡 **WebSocket Events**

### **Frontend - Listen to Events**

```javascript
const socket = io();

// Listen for access control results
socket.on('access_control_result', (data) => {
    console.log('Access Control Result:', data);

    if (data.access === 'Authorized') {
        // Show success notification
        showNotification({
            type: 'success',
            title: 'Access Granted',
            message: `Welcome, ${data.vehicle.owner_name}!`,
            plate: data.plate_number
        });
    } else {
        // Show denied notification
        showNotification({
            type: 'error',
            title: 'Access Denied',
            message: data.message,
            plate: data.plate_number
        });
    }
});

// Regular detection events (includes access_result if available)
socket.on('new_detection', (data) => {
    data.detections.forEach(detection => {
        if (detection.access_result) {
            console.log('Detection with access control:', detection);
        }
    });
});
```

### **Event Data Structure**

**`access_control_result` event:**
```json
{
  "access": "Authorized",
  "plate_number": "F1818HG",
  "confidence": 88.5,
  "vehicle": {
    "id": 1,
    "plate_number": "F1818HG",
    "owner_name": "bradpitt",
    "vehicle_type": "karyawan",
    "contact_info": "098982812",
    "status": "Hadir"
  },
  "access_log_id": 123,
  "timestamp": "2025-10-16T10:30:45.123Z",
  "message": "Selamat datang, bradpitt!"
}
```

**`access_control_result` event (denied):**
```json
{
  "access": "Denied",
  "plate_number": "TEST9999",
  "confidence": 92.3,
  "reason": "Vehicle not registered",
  "access_log_id": 124,
  "timestamp": "2025-10-16T10:31:15.456Z",
  "message": "Akses ditolak. Kendaraan TEST9999 tidak terdaftar."
}
```

---

## 🗃️ **Database Schema**

### **Table: `vehicles`**
```sql
CREATE TABLE vehicles (
    id INT PRIMARY KEY AUTO_INCREMENT,
    plate_number VARCHAR(10) NOT NULL UNIQUE,
    owner_name VARCHAR(100) NOT NULL,
    vehicle_type VARCHAR(20) NOT NULL,
    contact_info VARCHAR(50),
    status VARCHAR(10) DEFAULT 'Tidak Hadir',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### **Table: `access_log`**
```sql
CREATE TABLE access_log (
    id INT PRIMARY KEY AUTO_INCREMENT,
    vehicle_id INT,
    plate_number VARCHAR(10) NOT NULL,
    acces_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(10) NOT NULL,
    image_url VARCHAR(255),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
);
```

---

## ⚙️ **Configuration Options**

### **Environment Variables (`.env`)**

```env
# MySQL Connection
MYSQL_HOST=127.0.0.1          # MySQL server address
MYSQL_PORT=3307               # MySQL port
MYSQL_USER=root               # MySQL username
MYSQL_PASSWORD=               # MySQL password (empty for no password)
MYSQL_DATABASE=plat_detection # Database name

# Connection Pool
MYSQL_POOL_SIZE=5             # Max connections in pool
MYSQL_MAX_OVERFLOW=10         # Max overflow connections
MYSQL_POOL_TIMEOUT=30         # Connection timeout (seconds)

# Feature Flags
USE_MYSQL_DATABASE=True           # Enable MySQL integration
ENABLE_ACCESS_CONTROL=True        # Enable access control logic
LOG_DENIED_ACCESS=True            # Log denied access attempts
AUTO_UPDATE_VEHICLE_STATUS=True   # Auto-update vehicle status

# Dual Mode
ENABLE_SQLITE_LOGGING=True        # Keep SQLite logging active
ENABLE_MYSQL_ACCESS_CONTROL=True  # Enable MySQL access control
```

### **Python Configuration (`config.py`)**

```python
# Programmatically override config
from config import MySQLConfig

MySQLConfig.USE_MYSQL_DATABASE = True
MySQLConfig.ENABLE_ACCESS_CONTROL = True
```

---

## 🔧 **Troubleshooting**

### **Problem: Cannot connect to MySQL**

**Symptoms:**
```
❌ Cannot connect to MySQL server
```

**Solutions:**
1. Check MySQL server is running:
   ```bash
   # Check MySQL status
   mysql -u root -P 3307 -h 127.0.0.1 -p -e "SELECT 1"
   ```

2. Verify `.env` credentials:
   - Correct host and port
   - Valid username and password

3. Check firewall/network:
   - MySQL port (3307) not blocked
   - localhost/127.0.0.1 accessible

### **Problem: Tables not found**

**Symptoms:**
```
❌ Table 'vehicles' not found!
```

**Solutions:**
1. Import SQL schema:
   ```bash
   python setup_database.py
   ```

2. Manual import:
   ```bash
   mysql -u root -P 3307 -h 127.0.0.1 -p plat_detection < contoh/plat_detection.sql
   ```

### **Problem: Access always denied**

**Symptoms:**
```
❌ ACCESS DENIED for all vehicles
```

**Solutions:**
1. Check vehicles table has data:
   ```sql
   SELECT * FROM vehicles;
   ```

2. Verify plate_number match:
   - Case-sensitive matching
   - No extra spaces
   - Correct formatting

3. Register test vehicle:
   ```python
   from access_controller import AccessController
   controller = AccessController()
   controller.register_vehicle("F1818HG", "bradpitt", "karyawan", "098982812")
   ```

### **Problem: SQLite still used instead of MySQL**

**Symptoms:**
```
⚠️ MySQL requested but not available
```

**Solutions:**
1. Install MySQL dependencies:
   ```bash
   pip install pymysql python-dotenv
   ```

2. Check `.env` file exists and readable

3. Verify `USE_MYSQL_DATABASE=True` in `.env`

---

## 📚 **Additional Resources**

### **Files Created**
- `mysql_database.py` - MySQL connection handler
- `access_controller.py` - Access control logic
- `test_mysql_connection.py` - Connection testing
- `test_mysql_integration.py` - Integration testing
- `setup_database.py` - Automated database setup
- `.env` - MySQL credentials
- `.env.example` - Template for credentials

### **Files Modified**
- `requirements.txt` - Added pymysql, python-dotenv
- `config.py` - Added MySQLConfig class
- `stream_manager.py` - MySQL integration
- `headless_stream.py` - WebSocket events

### **Sample Data**
- `contoh/plat_detection.sql` - Database schema + sample vehicle

---

## 🎉 **Quick Start Summary**

```bash
# 1. Install dependencies
pip install pymysql python-dotenv

# 2. Configure credentials (edit .env if needed)
# Default: 127.0.0.1:3307, user: root, password: (empty)

# 3. Setup database
python setup_database.py

# 4. Test connection
python test_mysql_connection.py

# 5. Run integration tests
python test_mysql_integration.py

# 6. Start application
python headless_stream.py

# 7. Access web interface
# Open: http://localhost:5000
```

---

## 🤝 **Support**

Jika ada pertanyaan atau masalah:
1. Check troubleshooting section
2. Run `python test_mysql_connection.py` untuk diagnostic
3. Check MySQL server logs
4. Verify `.env` configuration

---

**✅ Setup Complete!** MySQL Access Control System siap digunakan! 🎉
