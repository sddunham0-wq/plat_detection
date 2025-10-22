# 🚨 QUICK FIX - MySQL Not Running

## Problem:
```
Error: Can't connect to MySQL server on '127.0.0.1' [Errno 61]
```

MySQL server berhenti/tidak running.

---

## ✅ SOLUTION (Pilih salah satu):

### **Option 1: Manual Start (RECOMMENDED)**
```bash
# Start MySQL di background
mysqld_safe &

# Atau
mysql.server start
```

### **Option 2: Homebrew (Jika Option 1 gagal)**
```bash
# Unload dulu
brew services stop mysql

# Load ulang
brew services start mysql

# Check status
brew services list | grep mysql
```

### **Option 3: Direct mysqld**
```bash
# Kill existing process
pkill -9 mysqld

# Start fresh
/opt/homebrew/bin/mysqld --datadir=/opt/homebrew/var/mysql &

# Wait 5 seconds
sleep 5

# Test
mysql -u root -p
```

---

## 🧪 Verify MySQL Running

```bash
# Check process
ps aux | grep mysqld

# Check port
lsof -i :3306

# Test connection
mysql -u root -h 127.0.0.1 -P 3306 -p
```

**Expected:**
```
mysql> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| plat_detection     |
| information_schema |
| mysql              |
+--------------------+
```

---

## 🎯 After MySQL Started

### **1. Test Database Connection**
```bash
cd /Users/andra/Documents/DWI/project-plat-detection-alfi
python3 test_mysql_connection.py
```

Expected:
```
✅ MySQL connection test successful
  Total Vehicles: X
  Total Access Logs: X
```

### **2. Start Flask Server**
```bash
python3 headless_stream.py --port 5010
```

### **3. Test CRUD Web**
```
http://localhost:5010/vehicles      ← Should show data
http://localhost:5010/access-log    ← Should show logs
```

---

## 📋 Manual Commands (Copy-Paste)

```bash
# 1. Start MySQL
mysql.server start

# 2. Wait 3 seconds
sleep 3

# 3. Check if running
lsof -i :3306

# 4. Test connection
python3 test_mysql_connection.py

# 5. Start Flask server
python3 headless_stream.py --port 5010
```

---

## ⚠️ If Still Not Working

**Kemungkinan data directory corrupt:**
```bash
# Backup data
cp -r /opt/homebrew/var/mysql /opt/homebrew/var/mysql.backup

# Reinitialize MySQL
mysqld --initialize-insecure --datadir=/opt/homebrew/var/mysql

# Start server
mysql.server start

# Reset root password (if needed)
mysql -u root
ALTER USER 'root'@'localhost' IDENTIFIED BY '';
FLUSH PRIVILEGES;
```

---

## 🆘 Last Resort - Reinstall MySQL

```bash
# Stop MySQL
brew services stop mysql

# Backup data
mysqldump -u root -p --all-databases > backup.sql

# Uninstall
brew uninstall mysql
rm -rf /opt/homebrew/var/mysql

# Reinstall
brew install mysql

# Start
brew services start mysql

# Restore data
mysql -u root < backup.sql
```

---

## 💡 Pro Tip

**Auto-start MySQL on boot:**
```bash
brew services start mysql
```

This ensures MySQL always runs when Mac boots up!

---

**Need more help?** Check full guide: `MYSQL_SETUP_GUIDE.md`
