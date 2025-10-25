# Manual Override System - Installation Guide

## 📋 Overview

Manual Override System menambahkan fitur:
- ✅ Manual approve/reject access control decisions
- ✅ OCR correction untuk plat yang salah terbaca
- ✅ Temporary access untuk tamu/kendaraan tidak terdaftar
- ✅ Real-time alerts dengan anti-spam protection
- ✅ Audit trail untuk semua manual actions
- ✅ User-configurable alert preferences

---

## 🗄️ Database Migration

### Step 1: Backup Database (IMPORTANT!)

```bash
# Backup existing database
mysqldump -u root -p cctv_access_control > backup_before_override_$(date +%Y%m%d).sql

# Verify backup
ls -lh backup_before_override_*.sql
```

### Step 2: Run Migration Script

```bash
# Navigate to project directory
cd /Users/andra/Documents/DWI/project-plat-detection-alfi

# Run migration script
mysql -u root -p cctv_access_control < scripts/create_override_tables.sql
```

**Expected Output:**
```
Tables created:
+-------------------+
| manual_overrides  |
| temporary_access  |
| alert_settings    |
+-------------------+

access_log new columns:
+------------------+
| manual_override  |
| override_reason  |
| reviewed_by      |
| ocr_confidence   |
+------------------+

Views created:
+-----------------+
| pending_reviews |
+-----------------+

Stored procedures:
+---------------------------+
| check_temporary_access    |
| cleanup_expired_temporary |
+---------------------------+

✅ Manual Override System Migration Complete!
```

### Step 3: Verify Migration

```bash
# Login to MySQL
mysql -u root -p cctv_access_control

# Check tables
SHOW TABLES LIKE '%override%';
SHOW TABLES LIKE '%temporary%';
SHOW TABLES LIKE '%alert%';

# Check new columns in access_log
DESCRIBE access_log;

# Check default alert settings
SELECT * FROM alert_settings;

# Exit MySQL
EXIT;
```

---

## ⚙️ Configuration

### Environment Variables (Optional)

Create or update `.env` file:

```bash
# Manual Override Settings
OVERRIDE_PIN=1234                           # Change this in production!
ENABLE_MANUAL_OVERRIDE=True
OCR_CONFIDENCE_THRESHOLD=75.0
ENABLE_AUDIO_ALERTS=True
ALERT_VOLUME=0.8

# Anti-Spam Settings
DUPLICATE_COOLDOWN_PERIOD=30
MIN_SOUND_INTERVAL=2.0
BATCH_TIMEOUT=3.0
BUSY_THRESHOLD=20

# Quiet Hours
ENABLE_QUIET_HOURS=False
QUIET_START_TIME=22:00
QUIET_END_TIME=06:00
```

### Verify Configuration

```python
# Test configuration
python3 -c "
from config import OverrideConfig
print(f'Override Enabled: {OverrideConfig.ENABLE_MANUAL_OVERRIDE}')
print(f'OCR Threshold: {OverrideConfig.OCR_CONFIDENCE_THRESHOLD}%')
print(f'Override PIN: {OverrideConfig.OVERRIDE_PIN}')
print(f'Audio Alerts: {OverrideConfig.ENABLE_AUDIO_ALERTS}')
"
```

---

## 🧪 Testing Database Connection

```bash
# Test MySQL connection dengan override methods
python3 test_mysql_connection.py
```

**Expected Output:**
```
Testing MySQL connection...
✅ Connection successful!

Database Statistics:
  Total Vehicles: X
  Total Access Logs: Y
  Access Today: Z
```

### Test Override Methods

```python
# Create test script: test_override.py
from mysql_database import MySQLPlateDatabase

db = MySQLPlateDatabase.get_instance()

# Test 1: Get pending reviews
print("\n=== Test 1: Pending Reviews ===")
reviews = db.get_pending_reviews(limit=5)
print(f"Found {len(reviews)} pending reviews")

# Test 2: Grant temporary access
print("\n=== Test 2: Temporary Access ===")
success = db.grant_temporary_access(
    plate_number="TEST123",
    granted_by="operator",
    reason="Testing override system",
    duration="1-hour"
)
print(f"Grant access: {'✅ Success' if success else '❌ Failed'}")

# Test 3: Check temporary access
print("\n=== Test 3: Check Access ===")
has_access, reason = db.check_temporary_access("TEST123")
print(f"Has access: {has_access}")
print(f"Reason: {reason}")

# Test 4: Get alert settings
print("\n=== Test 4: Alert Settings ===")
settings = db.get_alert_settings('default')
print(f"Audio enabled: {settings.get('enable_audio')}")
print(f"Volume: {settings.get('audio_volume')}")

db.close_all_connections()
```

Run test:
```bash
python3 test_override.py
```

---

## 📁 File Structure

```
project-plat-detection-alfi/
├── scripts/
│   ├── create_override_tables.sql      # ✅ Database migration script
│   └── README_OVERRIDE_SETUP.md        # ✅ This file
├── utils/
│   └── alert_anti_spam.py              # ✅ Anti-spam system classes
├── config.py                            # ✅ Updated with OverrideConfig
├── mysql_database.py                    # ✅ Updated with override methods
├── headless_stream.py                   # ⏳ TODO: Add API endpoints
└── templates/
    └── access_override.html             # ⏳ TODO: Create UI
```

---

## 🔐 Security Considerations

### 1. Change Default PIN

**IMPORTANT:** Change default PIN before production!

```python
# In .env file
OVERRIDE_PIN=your_secure_pin_here

# Or in config.py
class OverrideConfig:
    OVERRIDE_PIN = os.getenv('OVERRIDE_PIN', 'your_secure_pin')
```

### 2. Database Permissions

```sql
-- Grant permissions for override tables
GRANT SELECT, INSERT, UPDATE, DELETE ON cctv_access_control.manual_overrides TO 'cctv_user'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON cctv_access_control.temporary_access TO 'cctv_user'@'localhost';
GRANT SELECT, INSERT, UPDATE ON cctv_access_control.alert_settings TO 'cctv_user'@'localhost';

-- Allow execution of stored procedures
GRANT EXECUTE ON PROCEDURE cctv_access_control.check_temporary_access TO 'cctv_user'@'localhost';
GRANT EXECUTE ON PROCEDURE cctv_access_control.cleanup_expired_temporary_access TO 'cctv_user'@'localhost';

FLUSH PRIVILEGES;
```

### 3. Audit Trail

All manual override actions are logged in `manual_overrides` table:
- Who performed the action (operator_pin, operator_name)
- What was changed (original_plate → corrected_plate)
- When it happened (created_at timestamp)
- Why it was done (reason field)
- Access duration and expiry

---

## 📊 Monitoring & Maintenance

### Auto-Cleanup Scheduled Event

Database automatically cleans up expired temporary access every hour:

```sql
-- Check scheduled event status
SHOW EVENTS LIKE 'cleanup_temporary_access_event';

-- Manually run cleanup
CALL cleanup_expired_temporary_access();
```

### Monitor Override Usage

```sql
-- Check recent manual overrides
SELECT * FROM manual_overrides
ORDER BY created_at DESC
LIMIT 10;

-- Check active temporary access
SELECT * FROM temporary_access
WHERE is_active = TRUE;

-- Check alert settings by user
SELECT * FROM alert_settings;

-- Check pending reviews
SELECT * FROM pending_reviews;
```

---

## 🐛 Troubleshooting

### Issue: Migration fails with "Table already exists"

**Solution:**
```bash
# Drop existing tables (CAUTION!)
mysql -u root -p cctv_access_control -e "
DROP TABLE IF EXISTS manual_overrides;
DROP TABLE IF EXISTS temporary_access;
DROP TABLE IF EXISTS alert_settings;
"

# Re-run migration
mysql -u root -p cctv_access_control < scripts/create_override_tables.sql
```

### Issue: Cannot import OverrideConfig

**Solution:**
```bash
# Verify config.py has OverrideConfig class
grep -n "class OverrideConfig" config.py

# Test import
python3 -c "from config import OverrideConfig; print('✅ Import successful')"
```

### Issue: Stored procedures not created

**Solution:**
```bash
# Check delimiter setting
mysql -u root -p cctv_access_control -e "
DELIMITER //
CREATE PROCEDURE test_proc() BEGIN SELECT 1; END //
DELIMITER ;
CALL test_proc();
DROP PROCEDURE test_proc;
"
```

---

## 🚀 Next Steps

After successful database migration:

1. **Backend Integration** (⏳ Pending)
   - Add API endpoints in `headless_stream.py`
   - Integrate anti-spam system
   - Add WebSocket events

2. **Frontend Development** (⏳ Pending)
   - Create access_override.html template
   - Implement JavaScript alert manager
   - Add sound files

3. **Testing**
   - End-to-end workflow testing
   - Load testing with anti-spam
   - User acceptance testing

---

## 📝 Migration Rollback

If you need to rollback migration:

```bash
# Restore from backup
mysql -u root -p cctv_access_control < backup_before_override_YYYYMMDD.sql

# Or manually drop tables
mysql -u root -p cctv_access_control -e "
DROP TABLE IF EXISTS manual_overrides;
DROP TABLE IF EXISTS temporary_access;
DROP TABLE IF EXISTS alert_settings;
DROP VIEW IF EXISTS pending_reviews;
DROP PROCEDURE IF EXISTS check_temporary_access;
DROP PROCEDURE IF EXISTS cleanup_expired_temporary_access;
DROP EVENT IF EXISTS cleanup_temporary_access_event;

ALTER TABLE access_log
DROP COLUMN IF EXISTS manual_override,
DROP COLUMN IF EXISTS override_reason,
DROP COLUMN IF EXISTS reviewed_by,
DROP COLUMN IF EXISTS ocr_confidence;
"
```

---

## 📞 Support

For issues or questions:
- Check logs: `logs/` directory
- Database errors: Check MySQL error log
- Configuration issues: Verify `.env` and `config.py`

---

**Installation Date:** 2025-10-25
**Version:** 1.0.0
**Status:** ✅ Backend Ready | ⏳ Frontend Pending
