-- ==========================================
-- Manual Override System Database Migration
-- ==========================================
-- Created: 2025-10-25
-- Purpose: Add manual override functionality for access control
-- ==========================================

USE plat_detection;

-- ==========================================
-- Table: manual_overrides
-- Purpose: Log semua manual override actions
-- ==========================================
CREATE TABLE IF NOT EXISTS manual_overrides (
    id INT PRIMARY KEY AUTO_INCREMENT,
    detection_id INT,
    original_plate VARCHAR(20),
    corrected_plate VARCHAR(20),
    original_decision ENUM('granted', 'denied', 'pending') DEFAULT 'pending',
    override_decision ENUM('approved', 'rejected') NOT NULL,
    reason VARCHAR(255),
    operator_pin VARCHAR(10),
    operator_name VARCHAR(100),
    duration ENUM('one-time', '1-hour', '1-day', 'permanent') DEFAULT 'one-time',
    expire_at DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_detection_id (detection_id),
    INDEX idx_corrected_plate (corrected_plate),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ==========================================
-- Table: temporary_access
-- Purpose: Temporary access permissions untuk kendaraan tidak terdaftar
-- ==========================================
CREATE TABLE IF NOT EXISTS temporary_access (
    id INT PRIMARY KEY AUTO_INCREMENT,
    plate_number VARCHAR(20) UNIQUE NOT NULL,
    granted_by VARCHAR(100),
    reason VARCHAR(255),
    duration ENUM('one-time', '1-hour', '1-day', 'permanent') DEFAULT 'one-time',
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expire_at DATETIME NULL,
    access_count INT DEFAULT 0,
    last_access DATETIME NULL,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_plate_number (plate_number),
    INDEX idx_expire_at (expire_at),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ==========================================
-- Table: alert_settings
-- Purpose: User preferences untuk alert notifications
-- ==========================================
CREATE TABLE IF NOT EXISTS alert_settings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(100) DEFAULT 'default' UNIQUE,

    -- Sound settings
    enable_audio BOOLEAN DEFAULT TRUE,
    audio_volume DECIMAL(3,2) DEFAULT 0.80,
    sound_denied BOOLEAN DEFAULT TRUE,
    sound_granted_auto BOOLEAN DEFAULT FALSE,
    sound_granted_manual BOOLEAN DEFAULT TRUE,
    sound_manual_required BOOLEAN DEFAULT TRUE,

    -- Visual settings
    auto_dismiss_seconds INT DEFAULT 5,
    max_visible_alerts INT DEFAULT 3,
    enable_grouping BOOLEAN DEFAULT TRUE,

    -- Priority filters
    show_critical BOOLEAN DEFAULT TRUE,
    show_high BOOLEAN DEFAULT TRUE,
    show_medium BOOLEAN DEFAULT FALSE,
    show_low BOOLEAN DEFAULT FALSE,

    -- Quiet hours
    enable_quiet_hours BOOLEAN DEFAULT FALSE,
    quiet_start_time TIME DEFAULT '22:00:00',
    quiet_end_time TIME DEFAULT '06:00:00',

    -- DND mode
    enable_dnd BOOLEAN DEFAULT FALSE,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ==========================================
-- Alter existing access_log table
-- Purpose: Add manual override tracking fields
-- ==========================================
ALTER TABLE access_log
    ADD COLUMN manual_override BOOLEAN DEFAULT FALSE,
    ADD COLUMN override_reason VARCHAR(255) NULL,
    ADD COLUMN reviewed_by VARCHAR(100) NULL,
    ADD COLUMN ocr_confidence DECIMAL(5,2) NULL,
    ADD INDEX idx_manual_override (manual_override);

-- ==========================================
-- Insert default alert settings
-- ==========================================
INSERT INTO alert_settings (user_id) VALUES ('default')
ON DUPLICATE KEY UPDATE user_id = 'default';

-- ==========================================
-- View: pending_reviews
-- Purpose: Quick access ke detections yang butuh manual review
-- ==========================================
CREATE OR REPLACE VIEW pending_reviews AS
SELECT
    al.id,
    al.plate_number,
    al.ocr_confidence,
    al.status,
    al.acces_time,
    al.image_url,
    v.owner_name,
    v.vehicle_type,
    CASE
        WHEN al.ocr_confidence < 60 THEN 'CRITICAL'
        WHEN al.ocr_confidence < 75 THEN 'HIGH'
        ELSE 'MEDIUM'
    END as priority
FROM access_log al
LEFT JOIN vehicles v ON al.vehicle_id = v.id
WHERE al.manual_override = FALSE
    AND al.ocr_confidence < 75
    AND al.acces_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
ORDER BY
    CASE
        WHEN al.ocr_confidence < 60 THEN 1
        WHEN al.ocr_confidence < 75 THEN 2
        ELSE 3
    END,
    al.acces_time DESC;

-- ==========================================
-- Stored Procedure: check_temporary_access
-- Purpose: Check apakah plate number punya temporary access yang masih valid
-- ==========================================
DELIMITER //

CREATE PROCEDURE IF NOT EXISTS check_temporary_access(
    IN p_plate_number VARCHAR(20),
    OUT p_has_access BOOLEAN,
    OUT p_reason VARCHAR(255)
)
BEGIN
    DECLARE v_expire_at DATETIME;
    DECLARE v_duration ENUM('one-time', '1-hour', '1-day', 'permanent');
    DECLARE v_access_count INT;

    -- Check if temporary access exists
    SELECT expire_at, duration, access_count, reason
    INTO v_expire_at, v_duration, v_access_count, p_reason
    FROM temporary_access
    WHERE plate_number = p_plate_number
        AND is_active = TRUE
    LIMIT 1;

    -- Check validity
    IF v_duration IS NOT NULL THEN
        IF v_duration = 'permanent' THEN
            SET p_has_access = TRUE;
        ELSEIF v_duration = 'one-time' AND v_access_count = 0 THEN
            SET p_has_access = TRUE;
            -- Increment access count
            UPDATE temporary_access
            SET access_count = access_count + 1,
                last_access = NOW()
            WHERE plate_number = p_plate_number;
        ELSEIF v_expire_at IS NOT NULL AND NOW() < v_expire_at THEN
            SET p_has_access = TRUE;
            -- Update last access
            UPDATE temporary_access
            SET access_count = access_count + 1,
                last_access = NOW()
            WHERE plate_number = p_plate_number;
        ELSE
            -- Expired or used
            SET p_has_access = FALSE;
            UPDATE temporary_access
            SET is_active = FALSE
            WHERE plate_number = p_plate_number;
        END IF;
    ELSE
        SET p_has_access = FALSE;
        SET p_reason = NULL;
    END IF;
END //

DELIMITER ;

-- ==========================================
-- Stored Procedure: cleanup_expired_temporary_access
-- Purpose: Auto-cleanup expired temporary access records
-- ==========================================
DELIMITER //

CREATE PROCEDURE IF NOT EXISTS cleanup_expired_temporary_access()
BEGIN
    -- Deactivate expired time-based access
    UPDATE temporary_access
    SET is_active = FALSE
    WHERE is_active = TRUE
        AND expire_at IS NOT NULL
        AND NOW() > expire_at;

    -- Deactivate used one-time access
    UPDATE temporary_access
    SET is_active = FALSE
    WHERE is_active = TRUE
        AND duration = 'one-time'
        AND access_count > 0;

    SELECT ROW_COUNT() as cleaned_records;
END //

DELIMITER ;

-- ==========================================
-- Event: Auto-cleanup every hour
-- ==========================================
SET GLOBAL event_scheduler = ON;

CREATE EVENT IF NOT EXISTS cleanup_temporary_access_event
ON SCHEDULE EVERY 1 HOUR
DO CALL cleanup_expired_temporary_access();

-- ==========================================
-- Grant permissions (adjust as needed)
-- ==========================================
-- GRANT SELECT, INSERT, UPDATE ON plat_detection.* TO 'cctv_user'@'localhost';

-- ==========================================
-- Verification queries
-- ==========================================
-- Show created tables
SELECT 'Tables created:' as status;
SHOW TABLES LIKE '%override%';
SHOW TABLES LIKE '%temporary_access%';
SHOW TABLES LIKE '%alert_settings%';

-- Show modified columns
SELECT 'access_log new columns:' as status;
SHOW COLUMNS FROM access_log LIKE '%override%';

-- Show views
SELECT 'Views created:' as status;
SHOW FULL TABLES WHERE TABLE_TYPE LIKE 'VIEW';

-- Show procedures
SELECT 'Stored procedures:' as status;
SHOW PROCEDURE STATUS WHERE Db = 'plat_detection';

-- Show events
SELECT 'Scheduled events:' as status;
SHOW EVENTS;

SELECT '✅ Manual Override System Migration Complete!' as status;
