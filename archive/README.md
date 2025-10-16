# Archive - Old Detection Files

Folder ini berisi file-file lama yang tidak lagi digunakan dalam sistem.

## Files

### `deteksi_plat.py`
- **Status**: Deprecated (tidak dipakai lagi)
- **Alasan**: Code lama dengan MySQL connector dan hardcoded credentials
- **Replacement**: `app.py` dengan `utils/` modules yang lebih clean

### `deteksi_plat_enhanced.py`
- **Status**: Deprecated (tidak dipakai lagi)
- **Alasan**: Masih pakai MySQL, tidak konsisten dengan app.py yang pakai SQLite
- **Replacement**: `app.py` sudah include enhanced features

## Sistem Aktif

Sistem yang **currently active** adalah:
- **Main App**: `app.py` (Flask web interface)
- **Detection**: `utils/plate_detector.py`
- **OCR**: `utils/ocr_processor.py`
- **Validation**: `utils/plate_validator.py`
- **Vehicle Analysis**: `utils/vehicle_analyzer.py`
- **Database**: SQLite (`sistem_parkir_smk.db`)

## Kenapa di-Archive?

1. **Code Duplication**: 3 file berbeda dengan logika yang overlap
2. **Inconsistent Database**: MySQL vs SQLite
3. **Security Issues**: Hardcoded credentials
4. **Maintenance**: Lebih mudah maintain 1 codebase

## Restore Instructions

Jika perlu restore file lama:
```bash
# Copy kembali ke root folder
cp archive/deteksi_plat.py ../

# Atau edit langsung di archive folder
```

**Note**: Tidak disarankan untuk re-use file ini. Gunakan `app.py` sebagai base.

---

📅 Archived: January 2025
🔧 Maintained by: Project Team
