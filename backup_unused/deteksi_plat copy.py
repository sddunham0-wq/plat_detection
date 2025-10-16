import cv2
import pytesseract
import os
from datetime import datetime
from PIL import Image
import mysql.connector  # Kalau sudah ada dari sebelumnya

# Path ke Tesseract (sesuaikan kalau perlu)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# INPUT CUSTOM: Masukkan URL IP Camera di sini (bebas ganti!)
URL_CAMERA = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"  # Ganti dengan IP camera-mu!
# Contoh lain:
# URL_CAMERA = "rtsp://username:pass@10.0.0.50:554/stream1"
# URL_CAMERA = "http://192.168.1.200:8080/video"  # Untuk HTTP

# Fungsi deteksi dan crop plat (sama seperti sebelumnya)
def deteksi_dan_crop_plat(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), closed=True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 30 and w / h > 1.5:
                plat_crop = frame[y:y+h, x:x+w]
                return plat_crop, (x, y, w, h)
    return None, None

# Fungsi simpan ke DB (sama seperti sebelumnya)
def simpan_ke_db(plat, foto_path):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # Kosong untuk Laragon
            database='deteksi_plat_db'  # Ganti kalau sudah rename
        )
        cursor = conn.cursor()
        query = "INSERT INTO tb_kendaraan (Plat_Nomor, Foto_Plat) VALUES (%s, %s)"
        cursor.execute(query, (plat, foto_path))
        conn.commit()
        print("Data disimpan ke database!")
    except Exception as e:
        print(f"Error DB: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# Mulai stream dari IP Camera
cap = cv2.VideoCapture(URL_CAMERA)  # Ganti dari 0 ke URL_CAMERA!

# Cek kalau stream gak bisa dibuka
if not cap.isOpened():
    print(f"Error: Gak bisa buka stream dari {URL_CAMERA}")
    print("Cek: IP benar? Koneksi jaringan? Username/password?")
    exit()

print(f"Stream dari {URL_CAMERA} berhasil dibuka!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gak bisa baca frame. Cek koneksi camera.")
        break
    
    # Deteksi plat
    plat_crop, bbox = deteksi_dan_crop_plat(frame)
    
    if plat_crop is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nama_file = f"gambarplat/plat_{timestamp}.jpg"
        cv2.imwrite(nama_file, plat_crop)
        
        teks_plat = pytesseract.image_to_string(plat_crop, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        teks_plat = teks_plat.strip().upper()
        
        print(f"Plat terdeteksi: {teks_plat}")
        print(f"Gambar disimpan: {nama_file}")
        
        # Gambar kotak
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, teks_plat, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Simpan ke DB
        simpan_ke_db(teks_plat, nama_file)
    
    # Tampilkan frame
    cv2.imshow('Deteksi Plat Nomor - IP Camera', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()