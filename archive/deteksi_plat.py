import cv2  # Untuk proses gambar/video
import pytesseract  # Untuk OCR (baca teks)
import os  # Untuk buat folder dan simpan file
from datetime import datetime  # Untuk timestamp
from PIL import Image  # Untuk crop gambar

# Path ke Tesseract (sesuaikan jika perlu, Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Fungsi untuk deteksi dan crop plat nomor
def deteksi_dan_crop_plat(frame):
    # Konversi ke grayscale untuk deteksi mudah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi tepi (edge detection) untuk temukan kotak plat
    edges = cv2.Canny(gray, 50, 150)
    
    # Cari kontur (bentuk kotak)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter kontur yang berbentuk persegi panjang (seperti plat)
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), closed=True)
        if len(approx) == 4:  # Persegi panjang punya 4 sudut
            x, y, w, h = cv2.boundingRect(contour)
            # Asumsi plat punya aspek ratio tertentu (lebar > tinggi, misal 2:1)
            if w > 100 and h > 30 and w / h > 1.5:
                # Crop bagian plat
                plat_crop = frame[y:y+h, x:x+w]
                return plat_crop, (x, y, w, h)
    return None, None

# Mulai kamera (0 untuk webcam default, ganti dengan path video CCTV jika ada)
cap = cv2.VideoCapture('rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0')  # Ganti 0 dengan 'rtsp://ip_cctv' untuk CCTV real

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi plat
    plat_crop, bbox = deteksi_dan_crop_plat(frame)
    
    if plat_crop is not None:
        # Simpan gambar crop ke folder static
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nama_file = f"gambarplat/plat_{timestamp}.jpg"
        cv2.imwrite(nama_file, plat_crop)
        
        # Baca teks plat dengan OCR
        teks_plat = pytesseract.image_to_string(plat_crop, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        teks_plat = teks_plat.strip().upper()  # Bersihkan teks

        import mysql.connector  # Tambah di atas

        # Fungsi simpan ke DB
        def simpan_ke_db(plat, foto_path):
            try:
                conn = mysql.connector.connect(
                    host='localhost',
                    user='root',  # Default XAMPP
                    password='',  # Kosong di XAMPP
                    database='deteksi_plat_db'
                )
                cursor = conn.cursor()
                query = "INSERT INTO tb_kendaraan (Plat_Nomor, Foto_Plat) VALUES (%s, %s)"
                cursor.execute(query, (plat, foto_path))
                conn.commit()
                print("Data disimpan ke database!")
            except Exception as e:
                print(f"Error DB: {e}")
            finally:
                if conn.is_connected():
                    cursor.close()
                    conn.close()

        # Panggil fungsi ini setelah deteksi
        if plat_crop is not None:
            # ... (kode sebelumnya)
            simpan_ke_db(teks_plat, nama_file)
                
        print(f"Plat terdeteksi: {teks_plat}")
        print(f"Gambar disimpan: {nama_file}")
        
        # Gambar kotak di frame asli
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, teks_plat, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Tampilkan frame
    cv2.imshow('Deteksi Plat Nomor', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()