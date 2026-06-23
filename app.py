import os
import base64
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # 1. Baca file gambar langsung dari memory buffer (Aman untuk Vercel Serverless)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is not None:
                # 2. Ambil Informasi Matriks Citra Asli
                tinggi, lebar = img.shape[0], img.shape[1]
                resolusi = f"{lebar} x {tinggi} piksel"
                ukuran_file = f"{len(file_bytes) / 1024:.2f} KB"
                
                if len(img.shape) == 3:
                    channels = img.shape[2]
                    tipe_warna = "Color (BGR)"
                else:
                    channels = 1
                    tipe_warna = "Grayscale"
                
                # Mengambil sampel nilai piksel di koordinat tengah gambar
                tengah_y, tengah_x = tinggi // 2, lebar // 2
                nilai_piksel_tengah = img[tengah_y, tengah_x]
                
                if channels == 3:
                    b, g, r = nilai_piksel_tengah
                    info_piksel = f"Piksel Tengah ({tengah_x}, {tengah_y}) -> B:{b}, G:{g}, R:{r}"
                else:
                    info_piksel = f"Piksel Tengah ({tengah_x}, {tengah_y}) -> Intensitas: {nilai_piksel_tengah}"

                # 3. Konversi matriks gambar ke format Base64 (Data URL) agar bisa dikirim langsung ke HTML
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                img_data_url = f"data:image/jpeg;base64,{img_base64}"

                # Kirim data ke frontend
                info_citra = {
                    'img_data': img_data_url,
                    'filename': file.filename,
                    'ukuran': ukuran_file,
                    'resolusi': resolusi,
                    'channels': f"{channels} ({tipe_warna})",
                    'piksel': info_piksel
                }
                
                return render_template('index.html', info=info_citra)
                
    return render_template('index.html', info=None)

# Kode ini tetap dipertahankan untuk running lokal, Vercel akan otomatis membaca objek 'app'
if __name__ == '__main__':
    app.run(debug=True)