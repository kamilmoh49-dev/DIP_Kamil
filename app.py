import os
import base64
import shutil
import atexit
import signal
import sys
import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi upload folder
UPLOAD_FOLDER = 'uploads'
ORIGINAL_FOLDER = 'originals'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ORIGINAL_FOLDER'] = ORIGINAL_FOLDER

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Cek apakah ekstensi file diperbolehkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============ AUTO DELETE TOTAL SAAT SERVER DITUTUP ============

def cleanup_folders():
    """Hapus SEMUA file dan folder uploads & originals saat server ditutup"""
    print("\n🧹 Membersihkan folder temporary...")
    try:
        # Hapus folder uploads dan semua isinya
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            print(f"✅ Folder '{UPLOAD_FOLDER}' dan isinya dihapus")
        
        # Hapus folder originals dan semua isinya
        if os.path.exists(ORIGINAL_FOLDER):
            shutil.rmtree(ORIGINAL_FOLDER)
            print(f"✅ Folder '{ORIGINAL_FOLDER}' dan isinya dihapus")
        
        print("✅ Semua file dan folder temporary berhasil dihapus!")
    except Exception as e:
        print(f"❌ Error saat membersihkan: {e}")

# Jalankan cleanup di awal (bersihin file lama)
cleanup_folders()

# Buat ulang folder setelah cleanup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)

def signal_handler(sig, frame):
    """Handler untuk SIGINT (Ctrl+C)"""
    print("\n🛑 Server dimatikan...")
    cleanup_folders()
    sys.exit(0)

# Daftarkan handler
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_folders)

# ============ ROUTE ============

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    """Endpoint untuk mengirim file gambar yang telah diupload"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/convert/grayscale', methods=['POST'])
def convert_grayscale():
    """Konversi citra RGB ke Grayscale dengan intensity manual"""
    try:
        data = request.get_json()
        current_filename = data.get('filename', 'image.jpg')
        intensity = data.get('intensity', 100)
        
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], current_filename)
        img = cv2.imread(original_path)
        
        if img is None:
            return jsonify({'error': 'Gagal membaca file original'}), 400
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        factor = intensity / 100.0
        img_result = (img * (1 - factor) + gray_3channel * factor).astype(np.uint8)
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
        cv2.imwrite(output_path, img_result)
        
        _, buffer = cv2.imencode('.jpg', img_result)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'success': True,
            'image_data': img_data_url,
            'filename': current_filename,
            'message': f'Berhasil konversi ke Grayscale: {intensity}%'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/convert/binary', methods=['POST'])
def convert_binary():
    """Konversi citra RGB ke Biner dengan threshold manual"""
    try:
        data = request.get_json()
        current_filename = data.get('filename', 'image.jpg')
        threshold_value = data.get('threshold', 127)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
        img = cv2.imread(filepath)
        
        if img is None:
            return jsonify({'error': 'Gagal membaca file gambar'}), 400
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        binary_3channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
        cv2.imwrite(output_path, binary)
        
        _, buffer = cv2.imencode('.jpg', binary_3channel)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'success': True,
            'image_data': img_data_url,
            'filename': current_filename,
            'message': f'Berhasil konversi ke Biner dengan threshold {threshold_value}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/convert/brightness', methods=['POST'])
def convert_brightness():
    """Adjust brightness (pencerahan) gambar - selalu dari original"""
    try:
        data = request.get_json()
        current_filename = data.get('filename', 'image.jpg')
        brightness_value = data.get('brightness', 0)
        
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], current_filename)
        img = cv2.imread(original_path)
        
        if img is None:
            return jsonify({'error': 'Gagal membaca file original'}), 400
        
        img_float = img.astype(np.float32)
        img_float += brightness_value
        img_float = np.clip(img_float, 0, 255)
        img_result = img_float.astype(np.uint8)
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
        cv2.imwrite(output_path, img_result)
        
        _, buffer = cv2.imencode('.jpg', img_result)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'success': True,
            'image_data': img_data_url,
            'filename': current_filename,
            'message': f'Berhasil adjust brightness: {brightness_value}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/convert/negative', methods=['POST'])
def convert_negative():
    """Konversi citra ke negatif dengan intensity manual - selalu dari original"""
    try:
        data = request.get_json()
        current_filename = data.get('filename', 'image.jpg')
        intensity = data.get('intensity', 100)
        
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], current_filename)
        img = cv2.imread(original_path)
        
        if img is None:
            return jsonify({'error': 'Gagal membaca file original'}), 400
        
        img_negative_full = 255 - img
        factor = intensity / 100.0
        img_result = (img * (1 - factor) + img_negative_full * factor).astype(np.uint8)
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
        cv2.imwrite(output_path, img_result)
        
        _, buffer = cv2.imencode('.jpg', img_result)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'success': True,
            'image_data': img_data_url,
            'filename': current_filename,
            'message': f'Berhasil konversi ke Negatif: {intensity}%'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_image', methods=['POST'])
def reset_image():
    """Reset gambar ke citra asli"""
    try:
        data = request.get_json()
        current_filename = data.get('filename', 'image.jpg')
        
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], current_filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], current_filename)
        
        if not os.path.exists(original_path):
            return jsonify({'error': 'File original tidak ditemukan'}), 400
        
        shutil.copy2(original_path, upload_path)
        
        img = cv2.imread(upload_path)
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'success': True,
            'image_data': img_data_url,
            'filename': current_filename,
            'message': 'Berhasil reset ke citra asli'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ ROUTE UTAMA ============

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            original_path = os.path.join(app.config['ORIGINAL_FOLDER'], filename)
            shutil.copy2(upload_path, original_path)
            
            img = cv2.imread(upload_path)
            
            if img is not None:
                tinggi, lebar = img.shape[0], img.shape[1]
                resolusi = f"{lebar} x {tinggi} piksel"
                ukuran_file = f"{os.path.getsize(upload_path) / 1024:.2f} KB"
                
                if len(img.shape) == 3:
                    channels = img.shape[2]
                    tipe_warna = "Color (BGR)"
                else:
                    channels = 1
                    tipe_warna = "Grayscale"
                
                tengah_y, tengah_x = tinggi // 2, lebar // 2
                nilai_piksel_tengah = img[tengah_y, tengah_x]
                
                if channels == 3:
                    b, g, r = nilai_piksel_tengah
                    info_piksel = f"Piksel Tengah ({tengah_x}, {tengah_y}) -> B:{b}, G:{g}, R:{r}"
                else:
                    info_piksel = f"Piksel Tengah ({tengah_x}, {tengah_y}) -> Intensitas: {nilai_piksel_tengah}"

                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                img_data_url = f"data:image/jpeg;base64,{img_base64}"

                info_citra = {
                    'img_data': img_data_url,
                    'filename': filename,
                    'ukuran': ukuran_file,
                    'resolusi': resolusi,
                    'channels': f"{channels} ({tipe_warna})",
                    'piksel': info_piksel
                }
                
                return render_template('index.html', info=info_citra)
                
    return render_template('index.html', info=None)

if __name__ == '__main__':
    # Untuk production (Vercel, Render, dll)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)