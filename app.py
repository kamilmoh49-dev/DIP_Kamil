import os
import base64
import shutil
import atexit
import signal
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from functools import wraps

# ============ KONFIGURASI ============
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Konfigurasi folder
UPLOAD_FOLDER = 'uploads'
ORIGINAL_FOLDER = 'originals'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ORIGINAL_FOLDER'] = ORIGINAL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)

# ============ FUNGSI HELPER ============

def allowed_file(filename):
    """Cek apakah ekstensi file diperbolehkan"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def get_image_info(image_path):
    """Mendapatkan informasi detail dari gambar"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        tinggi, lebar = img.shape[:2]
        resolusi = f"{lebar} x {tinggi} piksel"
        ukuran_file = f"{os.path.getsize(image_path) / 1024:.2f} KB"
        
        if len(img.shape) == 3:
            channels = img.shape[2]
            tipe_warna = "Color (BGR)"
        else:
            channels = 1
            tipe_warna = "Grayscale"
        
        # Ambil nilai piksel tengah
        tengah_y, tengah_x = tinggi // 2, lebar // 2
        nilai_piksel_tengah = img[tengah_y, tengah_x]
        
        if channels == 3:
            b, g, r = nilai_piksel_tengah
            info_piksel = f"Piksel Tengah ({tengah_x}, {tengah_y}) → B:{b}, G:{g}, R:{r}"
        else:
            info_piksel = f"Piksel Tengah ({tengah_x}, {tengah_y}) → Intensitas: {nilai_piksel_tengah}"
        
        return {
            'resolusi': resolusi,
            'ukuran': ukuran_file,
            'channels': f"{channels} ({tipe_warna})",
            'piksel': info_piksel,
            'width': lebar,
            'height': tinggi,
            'dimensi': resolusi,
            'format': image_path.rsplit('.', 1)[1].upper() + ' Image'
        }
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return None

def image_to_base64(image):
    """Konversi gambar OpenCV ke base64 URL"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def save_image_result(img, filename):
    """Simpan hasil operasi ke folder uploads"""
    try:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(output_path, img)
        return output_path
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def load_original_image(filename):
    """Load gambar original"""
    original_path = os.path.join(app.config['ORIGINAL_FOLDER'], filename)
    img = cv2.imread(original_path)
    if img is None:
        raise ValueError(f"Gagal membaca file: {filename}")
    return img

def load_current_image(filename):
    """Load gambar saat ini (dari uploads)"""
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(upload_path)
    if img is None:
        raise ValueError(f"Gagal membaca file: {filename}")
    return img

def validate_image_request(func):
    """Decorator untuk validasi request gambar"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON data'}), 400
            
            filename = data.get('filename')
            if not filename:
                return jsonify({'error': 'Filename tidak ditemukan'}), 400
            
            # Validasi filename
            if not secure_filename(filename) or '..' in filename:
                return jsonify({'error': 'Invalid filename'}), 400
            
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return jsonify({'error': str(e)}), 400
    return wrapper

# ============ AUTO CLEANUP ============

def cleanup_folders():
    """Hapus semua file temporary saat server ditutup"""
    logger.info("🧹 Membersihkan folder temporary...")
    try:
        for folder in [UPLOAD_FOLDER, ORIGINAL_FOLDER]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logger.info(f"✅ Folder '{folder}' dan isinya dihapus")
        logger.info("✅ Semua file temporary berhasil dihapus!")
    except Exception as e:
        logger.error(f"❌ Error saat membersihkan: {e}")

def signal_handler(sig, frame):
    """Handler untuk SIGINT (Ctrl+C)"""
    logger.info("\n🛑 Server dimatikan...")
    cleanup_folders()
    sys.exit(0)

# Daftarkan handler
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_folders)

# ============ ROUTE UTAMA ============

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    """Endpoint untuk mengirim file gambar yang telah diupload"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============ ROUTE UPLOAD ============

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint untuk upload file via AJAX"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'File kosong'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format file tidak didukung'}), 400
        
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Backup ke originals
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], filename)
        shutil.copy2(upload_path, original_path)
        
        # Baca gambar
        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({'error': 'Gagal membaca gambar'}), 400
        
        # Konversi ke base64
        img_base64 = image_to_base64(img)
        if not img_base64:
            return jsonify({'error': 'Gagal konversi gambar'}), 400
        
        # Dapatkan info
        info = get_image_info(upload_path)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image_data': img_base64,
            'info': {
                'dimensi': info.get('resolusi', ''),
                'format': info.get('format', ''),
                'ukuran': info.get('ukuran', ''),
                'channels': info.get('channels', ''),
                'piksel': info.get('piksel', '')
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

# ============ ROUTE RESET ============

@app.route('/reset_image', methods=['POST'])
@validate_image_request
def reset_image():
    """Reset gambar ke citra asli"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(original_path):
            return jsonify({'error': 'File original tidak ditemukan'}), 404
        
        # Copy original ke uploads
        shutil.copy2(original_path, upload_path)
        
        # Load image untuk return
        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({'error': 'Gagal membaca file'}), 400
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(img),
            'filename': filename,
            'message': 'Berhasil reset ke citra asli'
        })
        
    except Exception as e:
        logger.error(f"Error in reset: {e}")
        return jsonify({'error': str(e)}), 500

# ============ ROUTE OPERASI CITRA ============

@app.route('/convert/grayscale', methods=['POST'])
@validate_image_request
def convert_grayscale():
    """Konversi citra RGB ke Grayscale"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        intensity = data.get('intensity', 100)
        
        # Validasi intensity
        intensity = max(0, min(100, int(intensity)))
        
        # Load original
        img = load_original_image(filename)
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Blend dengan original berdasarkan intensity
        factor = intensity / 100.0
        result = (img * (1 - factor) + gray_3channel * factor).astype(np.uint8)
        
        # Simpan hasil
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Berhasil konversi ke Grayscale: {intensity}%'
        })
        
    except Exception as e:
        logger.error(f"Error in grayscale conversion: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert/binary', methods=['POST'])
@validate_image_request
def convert_binary():
    """Konversi citra ke Biner"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        threshold_value = data.get('threshold', 127)
        
        # Validasi threshold
        threshold_value = max(0, min(255, int(threshold_value)))
        
        # Load original
        img = load_original_image(filename)
        
        # Konversi ke grayscale dulu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        binary_3channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Simpan hasil
        save_image_result(binary_3channel, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(binary_3channel),
            'filename': filename,
            'message': f'Berhasil konversi ke Biner (threshold: {threshold_value})'
        })
        
    except Exception as e:
        logger.error(f"Error in binary conversion: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/apply/brightness', methods=['POST'])
@validate_image_request
def apply_brightness():
    """Apply brightness sesuai kontrol slider"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        brightness = data.get('value', 50)  # 0-100
        
        # Konversi dari 0-100 ke -255 - 255
        brightness_value = ((brightness - 50) / 50) * 255
        
        # Load original
        img = load_original_image(filename)
        
        # Apply brightness
        img_float = img.astype(np.float32)
        img_float += brightness_value
        img_float = np.clip(img_float, 0, 255)
        result = img_float.astype(np.uint8)
        
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Brightness: {brightness}%'
        })
        
    except Exception as e:
        logger.error(f"Brightness error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/apply/zoom', methods=['POST'])
@validate_image_request
def apply_zoom():
    """Apply zoom dengan resize"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        zoom = data.get('value', 100)  # 0-200
        
        # Validasi zoom
        zoom = max(0, min(200, float(zoom)))
        
        # Load original
        img = load_original_image(filename)
        
        # Zoom dengan resize
        scale = zoom / 100
        height, width = img.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        result = cv2.resize(img, (new_width, new_height))
        
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Zoom: {zoom}%'
        })
        
    except Exception as e:
        logger.error(f"Zoom error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/apply/rotate', methods=['POST'])
@validate_image_request
def apply_rotation():
    """Apply rotasi sesuai slider"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        angle = data.get('value', 0)  # -180 - 180
        
        # Validasi angle
        angle = max(-180, min(180, float(angle)))
        
        # Load original
        img = load_original_image(filename)
        
        # Rotasi
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Rotasi: {angle}°'
        })
        
    except Exception as e:
        logger.error(f"Rotation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/apply/flip', methods=['POST'])
@validate_image_request
def apply_flip():
    """Apply flip horizontal/vertical"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        direction = data.get('direction', 'h')  # 'h' atau 'v'
        
        # Validasi direction
        if direction not in ['h', 'v']:
            return jsonify({'error': 'Direction harus h atau v'}), 400
        
        # Load original
        img = load_original_image(filename)
        
        # Flip
        if direction == 'h':
            result = cv2.flip(img, 1)  # Horizontal
        else:
            result = cv2.flip(img, 0)  # Vertical
        
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Flip {direction} berhasil'
        })
        
    except Exception as e:
        logger.error(f"Flip error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/apply/all', methods=['POST'])
@validate_image_request
def apply_all_changes():
    """Terapkan semua perubahan sekaligus"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        # Ambil semua parameter
        brightness = data.get('brightness', 50)
        rotation = data.get('rotation', 0)
        zoom = data.get('zoom', 100)
        flip_h = data.get('flip_h', False)
        flip_v = data.get('flip_v', False)
        grayscale = data.get('grayscale', False)
        binary = data.get('binary', False)
        threshold = data.get('threshold', 128)
        
        # Load original
        img = load_original_image(filename)
        
        # Apply brightness
        if brightness != 50:
            brightness_value = ((brightness - 50) / 50) * 255
            img_float = img.astype(np.float32)
            img_float += brightness_value
            img_float = np.clip(img_float, 0, 255)
            img = img_float.astype(np.uint8)
        
        # Apply grayscale
        if grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply binary
        if binary:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        
        # Apply rotation
        if rotation != 0:
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        # Apply flip
        if flip_h:
            img = cv2.flip(img, 1)
        if flip_v:
            img = cv2.flip(img, 0)
        
        # Apply zoom
        if zoom != 100:
            scale = zoom / 100
            height, width = img.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        save_image_result(img, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(img),
            'filename': filename,
            'message': 'Semua perubahan berhasil diterapkan'
        })
        
    except Exception as e:
        logger.error(f"Apply all error: {e}")
        return jsonify({'error': str(e)}), 500

# ============ ROUTE OPERASI LANJUTAN ============

@app.route('/convert/negative', methods=['POST'])
@validate_image_request
def convert_negative():
    """Konversi citra ke negatif"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        intensity = data.get('intensity', 100)
        
        # Validasi intensity
        intensity = max(0, min(100, int(intensity)))
        
        # Load original
        img = load_original_image(filename)
        
        # Buat negatif
        negative = 255 - img
        
        # Blend berdasarkan intensity
        factor = intensity / 100.0
        result = (img * (1 - factor) + negative * factor).astype(np.uint8)
        
        # Simpan hasil
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Berhasil konversi ke Negatif: {intensity}%'
        })
        
    except Exception as e:
        logger.error(f"Error in negative conversion: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert/edge', methods=['POST'])
@validate_image_request
def convert_edge():
    """Edge detection dengan metode (Sobel, Prewitt, Canny)"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        method = data.get('method', 'canny')
        threshold1 = data.get('threshold1', 100)
        threshold2 = data.get('threshold2', 200)
        
        # Validasi threshold
        threshold1 = max(0, min(255, int(threshold1)))
        threshold2 = max(0, min(255, int(threshold2)))
        
        # Load original as grayscale
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], filename)
        img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Gagal membaca file'}), 400
        
        # Edge detection berdasarkan metode
        if method == 'canny':
            edges = cv2.Canny(img, threshold1, threshold2)
        elif method == 'sobel':
            # Sobel edge detection
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
            # Terapkan threshold1 sebagai ambang batas tepi (dinamis dari slider)
        elif method == 'prewitt':
            kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            prewittx = cv2.filter2D(img, cv2.CV_64F, kernelx)
            prewitty = cv2.filter2D(img, cv2.CV_64F, kernely)
            edges = np.sqrt(prewittx**2 + prewitty**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
            # Terapkan threshold1 sebagai ambang batas tepi (dinamis dari slider)
        else:
            return jsonify({'error': f'Metode {method} tidak didukung'}), 400
        
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Simpan hasil
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Berhasil edge detection ({method})'
        })
        
    except Exception as e:
        logger.error(f"Error in edge detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert/sharpen', methods=['POST'])
@validate_image_request
def convert_sharpen():
    """Sharpening image"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        strength = data.get('strength', 1.0)
        
        # Validasi strength
        strength = max(0.1, min(5.0, float(strength)))
        
        # Load original
        img = load_original_image(filename)
        
        # Kernel sharpen
        kernel = np.array([
            [0, -1, 0],
            [-1, 4 + strength, -1],
            [0, -1, 0]
        ])
        
        result = cv2.filter2D(img, -1, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Simpan hasil
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': f'Berhasil sharpen (strength: {strength})'
        })
        
    except Exception as e:
        logger.error(f"Error in sharpening: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert/fourier', methods=['POST'])
@validate_image_request
def convert_fourier():
    """Transformasi Fourier"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        # Load original as grayscale
        original_path = os.path.join(app.config['ORIGINAL_FOLDER'], filename)
        img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Gagal membaca file'}), 400
        
        # Fourier Transform
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Normalisasi ke 0-255
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
        
        # Konversi ke BGR untuk display
        result = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2BGR)
        
        # Simpan hasil
        save_image_result(result, filename)
        
        return jsonify({
            'success': True,
            'image_data': image_to_base64(result),
            'filename': filename,
            'message': 'Berhasil transformasi Fourier'
        })
        
    except Exception as e:
        logger.error(f"Error in Fourier transform: {e}")
        return jsonify({'error': str(e)}), 500

# ============ ROUTE HISTOGRAM ============

@app.route('/get_histogram', methods=['POST'])
@validate_image_request
def get_histogram():
    """Mendapatkan data histogram dari gambar"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        # Cek file di folder uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File tidak ditemukan'}), 404
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Gagal membaca file gambar'}), 400
        
        # Hitung histogram untuk setiap channel (B, G, R)
        hist_data = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist_data.append(hist.flatten().tolist())
        
        return jsonify({
            'success': True,
            'histogram': hist_data,
            'message': 'Histogram berhasil diambil'
        })
        
    except Exception as e:
        logger.error(f"Histogram error: {e}")
        return jsonify({'error': str(e)}), 500

# ============ ROUTE GET INFO ============

@app.route('/get_image_info', methods=['POST'])
@validate_image_request
def get_image_info_api():
    """API untuk mendapatkan info gambar"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(upload_path):
            return jsonify({'error': 'File tidak ditemukan'}), 404
        
        info = get_image_info(upload_path)
        if not info:
            return jsonify({'error': 'Gagal membaca info gambar'}), 400
        
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return jsonify({'error': str(e)}), 500

# ============ MAIN ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)