# ==============================================================================
# FILE: modules/detector.py
# FUNGSI: Modul Deteksi Wajah (Face Detection) Berbasis YuNet ONNX
# PENJELASAN TEORI AI:
# Deteksi wajah adalah tahap PERTAMA dalam sistem pengenalan wajah.
# Tujuannya HANYA untuk menjawab: "Di mana letak wajah di dalam gambar/video?"
# Kita menggunakan arsitektur 'YuNet', yaitu model Deep Learning convolutional
# ultra-ringan (~230 Kilobyte) resmi dari OpenCV Zoo.
# Keunggulan YuNet:
# 1. Mampu mendeteksi wajah dalam resolusi rendah/tinggi dengan sangat cepat (>30 FPS di CPU).
# 2. Tidak hanya memberikan kotak pembatas (Bounding Box), tetapi juga menghasilkan
#    5 titik 'Facial Landmarks' (Mata Kanan, Mata Kiri, Ujung Hidung, Sudut Mulut Kanan, Sudut Mulut Kiri).
# Titik-titik ini SANGAT PENTING agar di tahap berikutnya wajah bisa 'diluruskan' (alignment)
# jika posisi kepala orang miring ke samping.
# ==============================================================================

import cv2                         # Library Computer Vision paling populer di dunia
import numpy as np                 # Library manipulasi array data numerik
import sys
from pathlib import Path           # Library penanganan path file
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Tuple, Optional
from config import MODELS_DIR      # Mengambil lokasi folder penyimpanan model AI

class FaceDetector:
    """
    Kelas pembungkus (wrapper) untuk menginisialisasi dan menjalankan
    model YuNet Face Detector menggunakan modul DNN bawaan OpenCV.
    """
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.6, nms_threshold: float = 0.3):
        """
        KONSTRUKTOR:
        - model_path: Lokasi file .onnx model YuNet.
        - conf_threshold (Confidence Threshold): Nilai keyakinan minimal (0.0 s/d 1.0)
          agar sebuah objek dianggap benar-benar wajah manusia (default 0.6 atau 60%).
        - nms_threshold (Non-Maximum Suppression): Teknik untuk menghapus kotak deteksi
          yang saling tumpang-tindih di wajah yang sama, sehingga 1 wajah hanya punya 1 kotak rapi.
        """
        if model_path is None:
            # Gunakan file model default jika tidak ditentukan manual
            model_path = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
            
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.detector = None
        self._current_size = (320, 320) # Ukuran frame default awal (lebar, tinggi)
        
        # Inisialisasi model hanya jika file model sudah ada di harddisk
        if Path(model_path).exists():
            self._init_detector()

    def _init_detector(self):
        """
        LOGIKA INISIALISASI OPENCV YUNET:
        Fungsi cv2.FaceDetectorYN.create() memuat graf komputasi ONNX langsung
        ke memori menggunakan C++ engine OpenCV, sehingga tidak butuh framework
        tambahan seperti TensorFlow atau PyTorch yang berat.
        """
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",                     # Config kosong karena arsitektur sudah tertanam di dalam format ONNX
            input_size=self._current_size, # Resolusi gambar yang akan diproses
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,                    # Jumlah maksimal kandidat wajah yang dicek dalam 1 frame
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV, # Menggunakan backend OpenCV DNN
            target_id=cv2.dnn.DNN_TARGET_CPU       # Komputasi diarahkan ke CPU (hemat baterai/resource)
        )

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        LOGIKA DETEKSI WAJAH PADA FRAME:
        Menerima 1 frame gambar (berupa array pixel BGR dari kamera/file).
        
        Returns:
            List of dictionary berisi:
            - 'box': (x, y, lebar, tinggi) -> koordinat kotak wajah di gambar
            - 'score': persentase keyakinan AI bahwa ini adalah wajah (0.0 - 1.0)
            - 'raw': array numpy 15 elemen yang berisi data mentah landmarks
                     untuk dioper ke FaceRecognizerSF (tahap pengenalan).
        """
        # Cek apakah detector sudah siap
        if self.detector is None:
            if Path(self.model_path).exists():
                self._init_detector()
            else:
                return [] # Kembalikan list kosong jika file model belum tersedia
                
        # 1. Sinkronisasi Resolusi:
        # YuNet membutuhkan info dimensi gambar yang persis sama dengan frame kamera.
        h, w = frame.shape[:2]             # Mengambil tinggi (h) dan lebar (w) dari frame
        if (w, h) != self._current_size:
            self._current_size = (w, h)
            self.detector.setInputSize(self._current_size) # Update ukuran input model secara dinamis
            
        # 2. Proses Inferensi (Prediksi AI):
        # _, faces = detector.detect(frame)
        # Nilai balik 'faces' adalah array 2 dimensi dengan bentuk (N, 15),
        # di mana N adalah jumlah wajah yang ditemukan.
        # Tiap baris berisi 15 angka:
        # [x, y, w, h, mata_kanan_x, mata_kanan_y, mata_kiri_x, mata_kiri_y,
        #  hidung_x, hidung_y, sudut_mulut_kanan_x, sudut_mulut_kanan_y,
        #  sudut_mulut_kiri_x, sudut_mulut_kiri_y, confidence_score]
        _, faces = self.detector.detect(frame)
        results = []
        
        # 3. Parsing Data Wajah:
        if faces is not None:
            for face in faces:
                # 4 elemen pertama adalah bounding box [x, y, w, h]
                bbox = [int(v) for v in face[0:4]]
                
                # Validasi agar koordinat x dan y tidak negatif (keluar dari layar)
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                
                # Elemen terakhir (indeks -1) adalah skor confidence
                score = float(face[-1])
                
                results.append({
                    "box": tuple(bbox),
                    "score": score,
                    "raw": face            # Simpan seluruh 15 angka mentah untuk alignment SFace
                })
                
        return results
