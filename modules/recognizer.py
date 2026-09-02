# ==============================================================================
# FILE: modules/recognizer.py
# FUNGSI: Modul Pengenalan Wajah (Face Recognition) Berbasis SFace ONNX
# PENJELASAN TEORI AI:
# Jika detektor wajah hanya tahu "ADA WAJAH", modul ini bertugas menjawab: "SIAPA INI?"
# Komputer TIDAK mengenali wajah dengan membandingkan pixel warna gambar (karena jika
# cahaya berubah sedikit saja, nilai pixel akan berubah total).
# 
# Cara kerja Face Recognition modern:
# 1. FACE ALIGNMENT (Perataan Wajah):
#    Memutar dan menyejajarkan posisi wajah berdasarkan titik mata dan hidung, lalu
#    memotongnya (crop) menjadi ukuran standar 112x112 pixel.
# 2. FEATURE EXTRACTION (Ekstraksi Ciri):
#    Gambar wajah 112x112 dimasukkan ke Jaringan Saraf Tiruan Dalam (Deep Neural Network SFace).
#    Model ini memetakan ciri-ciri geometris unik wajah (jarak mata, lekukan hidung, bentuk rahang)
#    menjadi deretan 128 ANGKA saja. Deretan angka ini disebut "FACE EMBEDDING" (sidik jari digital wajah).
# 3. VECTOR MATCHING (Pencocokan Vektor):
#    Dua wajah dibandingkan dengan menghitung sudut kosinus antar vektornya (COSINE SIMILARITY).
#    Semakin searah vektornya (mendekati 1.0), semakin yakin AI bahwa kedua foto adalah orang yang sama!
# ==============================================================================

import cv2                         # Library OpenCV
import numpy as np                 # Library komputasi matriks/vektor
import sys
from pathlib import Path           # Library path direktori
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Dict, Tuple, Optional
from config import MODELS_DIR, SIMILARITY_THRESHOLD # Mengambil konfigurasi threshold

class FaceRecognizer:
    """
    Kelas untuk mengekstrak vektor ciri wajah (feature embedding)
    dan mengukur tingkat kemiripan antar dua wajah.
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        KONSTRUKTOR:
        Memuat model SFace (face_recognition_sface_2021dec.onnx).
        """
        if model_path is None:
            model_path = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")
            
        self.model_path = model_path
        self.recognizer = None
        
        if Path(model_path).exists():
            self._init_recognizer()

    def _init_recognizer(self):
        """
        Inisialisasi OpenCV FaceRecognizerSF:
        SF singkatan dari 'SFace' (arsitektur efisien untuk face recognition di edge device / CPU).
        """
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def extract_feature(self, frame: np.ndarray, face_raw: np.ndarray) -> Optional[np.ndarray]:
        """
        LOGIKA EKSTRAKSI FITUR:
        Menerima frame penuh dan array 15 elemen dari detektor YuNet.
        
        Langkah 1: alignCrop()
        Secara otomatis melakukan transformasi afin (affine transformation) untuk
        memutar wajah agar tegak lurus sempurna dan memotongnya ke dimensi 112x112 pixel.
        
        Langkah 2: feature()
        Memproses crop wajah tersebut ke dalam Deep Neural Network SFace untuk
        menghasilkan vektor embedding berdimensi (1, 128).
        """
        if self.recognizer is None:
            if Path(self.model_path).exists():
                self._init_recognizer()
            else:
                return None
                
        # 1. Meluruskan posisi wajah berdasarkan landmark mata & hidung
        aligned_face = self.recognizer.alignCrop(frame, face_raw)
        
        # 2. Ekstrak vektor embedding (128 angka float32)
        feature = self.recognizer.feature(aligned_face)
        return feature

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        LOGIKA PENGUKURAN KEMIRIPAN (COSINE SIMILARITY):
        Rumus Matematika:
            similarity = (A . B) / (||A|| * ||B||)
        Keterangan:
        - Jika score >= 0.55 : Sangat mirip (kemungkinan besar orang yang sama).
        - Jika score < 0.30  : Tidak mirip (orang berbeda).
        """
        if self.recognizer is None:
            if Path(self.model_path).exists():
                self._init_recognizer()
            else:
                # Fallback jika model belum termuat: hitung manual dengan rumus numpy dot product
                f1 = feat1.flatten()
                f2 = feat2.flatten()
                dot_product = np.dot(f1, f2)
                norm = np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-6
                return float(dot_product / norm)
                
        # Menggunakan fungsi bawaan C++ dari OpenCV yang sudah dioptimasi kecepatan komputasinya
        score = self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    def identify(self, feature: np.ndarray, known_users: List[Dict], threshold: float = SIMILARITY_THRESHOLD) -> Tuple[Optional[Dict], float]:
        """
        LOGIKA IDENTIFIKASI (1-to-N MATCHING):
        Mencocokkan satu wajah yang terlihat di kamera terhadap 'N' orang yang ada di database.
        
        Cara kerja:
        1. Lakukan perulangan (loop) ke seluruh user di database.
        2. Hitung nilai kemiripan vektor wajah di kamera dengan vektor wajah user di DB.
        3. Catat siapa user dengan skor kemiripan tertinggi (best_user).
        4. Terakhir, periksa apakah skor tertinggi tersebut melampaui 'threshold' (0.55).
           - Jika YA: Wajah dikenali sebagai user tersebut!
           - Jika TIDAK: Anggap sebagai orang tak dikenal ("UNKNOWN").
        """
        if not known_users or feature is None:
            return None, 0.0
            
        best_score = -1.0
        best_user = None
        
        # Loop pencarian vektor terdekat (Nearest Neighbor)
        for user in known_users:
            stored_emb = user.get("embedding")
            if stored_emb is None:
                continue
                
            score = self.compute_similarity(feature, stored_emb)
            if score > best_score:
                best_score = score
                best_user = user
                
        # Validasi batas minimal keyakinan
        if best_score >= threshold:
            return best_user, best_score
            
        return None, best_score
