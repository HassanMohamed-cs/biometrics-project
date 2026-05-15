import cv2
import dlib
import numpy as np
from skimage.feature import hog

from hema import load_dataset, preprocess_hist_eq


# ── dlib models (Method 1) ────────────────────────────────────────────────────
face_detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1 — dlib ResNet-128 descriptor
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dlib_features(img: np.ndarray) -> np.ndarray | None:
    # dlib needs uint8 RGB
    img_u8  = (img * 255).clip(0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    faces = face_detector(img_rgb, 1)
    if len(faces) == 0:
        return None

    face  = faces[0]
    shape = predictor(img_rgb, face)
    descriptor = face_rec_model.compute_face_descriptor(img_rgb, shape)
    return np.array(descriptor)          # (128,)


def feature_matrix_dlib(dataset_path: str = "splited_dataset") -> np.ndarray:
    data, labels = load_dataset(dataset_path)
    data = [preprocess_hist_eq(img) for img in data]

    features = []
    zero_count = 0
    for img in data:
        if img is None:
            feat = np.zeros(128)
            zero_count += 1
        else:
            feat = extract_dlib_features(img)
            if feat is None:
                feat = np.zeros(128)
                zero_count += 1
        features.append(feat)

    FM = np.array(features).T          # (128, N)
    print(f"[dlib] FM shape: {FM.shape}  |  zero vectors: {zero_count}")
    np.save("FM_DLIB.npy", FM)
    return FM


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2 — HOG descriptor
# ═══════════════════════════════════════════════════════════════════════════════

# HOG hyper-parameters (tuned for 128×128 face patches)
_HOG_ORIENTATIONS  = 9
_HOG_PIXELS_CELL   = (8, 8)
_HOG_CELLS_BLOCK   = (2, 2)

def extract_hog_features(img: np.ndarray) -> np.ndarray:
    descriptor = hog(
        img,
        orientations=_HOG_ORIENTATIONS,
        pixels_per_cell=_HOG_PIXELS_CELL,
        cells_per_block=_HOG_CELLS_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,   # gamma-correction pre-step
    )
    return descriptor


def feature_matrix_hog(dataset_path: str = "splited_dataset") -> np.ndarray:
    data, labels = load_dataset(dataset_path)
    data = [preprocess_hist_eq(img) for img in data]

    features = []
    for img in data:
        if img is None:
            # We need a placeholder; compute dummy HOG dimension once
            if len(features) == 0:
                dummy = np.zeros((128, 128), dtype=np.float32)
                feat = extract_hog_features(dummy)
            else:
                feat = np.zeros_like(features[0])
        else:
            feat = extract_hog_features(img)
        features.append(feat)

    FM = np.array(features).T          # (D_hog, N)
    print(f"[HOG] FM shape: {FM.shape}")
    np.save("FM_HOG.npy", FM)
    return FM

