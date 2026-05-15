import os
import cv2 as cv
import numpy as np

# ── Shared constants ──────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def _detect_face_gray(gray: np.ndarray):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.003, minNeighbors=4, minSize=(35, 35)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y : y + h, x : x + w]
    face = cv.resize(face, IMG_SIZE)
    return face

def preprocess_hist_eq(img: np.ndarray) -> np.ndarray | None:
    if img is None:
        return None

    # If the image is already grayscale (2-D), use it directly
    if img.ndim == 2:
        gray = img
    else:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    enhanced = clahe.apply(gray)
    face = _detect_face_gray(enhanced)
    if face is None:
        return None

    return face.astype(np.float32) / 255.0



def load_dataset(dataset_path: str = "data"):
    images, labels = [], []
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pgm"}

    for person_dir in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_dir)
        if not os.path.isdir(person_path):
            continue
        person_id = person_dir.replace("person", "")

        for root, _, files in os.walk(person_path):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in supported:
                    continue
                fpath = os.path.join(root, fname)
                img = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[WARNING] Could not read: {fpath}")
                    continue
                images.append(img)
                labels.append(person_id)

    print(f"[load_dataset] Loaded {len(images)} images "
          f"from {len(set(labels))} subjects.")
    return images, labels
