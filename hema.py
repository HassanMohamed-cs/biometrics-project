import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

IMG_SIZE = (128, 128)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def detect_and_preprocess(img):
    # BGR -> YCrCb
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # split channels
    y, cr, cb = cv.split(ycrcb)

    # Enhance illumination
    y = clahe.apply(y)

    # Face Detection
    faces = face_cascade.detectMultiScale(y, scaleFactor=1.003, minNeighbors=4, minSize=(35, 35))

    if len(faces) == 0:
        return None

    # largest face
    x, y1, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = y[y1:y1+h, x:x+w]

    face = cv.resize(face, IMG_SIZE)

    face = face.astype(np.float32) / 255.0

    return face