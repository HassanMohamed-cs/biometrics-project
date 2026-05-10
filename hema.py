import os
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = (128, 128)

# CLAHE object
clahe = cv.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


def detect_and_preprocess(img):

    # =========================
    # BGR -> YCrCb
    # =========================
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # split channels
    y, cr, cb = cv.split(ycrcb)

    # =========================
    # Enhance illumination
    # =========================
    y = clahe.apply(y)

    # =========================
    # Face Detection
    # =========================
    faces = face_cascade.detectMultiScale(
        y,
        scaleFactor=1.0025,
        minNeighbors=4,
        minSize=(50, 50)
    )

    # no face found
    if len(faces) == 0:
        return None

    # largest face
    x, y1, w, h = max(faces, key=lambda f: f[2] * f[3])

    # crop face from enhanced Y channel
    face = y[y1:y1+h, x:x+w]

    # resize
    face = cv.resize(face, IMG_SIZE)

    # normalize
    face = face.astype(np.float32) / 255.0

    return face


def load_and_preprocess(path):

    data = []
    labels = []

    persons = sorted(
        os.listdir(path),
        key=lambda x: int(x.replace("person", ""))
    )

    for label, person in enumerate(persons):

        if not person.startswith("person"):
            continue

        person_path = os.path.join(path, person)

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            img = cv.imread(img_path)

            if img is None:
                continue

            face = detect_and_preprocess(img)

            # skip if no face detected
            if face is None:
                continue

            data.append(face)
            labels.append(label)

    return np.array(data), np.array(labels)


if __name__ == "__main__":

    dataset_path = "dst"

    data, labels = load_and_preprocess(dataset_path)

    print("Total images:", len(data))
    print("Sample label:", labels[0])
    print("Shape:", data[0].shape)

    # display sample
    display = (data[11]).astype(np.uint8)

    display = cv.resize(display, (600, 600))

    cv.imshow("Face Preprocessed", display)

    cv.waitKey(0)
    cv.destroyAllWindows()