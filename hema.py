import os
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = (128, 128)


def detect_and_preprocess(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=1,
        minSize=(50, 50)
    )

    # لو مفيش وش → تجاهل الصورة
    if len(faces) == 0:
        return None
    # خد أكبر وجه (الأكثر احتمالًا يكون الأساسي)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = gray[y:y+h, x:x+w]

    face = cv.resize(face, IMG_SIZE)

    # normalization بسيط بدل الهستوجرام العنيف
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

            if face is None:
                continue  # skip images without faces

            data.append(face)
            labels.append(label)

    return np.array(data), np.array(labels)


if __name__ == "__main__":

    dataset_path = "dst"

    data, labels = load_and_preprocess(dataset_path)

    print("Total images:", len(data))
    print("Sample label:", labels[0])
    print("Shape:", data[0].shape)

    display = (data[11]*255).astype(np.uint8)
    display = cv.resize(display, (600, 600))

    cv.imshow("Face Preprocessed", display)

    cv.waitKey(0)
    cv.destroyAllWindows()