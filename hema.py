import os
import cv2
import numpy as np


def load_dataset(path):
    data = []
    labels = []

    for label, person in enumerate(os.listdir(path)):


        if not person.startswith("person"):
            continue

        person_path = os.path.join(path, person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path, 0)
            if img is None:
                continue

            data.append(img)
            labels.append(label)

    return data, labels


def preprocess_hist_eq(img):
    img = cv2.resize(img, (100, 100))
    return cv2.equalizeHist(img)



def preprocess_gamma(img, gamma=0.5):
    img = cv2.resize(img, (100, 100))

    img = img / 255.0
    img = np.power(img, gamma)
    img = (img * 255).astype("uint8")

    return img



if __name__ == "__main__":

    dataset_path = "data"

    data, labels = load_dataset(dataset_path)

    print("Total images:", len(data))


    data_he = [preprocess_hist_eq(img) for img in data]
    data_gamma = [preprocess_gamma(img) for img in data]


    display_original = cv2.resize(data[0], (600, 600))
    display_he = cv2.resize(data_he[0], (600, 600))
    display_gamma = cv2.resize(data_gamma[0], (600, 600))


    cv2.imshow("Original", display_original)
    cv2.imshow("HistEq", display_he)
    cv2.imshow("Gamma", display_gamma)

    cv2.waitKey(0)
    cv2.destroyAllWindows()