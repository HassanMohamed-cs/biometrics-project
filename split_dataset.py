import os
import shutil
from collections import defaultdict

source_folder = "./originalimages_part1"
destination_folder = "splited_dataset"

# نجمع الصور لكل شخص
people = defaultdict(list)

for img_name in os.listdir(source_folder):

    person_id = img_name.split("-")[0]

    people[person_id].append(img_name)

# تقسيم الصور
for person_id, images in people.items():

    # ترتيب الصور
    images.sort()

    # آخر صورة للـ identification
    identification_image = images[-1]

    # باقي الصور
    remaining_images = images[:-1]

    # 80% train
    train_count = round(len(remaining_images) * 0.8)

    train_images = remaining_images[:train_count]

    # 20% test
    test_images = remaining_images[train_count:]

    # إنشاء الفولدرات
    train_folder = os.path.join(destination_folder, f"person{person_id}", "train")
    test_folder = os.path.join(destination_folder, f"person{person_id}", "test")
    identification_folder = os.path.join(destination_folder, f"person{person_id}", "identification")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(identification_folder, exist_ok=True)

    # نسخ train
    for img in train_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(train_folder, img)
        )

    # نسخ test
    for img in test_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(test_folder, img)
        )

    # نسخ identification
    shutil.copy(
        os.path.join(source_folder, identification_image),
        os.path.join(identification_folder, identification_image)
    )

print("Done ✅")