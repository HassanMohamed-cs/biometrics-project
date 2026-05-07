import os
import shutil

source_folder = "./originalimages_part1"
destination_folder = "dst"

for img_name in os.listdir(source_folder):

    if not img_name.endswith(".jpg"):
        continue

    person_id = img_name.split("-")[0]

    person_folder = os.path.join(destination_folder, f"person{person_id}")

    os.makedirs(person_folder, exist_ok=True)

    src = os.path.join(source_folder, img_name)
    dst = os.path.join(person_folder, img_name)

    shutil.copy(src, dst)

print("Done ")