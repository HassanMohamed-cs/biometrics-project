import os
import shutil
from collections import defaultdict

SOURCE_FOLDER      = "originalimages_part1"
DESTINATION_FOLDER = "splited_dataset"
TRAIN_RATIO        = 0.8


def split_dataset(source: str = SOURCE_FOLDER, dest: str = DESTINATION_FOLDER) -> None:
    if not os.path.isdir(source):
        raise FileNotFoundError(f"Source folder not found: {source}")

    # Group images by person ID (prefix before the first '-')
    people: dict[str, list[str]] = defaultdict(list)
    for img_name in os.listdir(source):
        if not os.path.isfile(os.path.join(source, img_name)):
            continue
        person_id = img_name.split("-")[0]
        people[person_id].append(img_name)

    print(f"Found {len(people)} subjects.")

    for person_id, images in sorted(people.items()):
        images.sort()

        # Reserve last image for identification (closed-set probe)
        identification_image = images[-1]
        remaining            = images[:-1]

        train_count  = round(len(remaining) * TRAIN_RATIO)
        train_images = remaining[:train_count]
        test_images  = remaining[train_count:]

        train_folder          = os.path.join(dest, f"person{person_id}", "train")
        test_folder           = os.path.join(dest, f"person{person_id}", "test")
        identification_folder = os.path.join(dest, f"person{person_id}", "identification")

        for folder in (train_folder, test_folder, identification_folder):
            os.makedirs(folder, exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(source, img), os.path.join(train_folder, img))

        for img in test_images:
            shutil.copy(os.path.join(source, img), os.path.join(test_folder, img))

        shutil.copy(
            os.path.join(source, identification_image),
            os.path.join(identification_folder, identification_image),
        )

        print(
            f"  person{person_id}: "
            f"{len(train_images)} train | {len(test_images)} test | 1 identification"
        )

    print("Done ✅")


if __name__ == "__main__":
    split_dataset()
