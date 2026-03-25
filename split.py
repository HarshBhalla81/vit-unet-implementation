import os, shutil, random

SOURCE = "data/brain_tumor_dataset"
DEST = "split_data"

for split in ["train", "val"]:
    for cls in ["yes", "no"]:
        os.makedirs(f"{DEST}/{split}/{cls}", exist_ok=True)

for cls in ["yes", "no"]:
    imgs = os.listdir(f"{SOURCE}/{cls}")
    random.shuffle(imgs)

    split_idx = int(0.8 * len(imgs))

    for img in imgs[:split_idx]:
        shutil.copy(f"{SOURCE}/{cls}/{img}", f"{DEST}/train/{cls}/{img}")

    for img in imgs[split_idx:]:
        shutil.copy(f"{SOURCE}/{cls}/{img}", f"{DEST}/val/{cls}/{img}")