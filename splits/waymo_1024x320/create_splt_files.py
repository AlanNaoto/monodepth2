import os
import random
import sqlite3


def create_files_txt(imgs_dir, train_split, img_ext, metadata_db_file):
    # Initialize db file with information on frame end/finish (has other stuff as well)
    conn = sqlite3.connect(metadata_db_file)
    c = conn.cursor()
    c.execute('''SELECT * FROM waymo_metadata''')
    db_data = c.fetchall()

    # Begin parsing throough images as reference
    filtered_img_files = []
    files = sorted(os.listdir(imgs_dir))
    for frame_idx in range(len(files)):
        # Skipping start (idx 5) and end (idx 6) frames
        if db_data[frame_idx][5] != 1 and db_data[frame_idx][6] != 1:
            filtered_img_files.append(files[frame_idx])
    conn.close()

    train_amount = int(len(filtered_img_files)*train_split)
    val_amount = int(len(filtered_img_files) - train_amount)
    train_files = filtered_img_files[:train_amount]
    val_files = filtered_img_files[train_amount:]

    random.shuffle(train_files)
    random.shuffle(val_files)

    with open("train_files.txt", 'w') as f:
        for frame_idx, frame in enumerate(train_files):
            f.write(frame.replace(img_ext, "") + "\n")

    with open("val_files.txt", 'w') as f:
        for frame_idx, frame in enumerate(val_files):
            f.write(frame.replace(img_ext, "") + "\n")

    print(f"Created {train_amount} train entries and {val_amount} val entries")


if __name__ == "__main__":
    # Input
    imgs_dir = "/home/alan/workspace/mestrado/dataset/WAYMO_1024x320/imgs_jpg"
    img_ext = ".jpg"
    metadata_db_file = "/home/alan/workspace/mestrado/dataset/WAYMO_1024x320/translation_metadata.db"
    train_split = 0.80

    create_files_txt(imgs_dir, train_split, img_ext, metadata_db_file)

