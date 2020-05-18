import os
import random
import sqlite3


def get_waymo_frames(imgs_dir, metadata_db_file):
    # Initialize db file with information on frame end/finish (has other stuff as well)
    conn = sqlite3.connect(metadata_db_file)
    c = conn.cursor()
    c.execute('''SELECT * FROM waymo_metadata''')
    db_data = c.fetchall()

    # Removing start and end of sequence frames
    filtered_img_files = []
    files = sorted(os.listdir(imgs_dir))
    for frame_idx in range(len(files)):
        # Skipping start (idx 5) and end (idx 6) frames
        if db_data[frame_idx][5] != 1 and db_data[frame_idx][6] != 1:
            filtered_img_files.append(files[frame_idx])
    conn.close()
    return filtered_img_files


def get_carla_frames(imgs_dir):
    files = sorted(os.listdir(imgs_dir))
    # Each sequence ends after 60 frames. Therefore, I have to remove each frame multiple of number 0 and 60, since they don't have previous AND posterior frames
    filtered_img_files = []
    for img_idx in range(len(files)):
        if ((img_idx+1) % 60 != 0) and ((img_idx%60) != 0):
            filtered_img_files.append(files[img_idx])
    return filtered_img_files


def split_train_val(filtered_img_files):
    # 80% (training) = 80% (waymo) + 80% (carla)
    # Separate data splits and create txt files
    train_amount = int(len(filtered_img_files)*train_split)
    val_amount = int(len(filtered_img_files) - train_amount)
    train_files = filtered_img_files[:train_amount]
    val_files = filtered_img_files[train_amount:]
    return train_files, val_files


def create_files_txt(carla_imgs_dir, waymo_imgs_dir, waymo_metadata_db_file, \
                     train_split, img_ext):
    """
    Main idea: assign normal idxs to CARLA (0-19499), but assign offset indexes
    for waymo (19500->end) so that later dir parsing is easier.
    """
    waymo_frames = get_waymo_frames(waymo_imgs_dir, waymo_metadata_db_file)
    carla_frames = get_carla_frames(carla_imgs_dir)
    print(carla_frames[0], carla_frames[-1])
    print(waymo_frames[0], waymo_frames[-1])
    waymo_idx_offset = len(os.listdir(carla_imgs_dir))
    for frame_idx in range(len(waymo_frames)):
        waymo_frames[frame_idx] = int(waymo_frames[frame_idx].replace(".jpg", ""))
        waymo_frames[frame_idx] += waymo_idx_offset
        waymo_frames[frame_idx] = f"{waymo_frames[frame_idx]:05d}.jpg"

    waymo_train_files, waymo_val_files = split_train_val(waymo_frames)
    carla_train_files, carla_val_files = split_train_val(carla_frames)
    train_files = waymo_train_files + carla_train_files
    val_files = waymo_val_files + carla_val_files
    random.shuffle(train_files)
    random.shuffle(val_files)

    with open("train_files.txt", 'w') as f:
        for frame_idx, frame in enumerate(train_files):
            f.write(frame.replace(img_ext, "") + "\n")

    with open("val_files.txt", 'w') as f:
        for frame_idx, frame in enumerate(val_files):
            f.write(frame.replace(img_ext, "") + "\n")

    print(f"Created {len(train_files)} train entries and {len(val_files)} val entries")


if __name__ == "__main__":
    # Input
    waymo_imgs_dir = "/home/alan/workspace/mestrado/dataset/WAYMO_1024x320/imgs_jpg"
    waymo_metadata_db_file = "/home/alan/workspace/mestrado/dataset/WAYMO_1024x320/translation_metadata.db"
    carla_imgs_dir = "/home/alan/workspace/mestrado/dataset/CARLA_1024x320/imgs_jpg"
    train_split = 0.80
    img_ext = ".jpg"
    create_files_txt(carla_imgs_dir, waymo_imgs_dir, waymo_metadata_db_file, \
                    train_split, img_ext)

