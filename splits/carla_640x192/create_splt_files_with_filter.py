import os
import random


def create_files_txt(imgs_dir, train_split):
    files = os.listdir(imgs_dir)
    
    # Each sequence ends after 60 frames. Therefore, I have to remove each frame multiple of number 0 and 60, since they don't have previous AND posterior frames
    filtered_img_files = []
    for img_idx in range(len(files)):
        if img_idx // 60 != 0:
            filtered_img_files.append(files[img_idx])

    random.shuffle(filtered_img_files)
    train_amount = int(len(filtered_img_files)*train_split)
    #val_amount = int(len(filtered_img_files) - train_amount)

    lines_train = 0
    with open("train_files.txt", 'w') as f:
        for frame_idx, frame in enumerate(filtered_img_files):            
            if frame_idx < train_amount:
                f.write(frame.replace(".png", "") + "\n")                
            else:
                break
            lines_train += 1
    
    lines_val = 0
    with open("val_files.txt", 'w') as f:
        val_frames = filtered_img_files[train_amount:]
        for frame_idx, frame in enumerate(val_frames):
            f.write(frame.replace(".png", "") + "\n")
            lines_val += 1

    print(f"Created {lines_train} train entries and {lines_val} val entries")        

if __name__ == "__main__":
    # Input
    imgs_dir = "/media/aissrtx2060/Seagate Expansion Drive/Data/CARLA_640x192/imgs_png_renamed"
    train_split = 0.80
    
    create_files_txt(imgs_dir, train_split)

