import os
import random


def create_files_txt(imgs_dir, train_split, img_ext):
    files = os.listdir(imgs_dir)
    # Each sequence ends after 60 frames. Therefore, I have to remove each frame multiple of number 0 and 60, since they don't have previous AND posterior frames
    filtered_img_files = []
    for img_idx in range(len(files)):
        # This section is for 10 frames only
        if img_idx != 0 and img_idx != 9:
            filtered_img_files.append(files[img_idx])
        # Section below is for the whole dataset
        # if img_idx % 60 != 0:
        #     filtered_img_files.append(files[img_idx])
        
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
    imgs_dir = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/CARLA/CARLA_1024x320/imgs_jpg"
    train_split = 0.80
    img_ext = ".jpg"
    
    create_files_txt(imgs_dir, train_split, img_ext)

