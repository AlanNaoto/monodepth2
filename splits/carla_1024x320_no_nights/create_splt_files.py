import os
import random


def create_files_txt(imgs_dir, train_split, img_ext):
    files = sorted(os.listdir(imgs_dir))
    night_frames = 0
    filtered_img_files = []
    for img_idx in range(len(files)):
        # Each sequence ends after 60 frames. Therefore, I have to remove each frame multiple of number 0 and 60, since they don't have previous AND posterior frames
        if ((img_idx+1) % 60 != 0) and ((img_idx%60) != 0):
            # Night frames appear at the end of each town sequence. That is, 60frames*13egos*5weathers=3900 frames per town and 780 frames per weather. Range of night = (3120, 3900)        
            if 3120 <= img_idx%3900 < 3900:
                night_frames+=1
            else:
                filtered_img_files.append(files[img_idx])
    print(f'skipped {night_frames} night frames')

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
    imgs_dir = "/home/alan/workspace/mestrado/dataset/CARLA_1024x320/imgs_jpg"
    train_split = 0.80
    img_ext = ".jpg"
    
    create_files_txt(imgs_dir, train_split, img_ext)

