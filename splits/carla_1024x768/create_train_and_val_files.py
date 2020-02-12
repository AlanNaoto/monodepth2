import os


def create_files_txt(imgs_dir, train_split):
    files = os.listdir(imgs_dir)
    train_amount = int(len(files)*train_split)
    val_amount = int(len(files) - train_amount)

    lines_train = 0
    with open("train_files.txt", 'w') as f:
        for frame_idx, frame in enumerate(files):
            if frame_idx < train_amount:
                f.write(frame.replace(".png", "") + "\n")
                lines_train += 1
            else:
                break
    
    lines_val = 0
    with open("val_files.txt", 'w') as f:
        for frame_idx, frame in enumerate(files[train_amount:]):
            f.write(frame.replace(".png", "") + "\n")
            lines_val += 1

    print(f"Created {lines_train} train entries and {lines_val} val entries")    


if __name__ == "__main__":
    # Input
    imgs_dir = "/media/aissrtx2060/Naotop_1TB/data/CARLA_1024x768/imgs"
    train_split = 0.80
    
    create_files_txt(imgs_dir, train_split)

