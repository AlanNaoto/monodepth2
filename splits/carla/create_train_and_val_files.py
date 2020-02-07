import os


def create_files_txt(dir_with_imgs, filename=None):
    files = os.listdir(dir_with_imgs)
    with open(filename, 'w') as f:
        [f.write(x.replace(".jpg", "") + "\n") for x in files]


if __name__ == "__main__":
    # Input
    train_imgs_dir = '/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/depth_estimation/monodepth2/data/imgs_train'
    val_imgs_dir = '/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/depth_estimation/monodepth2/data/imgs_val'

    create_files_txt(train_imgs_dir, filename='train_files.txt')
    create_files_txt(val_imgs_dir, filename='val_files.txt')

