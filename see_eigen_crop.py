import cv2


def see_eigen_crop(img):
    print(f'img shape: {img.shape}')

    # Mask given in monodepth2's code for a KITTI dataset
    # mask = np.zeros((512, 1382))
    # mask[153:371, 44:1197] = 1

    # Paint rectangle of crop in image
    overlay = img.copy()
    alpha = 0.4  # Transparency factor.
    eigen_crop = [(44, 153), (1197, 371)]
    cv2.rectangle(img, eigen_crop[0], eigen_crop[1], (0, 0, 255), -1)  # xmin ymin xmax ymax
    # Transparent rectangle
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imshow('eigencrop', img_new)
    cv2.waitKey(0)
    cv2.imwrite('eigencrop_visualized.jpg', img_new)


def see_carla_crop(img):
    print(f'img shape: {img.shape}')

    # Paint rectangle of crop in image
    overlay = img.copy()
    alpha = 0.4  # Transparency factor.
    # carla crop would be [155:371, 44:980]
    carla_crop = [(44, 155), (1024-44, 371)]
    cv2.rectangle(img, carla_crop[0], carla_crop[1], (0, 0, 255), -1)  # xmin ymin xmax ymax
    # Transparent rectangle
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imshow('carlacrop', img_new)
    cv2.waitKey(0)
    cv2.imwrite('carlacrop_visualized.jpg', img_new)


if __name__ == "__main__":
    img_kitti = cv2.imread("/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/kitti_sample/0000000000.png")
    img_carla = cv2.imread("/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/CARLA/CARLA_1024x320/imgs_jpg/00000.jpg")
    see_eigen_crop(img_kitti)
    see_carla_crop(img_carla)


