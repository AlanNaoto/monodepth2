import numpy as np
import PIL.Image as pil
import os
import cv2

from .mono_dataset import MonoDataset


class WaymoDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(WaymoDataset, self).__init__(*args, **kwargs)
        fov = 90 # Degrees
        img_width = 1024
        img_height = 320

        focal = img_width / (2 * np.tan(fov * np.pi / 360))
        self.K = np.array([[focal / img_width, 0, 0.5, 0],
                        [0, focal / img_height, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (img_width, img_height)

    def check_depth(self):
        return True

    def get_image_path(self, folder, frame_index, side):        
        img_idx = int(folder)
        img_idx = img_idx + frame_index  # Gets the previous, current or next frame for comparison
        img_file = f'{img_idx:05d}.jpg'
        image_path = os.path.join(self.data_path, 'imgs_jpg', img_file)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        frame_idx = int(folder)
        frame_idx = frame_idx + frame_index  # Gets the previous, current or next frame for comparison
        lidar_file = f'{frame_idx:05d}.npy'
        lidar_path = os.path.join(self.data_path, 'anns_lidar_npy', lidar_file)
        lidar_data = np.load(lidar_path)

        # Transform data from LIDAR standard to our img-like array standard
        depth_gt = np.zeros((1280, 1920))  # Original resolution (height, width)
        for lidar_point in lidar_data:
            depth_gt[int(lidar_point[1])][int(lidar_point[0])] = lidar_point[2]
        # Since we are resizing the GT, then its a very raw approximation
        depth_gt = cv2.resize(depth_gt, (self.full_res_shape[0], self.full_res_shape[1]))    
 
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color


