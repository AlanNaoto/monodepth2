import numpy as np
import PIL.Image as pil
import os
import pdb
import cv2

from .mono_dataset import MonoDataset


class MixedDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(MixedDataset, self).__init__(*args, **kwargs)
        fov = 90 # Degrees
        img_width = 1024
        img_height = 320

        focal = img_width / (2 * np.tan(fov * np.pi / 360))
        self.K = np.array([[focal / img_width, 0, 0.5, 0],
                        [0, focal / img_height, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (img_width, img_height)
    
    def check_dataset_and_img_idx(self, img_idx):
        """
        I've indexed the files sequentially, starting from carla (1-19499, and then waymo 19500->end)
        Therefore according to img_idx I can set the right directory to get the files
        """
        dataset = "CARLA_1024x320"
        if (img_idx - 19500) >= 0:
            img_idx = img_idx - 19500
            dataset = "WAYMO_1024x320"
        return dataset, img_idx

    def check_depth(self):
        return True

    def get_image_path(self, folder, frame_index, side):
        img_idx = int(folder)
        dataset, img_idx = self.check_dataset_and_img_idx(img_idx)
        img_idx = img_idx + frame_index  # Gets the previous, current or next frame for comparison
        img_file = f'{img_idx:05d}.jpg'
        image_path = os.path.join(self.data_path, dataset, 'imgs_jpg', img_file)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        img_idx = int(folder)
        dataset, img_idx = self.check_dataset_and_img_idx(img_idx)
        img_idx = img_idx + frame_index  # Gets the previous, current or next frame for comparison
        depth_file = f'{img_idx:05d}.npy'
        depth_path = os.path.join(self.data_path, dataset, 'depth_npy', depth_file)
        if dataset == "CARLA_1024x320":
            depth_gt = np.load(depth_path)
            #depth_gt = np.transpose(depth_gt)
        elif dataset == "WAYMO_1024x320":
            # Transform data from LIDAR standard to our img-like array standard
            depth_gt = np.zeros((1280, 1920))  # Original resolution (height, width)
            lidar_data = np.load(depth_path)
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


