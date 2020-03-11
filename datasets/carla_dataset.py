import numpy as np
import PIL.Image as pil
import os
import pdb

from .mono_dataset import MonoDataset


class CarlaDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CarlaDataset, self).__init__(*args, **kwargs)
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
        return False # don't have it on this computer

    def get_image_path(self, folder, frame_index, side):        
        img_idx = int(folder)
        img_idx = img_idx + frame_index  # Gets the previous, current or next frame for comparison
        img_file = f'{img_idx:05d}.jpg'
        image_path = os.path.join(self.data_path, 'imgs_jpg', img_file)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        frame_idx = int(folder)
        frame_idx = frame_idx + frame_index  # Gets the previous, current or next frame for comparison
        depth_file = f'{frame_idx:05d}.npy'
        depth_path = os.path.join(self.data_path, 'depth_npy', depth_file)
        depth_gt = np.load(depth_path)
        depth_gt = np.transpose(depth_gt)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color


