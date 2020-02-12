# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

'''
TODO List
trainer.py
    - [ok] Change path to train/val txt files containing names of images (around line 120) - REMEMBER TO CHANGE FOR EACH DATASET!
    - In function train(), implement continuation of training  
    - [ok] Check if "depth_gt" (in run_epoch()) is indeed defined or not
options.py
    - [linked, fix value for real data] Change image input width and height default values
    - [linked, fix value for real data] Change data_path to point to the new directory with images (and maybe with the npy depth files too)
    - [ok] Check if img extension will be png or jpg
    - [ok] Change dataset reference
kitti_dataset.py
    class KITTIDataset(MonoDataset)
        - [ok] Change camera K intrinsic matrix to the corresponding dataset
        - [ok] Adjust resolution shape
        - [ok] Check if depth data needs to be set  

progress - fixing the input image files in kitti_dataset.py
'''


def perform_finer_network_changes(opts):
    opts.scales = 0
    opts.disparity_smoothness = 0    
    return opts


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    # Path to save results and logs
    # opts.load_weights_folder = "weights"  # If from scratch, comment
    opts.data_path = "/media/aissrtx2060/Naotop_1TB/data/CARLA_high_res"  # Path to dataset dir with imgs and annotations
    opts.eval_out_dir = "evaluations"
    opts.model_name = "carla_high_res_3900"
    opts.log_dir = "log"

    # Network general settings
    opts.dataset = 'carla'
    opts.width = 1920
    opts.height = 1080
    opts.num_epochs =  80
    opts.learning_rate = 1E-4  # 1E-4
    opts.batch_size = 1    
    opts.weights_init = "scratch"  # scratch or pretrained
    opts.eval_mono = True
    opts.save_pred_disps = True
    opts.min_depth = 0.1
    opts.max_depth = 80.0

    # Fine tuning stuff
    # opts = perform_finer_network_changes(opts)

    trainer = Trainer(opts)
    trainer.train()


