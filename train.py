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
    opts.weights_init = "scratch"  # scratch or pretrained
    #opts.load_weights_folder = ""  # If from scratch, comment (still FIXME on loading the correct files)
    opts.data_path = "/media/aissrtx2060/Seagate Expansion Drive/Data/CARLA_640x192"  # Path to dataset dir with imgs and annotations
    opts.eval_out_dir = "evaluations"
    opts.model_name = "carla_640x192"
    opts.split = "carla_640x192"
    opts.log_dir = "/media/aissrtx2060/Seagate Expansion Drive/monodepth2_results/carla_640x192_sequential_names"  # Path where the weights and general logging will be saved
    opts.png = True

    # Network general settings
    opts.dataset = 'carla'
    opts.width = 640
    opts.height = 192
    opts.num_epochs =  15
    opts.learning_rate = 1E-4  # 1E-4
    opts.batch_size = 5
    opts.eval_mono = True
    opts.save_pred_disps = True
    opts.min_depth = 0.1
    opts.max_depth = 80.0

    # Fine tuning stuff
    # opts = perform_finer_network_changes(opts)    

    trainer = Trainer(opts)
    trainer.train()


