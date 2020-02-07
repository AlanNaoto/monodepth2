# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from aux_scripts.trainer import Trainer
from aux_scripts.options import MonodepthOptions

'''
TODO List
aux_scripts
    trainer.py
        - Change path to train/val txt files containing names of images (around line 120)
        - In function train(), implement continuation of training  
        - Check if "depth_gt" (in run_epoch()) is indeed defined or not
    options.py
        - [linked, fix value for real data] Change image input width and height default values
        - [linked, fix value for real data] Change data_path to point to the new directory with images (and maybe with the npy depth files too)
        - [ok] Check if img extension will be png or jpg
        - [ok] Change dataset reference
    kitti_dataset.py
        class KITTIDataset(MonoDataset)
            - Change camera K intrinsic matrix to the corresponding dataset
            - Adjust resolution shape
            - [+- ok, left as is] Check if depth data needs to be set  
'''


def perform_finer_network_changes(opts):
    opts.scales = 0
    opts.disparity_smoothness = 0
    opts.min_depth = 0.1
    opts.max_depth = 100.0
    return opts


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    # Path to save results and logs
    opts.load_weights_folder = ""  # If from scratch, leave empty
    opts.data_path = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/monodepth2/naoto_depth/dataset"
    opts.eval_out_dir = "evaluations"
    opts.model_name = "weights"
    opts.log_dir = "log"

    # Network general settings
    opts.dataset = 'carla'
    opts.width = 1024  # TODO Fix value to get from img sizes in dir
    opts.height = 768  # TODO same as above
    opts.num_epochs = 20
    opts.learning_rate = 1E-4
    opts.batch_size = 12
    opts.weights_init = "scratch"  # scratch or pretrained
    opts.eval_mono = True
    opts.save_pred_disps = True

    # Fine tuning stuff
    # opts = perform_finer_network_changes(opts)

    trainer = Trainer(opts)
    trainer.train()

