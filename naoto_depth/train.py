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
        - Change image input width and height default values
        - Change data_path to point to the new directory with images (and maybe with the npy depth files too)
        - Check if img extension will be png or jpg
        - Change split to Naoto_custom_split (to avoid later confusion with KITTI's zhou split, etc)          
'''

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

