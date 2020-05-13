# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

opts.weights_init = "scratch"  # For me, it is always from scratch (pretrained loads other weights from the repo)
opts.load_weights_folder = "../monodepth2_results/carla_1024x320_town_holdout_kitti_pretrained/models/weights_5"  # "pretrained_models/mono_1024x320"  # If not to load weights, assign None
opts.data_path = "/home/alan/workspace/mestrado/dataset/CARLA_1024x320"  # Path to root dataset dir containing CARLA and WAYMO for mixed. For unique, to its specific root dir
opts.eval_out_dir = "evaluations"
opts.model_name = "waymo_1024x320_town_holdout_carla_kitti_pretrained"  # "mixed_1024x320_town_holdout"
opts.split = "waymo_1024x320" # "mixed_1024x320"  # or _no_nights
opts.log_dir = "/home/alan/workspace/mestrado/monodepth2_results"  # Path where the weights and general logging will be saved (no need to put model name)
opts.png = False
opts.dataset='waymo'  # waymo or carla

# Network general settings
opts.width = 1024
opts.height = 320
opts.num_epochs = 30
opts.learning_rate = 1E-4  # 1E-4 from scratch. Maybe 1E-5 from pretrained?
opts.batch_size = 1
opts.eval_mono = True
opts.save_pred_disps = True
opts.min_depth = 0.1
opts.max_depth = 75.0  # Adjusted to waymo's max sensor range

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
