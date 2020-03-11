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
opts.load_weights_folder = None  # If not to load weights, assign None
opts.data_path = "/home/alan/workspace/mestrado/dataset/CARLA_1024x320"  # Path to dataset dir with imgs and annotations
opts.eval_out_dir = "evaluations"
opts.model_name = "carla_1024x320_no_nights"
opts.split = "carla_1024x320_no_nights"
opts.log_dir = "results/carla_1024x320_no_nights"  # Path where the weights and general logging will be saved
opts.png = False

# Network general settings
opts.dataset = 'carla'  # waymo
opts.width = 1024
opts.height = 320
opts.num_epochs = 10
opts.learning_rate = 1E-4  # 1E-4
opts.batch_size = 1
opts.eval_mono = True
opts.save_pred_disps = True
opts.min_depth = 0.1
opts.max_depth = 75.0  # Adjusted to waymo's max sensor range

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
