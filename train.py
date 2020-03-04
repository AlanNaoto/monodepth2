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
opts.load_weights_folder = "/media/alan/Seagate Expansion Drive/monodepth2_results/carla_640x192_town01/carla_640x192/models/weights_0"  # If not to load weights, assign None
opts.data_path = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/CARLA/CARLA_1024x320"  # Path to dataset dir with imgs and annotations
opts.eval_out_dir = "evaluations"
opts.model_name = "carla_1024x320"
opts.split = "carla_1024x320"
opts.log_dir = "results/carla_1024x320_10_frames"  # Path where the weights and general logging will be saved
opts.png = False

# Network general settings
opts.dataset = 'carla'
opts.width = 1024
opts.height = 320
opts.num_epochs = 10
opts.learning_rate = 1E-4  # 1E-4
opts.batch_size = 1
opts.eval_mono = True
opts.save_pred_disps = True
opts.min_depth = 0.1
opts.max_depth = 80.0

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
