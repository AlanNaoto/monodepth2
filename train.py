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

opts.weights_init = "scratch"  # scratch or pretrained
#opts.load_weights_folder = ""  # If from scratch, comment (still FIXME on loading the correct files)
opts.data_path = "/media/aissrtx2060/Seagate Expansion Drive/Data/CARLA_640x192"  # Path to dataset dir with imgs and annotations
opts.eval_out_dir = "evaluations"
opts.model_name = "carla_640x192"
opts.split = "carla_640x192"
opts.log_dir = "/media/aissrtx2060/Seagate Expansion Drive/monodepth2_results/carla_640x192_town01"  # Path where the weights and general logging will be saved
opts.png = False

# Network general settings
opts.dataset = 'carla'
opts.width = 640
opts.height = 192
opts.num_epochs = 10
opts.learning_rate = 1E-4  # 1E-4
opts.batch_size = 5
opts.eval_mono = True
opts.save_pred_disps = True
opts.min_depth = 0.1
opts.max_depth = 80.0

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
