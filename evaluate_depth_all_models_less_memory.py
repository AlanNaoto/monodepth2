from __future__ import absolute_import, division, print_function

import tempfile
import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def get_pred_disps(opt, split, tmp_dir_path, out_dir):
    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        #filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        train_or_val = {"train": "train_files.txt", "val": "val_files.txt"}    
        filenames = sorted(readlines(os.path.join(splits_dir, opt.eval_split, train_or_val[split])))  # sorted facilitates our life
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.carla_dataset.CarlaDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)  # Changed batch from 16 to 1 (before was evaluating only total/16 it seems?)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for frame_idx, data in enumerate(dataloader):
                if frame_idx % 100 == 0:
                    print(f"Creating disparity {frame_idx}/{len(dataloader)}")

                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                #pred_disps.append(pred_disp)
                frame_name = filenames[frame_idx]
                output_path = os.path.join(tmp_dir_path, f"{frame_name}.npy")
                np.save(output_path, pred_disp)
#        pred_disps = np.concatenate(pred_disps)
    
    #output_path = os.path.join(opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
#    print("-> Saving predicted disparities to ", output_path)
    print(f"Saved predicted disparities to temporary dir {tmp_dir_path}")
    disparity_files = [os.path.join(tmp_dir_path, x) for x in os.listdir(tmp_dir_path)]
    disparity_files = sorted(disparity_files)
    return disparity_files, filenames


def evaluate(opt, split, dataset, tmp_dir_root, out_dir):
    """Evaluates a pretrained model using a specified test set
    """
    #MIN_DEPTH = 1e-3
    #MAX_DEPTH = 80
    MIN_DEPTH = opt.min_depth
    MAX_DEPTH = opt.max_depth

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    tmp = tempfile.TemporaryDirectory(dir=tmp_dir_root)
    tmp_dir_path = tmp.name
    disparity_files, split_filenames = get_pred_disps(opt, split, tmp_dir_path, out_dir)

    #gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
    gt_dir = os.path.join(opt.data_path, "depth_npy")
    gt_depths = {x: os.path.join(gt_dir, x) for x in os.listdir(gt_dir)}

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    for i in range(len(split_filenames)):
        if i % 100 == 0:
            print(f"Computing error {i}/{len(split_filenames)}")

        gt_filename = split_filenames[i] + ".npy"  # This is done since predict is a subset from gt
        if dataset == 'waymo':
            gt_depth = np.zeros((1280, 1920))  # Original resolution (height, width)
            lidar_data = np.load(gt_depths[gt_filename])
            for lidar_point in lidar_data:
                gt_depth[int(lidar_point[1])][int(lidar_point[0])] = lidar_point[2]
            gt_depth = cv2.resize(gt_depth, (1024, 320))
        else:
            gt_depth = np.load(gt_depths[gt_filename])

        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = np.load(disparity_files[i])
        pred_disp = pred_disp[0]
        #pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))  # FIXME My GT is already the same shape as the predictions
        pred_depth = 1 / pred_disp

        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    metrics = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

    with open(os.path.join(out_dir, split, os.path.basename(opts.load_weights_folder)) + ".txt", 'w') as f:
        errors = mean_errors.tolist()
        for metric_idx, metric in enumerate(metrics):
            f.write(f"{metric}: {errors[metric_idx]:.5f}\n")
        f.write(f"Scaling ratios\n")
        std_scale = np.std(ratios/med)
        f.write(f"med: {med:.5f}  std: {std_scale:.5f}")
    tmp.cleanup()


if __name__ == "__main__":
    dir_model_weights = "/home/alan/workspace/mestrado/monodepth2_results/waymo_1024x320/models"
    out_dir = "/home/alan/workspace/mestrado/monodepth2_results/waymo_1024x320/metrics"
    options = MonodepthOptions()
    opts = options.parse()
    opts.eval_mono = True    
    opts.eval_split = "waymo_1024x320"  # Testing on complete carla 
    opts.data_path = "/home/alan/workspace/mestrado/dataset/WAYMO_1024x320"
    #opts.load_weights_folder = "/media/aissrtx2060/Naotop_1TB1/monodepth2_data/carla_1024x320_full/carla_1024x320/models/weights_9"
    opts.max_depth = 75.0
    opts.min_depth = 0.1
    dataset = 'waymo'
    tmp_dir_root = "/home/alan/workspace/mestrado/temp"
    splits = ['train', 'val']

    model_paths = [os.path.join(dir_model_weights, model) for model in os.listdir(dir_model_weights) if os.path.isdir(os.path.join(dir_model_weights, model))]
    for model_path in model_paths:
        opts.load_weights_folder = model_path
        for split in splits:
            os.makedirs(os.path.join(out_dir, split), exist_ok=True)
            evaluate(opts, split, dataset, tmp_dir_root, out_dir)
