#!/usr/bin/env python
import time
import datetime
import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from robot import Robot
from evaluator import Evaluator
from logger import Logger
from generator import grasp_generator, push_generator, get_pointcloud
import utils
global sample_iteration

def main(args):
    # --------------- Setup options ---------------
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # -------------- Testing options --------------
    is_testing = True
    test_target_seeking = args.test_target_seeking
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    max_motion_onecase = args.max_motion_onecase

    # ------ Pre-loading and logging options ------
    load_ckpt = args.load_ckpt  # Load pre-trained ckpt of model
    critic_ckpt_file = os.path.abspath(args.critic_ckpt) if load_ckpt else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    save_visualizations = args.save_visualizations

    # ------ Initialize some status variables -----
    seeking_target = False,
    margin_occupy_ratio = None,
    margin_occupy_norm = None,
    best_grasp_pix_ind = None,
    best_pix_ind = None,
    grasp_succeeded = False,
    grasp_effective = False,
    target_grasped = False

    color_img = cv2.imread("/home/jack/research/swiperl/real_world_prehensile_pics/normal_scene_processed/3_rgb.png")
    depth_img = np.load("/home/jack/research/swiperl/real_world_prehensile_pics/normal_scene_processed/3depth.npy")
    seg_img = torch.Tensor(cv2.imread("/home/jack/research/swiperl/real_world_prehensile_pics/normal_scene_processed/3_segmask.png"))
    uniques = torch.unique(seg_img.reshape(-1, 3), dim=0)
    new_seg_img = torch.zeros((seg_img.shape[0], seg_img.shape[1], uniques.shape[0]-1))

    target_idx = 2

    mask_idx = 1
    for i, u in enumerate(uniques):
        if (u != torch.Tensor([84, 1, 68])).all():
            new_seg_img[:, :, mask_idx-1][(seg_img == u)[:, :, 0]] = mask_idx
            mask_idx += 1

    seg_img = new_seg_img


    # cam_intrinsics = torch.Tensor([[450.004852,0,316.109192],[0, 450.004852, 179.689819],[0,0,1]])
    cam_intrinsics = torch.Tensor([[450.004852,0,260//2],[0, 450.004852, 180//2],[0,0,1]])
    cam_pose = torch.eye(4)

    np.random.seed(1234)
    evaluator = Evaluator(0.5, is_testing, load_ckpt, critic_ckpt_file, force_cpu)

    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap, seg_mask_heightmaps, workspace_limits = utils.get_heightmap(
        color_img, depth_img, seg_img, cam_intrinsics.numpy(), cam_pose, workspace_limits,
        heightmap_resolution) 

    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    # mask_heightmaps = utils.process_mask_heightmaps(segment_results, seg_mask_heightmaps) 

    # Choose target
    # if len(mask_heightmaps['names']) == 0 and test_target_seeking:
    #     seeking_target = True
    #     target_mask_heightmap = np.ones_like(valid_depth_heightmap)
    # else:
    #     seeking_target = False
    #     if target_name in mask_heightmaps['names']:
    #         target_id = mask_heightmaps['names'].index(target_name)
    #         target_mask_heightmap = mask_heightmaps['heightmaps'][target_id]
    #     else:
    #         target_id = 0  # Set the first object in the segmentation result as the target
    #         target_name = mask_heightmaps['names'][target_id]
    #         target_mask_heightmap = mask_heightmaps['heightmaps'][target_id]

    target_mask_heightmap = seg_mask_heightmaps[:, :, target_idx]
    
    y, x = np.where(target_mask_heightmap!=0)
    target_center = (int(np.mean(x)), int(np.mean(y)))

    # Generate grasp and push candidates
    point_cloud = get_pointcloud(heightmap_resolution, valid_depth_heightmap, workspace_limits)

    # Choose best push
    pushes = push_generator(target_center, valid_depth_heightmap)
    evaluator.model.load_state_dict(torch.load('saved_models/pushing.pkl'))

    push_confs = []
    for push in pushes:
        push_start_point = push[0]
        rot_ind = push[1]
        push_mask = push[2]

        depth_heightmap = np.asarray(valid_depth_heightmap * 370, dtype=np.int) # ***
        target_mask = target_mask_heightmap * 255
        push_mask_input = push_mask * 255
 
        confidence, _ = evaluator.forward(depth_heightmap, target_mask, push_mask_input)
        push_confs.append(confidence.item())
    print('best push value: ', np.max(push_confs))
    best_push_ind = np.argmax(push_confs)
    best_push_mask = pushes[best_push_ind][2]
    best_push_start_point = pushes[best_push_ind][0]
    best_push_rot_ind = pushes[best_push_ind][1]
    # push_start_position = point_cloud_reshaped[best_push_start_point[0], best_push_start_point[1]]

    # Choose best grasp
    grasps, grasp_mask_heightmaps, num_grasps = grasp_generator(target_center, point_cloud, valid_depth_heightmap)
    evaluator.model.load_state_dict(torch.load('saved_models/grasping.pkl'))

    ########### Coordinating between pushing and grasping ###########
    if num_grasps == 0:
        primitive_action = 'push'
    elif num_grasps > 100:
        sampled_inds = np.random.choice(np.arange(num_grasps), 100, replace=False)
    else:
        sampled_inds = np.random.choice(np.arange(num_grasps), num_grasps, replace=False)

    if num_grasps > 0:
        confs, grasp_inds, rot_inds = [], [], []
        grasp_masks = []
        for i in sampled_inds:
            grasp_mask_heightmap = grasp_mask_heightmaps[i][0]

            depth_heightmap = np.asarray(valid_depth_heightmap * 370, dtype=np.int) # ***
            target_mask = target_mask_heightmap * 255
            grasp_mask = grasp_mask_heightmap * 255

            confidence, _ = evaluator.forward(depth_heightmap, target_mask, grasp_mask)

            confs.append(confidence.item())
            grasp_inds.append(grasp_mask_heightmaps[i][1])
            rot_inds.append(grasp_mask_heightmaps[i][2])
            grasp_masks.append(grasp_mask_heightmaps[i][0])

        grasp_inds = np.hstack((np.array(rot_inds).reshape((-1, 1)), np.array(grasp_inds)))
        grasp_masks = np.array(grasp_masks)

        best_grasp_conf = np.max(confs)
        best_grasp_ind = np.argmax(confs)
        best_grasp_pix_ind = grasp_inds[best_grasp_ind]
        best_grasp_mask = grasp_masks[best_grasp_ind]

    else:
        best_grasp_conf = 0
        primitive_action = 'push'

    if best_grasp_conf < 1.0:  
        primitive_action = 'push'
        best_pix_ind = [best_push_rot_ind, best_push_start_point[0], best_push_start_point[1]]
    else:
        primitive_action = 'grasp'
        best_pix_ind = best_grasp_pix_ind

    print(primitive_action)
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(color_heightmap)
    ax[0].set_title("Color Heightmap")
    ax[1].imshow(color_heightmap)
    ax[1].imshow(best_push_mask, alpha=0.2)
    ax[1].set_title("Push Action Overlaid")
    ax[2].imshow(color_heightmap)
    ax[2].imshow(best_grasp_mask, alpha=0.2)
    ax[2].set_title("Grasp Action Overlaid")
    ax[3].imshow(target_mask_heightmap)
    ax[3].set_title("Target Object Segmentation Heightmap")
    fig.suptitle(f"Chosen Action: {primitive_action}")
    plt.show()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234)
    parser.add_argument('--force_cpu', dest='force_cpu', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_target_seeking', dest='test_target_seeking', action='store_true', default=False)
    parser.add_argument('--max_motion_onecase', dest='max_motion_onecase', type=int, action='store', default=20,
                        help='maximum number of motions per test trial')
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=5,
                        help='number of repeated test trials')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_true', default=False)
    parser.add_argument('--critic_ckpt', dest='critic_ckpt', action='store')
    parser.add_argument('--coordinator_ckpt', dest='coordinator_ckpt', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False)
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True)

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
