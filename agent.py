import os
import np
import torch

import utils
from evaluator import Evaluator
from generator import grasp_generator, push_generator, get_pointcloud
from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from rl.rl_utils import transform_pose

class GE_Grasp_agent(object):
    def __init__(self, **kwargs):
        # --------------- Setup options ---------------
        self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.heightmap_resolution = 0.002
        self.force_cpu = False
        self.load_ckpt = True
        # -------------- Testing options --------------
        self.is_testing = True
        # ------ Pre-loading and logging options ------
        self.critic_ckpt_file = os.path.abspath("critic_ckpt")

        # ------ Initialize some status variables -----
        # self.best_grasp_pix_ind = None

    # seperate a segmentation mask with N different object classes in it, to N seperate segmasks with only a single object in them
    def convert_segmask_to_GE_form(self, segmask):
        uniques = torch.unique(segmask.reshape(-1, 3), dim=0)
        new_seg_img = torch.zeros((segmask.shape[0], segmask.shape[1], uniques.shape[0]-1))

        mask_idx = 1
        for u in uniques:
            if (u != torch.Tensor([84, 1, 68])).all():
                new_seg_img[:, :, mask_idx-1][(segmask == u)[:, :, 0]] = mask_idx
                mask_idx += 1

        return new_seg_img

    def select_action(self, state, cam_intrinsics, target_idx):#only used when interact with the env
        color_img = state[0]
        depth_img = state[1]
        seg_img = state[2]

        seg_img = self.convert_segmask_to_GE_form(seg_img)

        cam_pose = torch.eye(4)
        evaluator = Evaluator(0.5, self.is_testing, self.load_ckpt, self.critic_ckpt_file, self.force_cpu)

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap, seg_mask_heightmaps, workspace_limits = utils.get_heightmap(
            color_img, depth_img, seg_img, cam_intrinsics.numpy(), cam_pose, workspace_limits,
            self.heightmap_resolution) 

        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        target_mask_heightmap = seg_mask_heightmaps[:, :, target_idx]

        y, x = np.where(target_mask_heightmap!=0)
        target_center = (int(np.mean(x)), int(np.mean(y)))
        target_position = point_cloud_reshaped[target_center]

        # Generate grasp and push candidates
        point_cloud = get_pointcloud(self.heightmap_resolution, valid_depth_heightmap, workspace_limits)

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
        point_cloud_reshaped = point_cloud.reshape((224, 224, -1))
        push_start_position = point_cloud_reshaped[best_push_start_point[0], best_push_start_point[1]]

        push_start_base_link = Pose()
        push_start_base_link.position.x = push_start_position[0].item()
        push_start_base_link.position.y = push_start_position[1].item()
        push_start_base_link.position.z = push_start_position[2].item
        push_start_base_link.orientation.w = 1
        push_start_base_link = transform_pose(push_start_base_link, "camera_depth_optical_frame", "base_link")
        

        random_rot_angle = (best_push_rot_ind - 1) * np.pi / 8
        random_rot_mat = np.array([[np.cos(random_rot_angle), -np.sin(random_rot_angle)],
                                   [np.sin(random_rot_angle), np.cos(random_rot_angle)]])
        push_direction = [target_position[0]-push_start_position[0], target_position[1]-push_start_position[1]]
        push_direction = push_direction / np.linalg.norm(push_direction)
        push_direction = np.dot(push_direction, random_rot_mat)

        push_length = 0.1
        push_end_position = push_start_position.copy()
        push_end_position[0: 2] += push_length * push_direction

        target_position_base_link = Pose()
        target_position_base_link.position.x = push_end_position[0]
        target_position_base_link.position.y = push_end_position[1]
        target_position_base_link.position.z = push_end_position[2]
        target_position_base_link.orientation.w = 1
        target_position_base_link = transform_pose(target_position_base_link, "camera_depth_optical_frame", "base_link")

        return (push_start_position, best_push_rot_ind, target_position_base_link)



        
