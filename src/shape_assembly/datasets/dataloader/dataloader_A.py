import os
import sys
import h5py
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import trimesh
from PIL import Image
import json
from progressbar import ProgressBar
import random
import copy
import time
import ipdb
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pdb import set_trace

os.environ['PYOPENGL_PLATFORM'] = 'egl'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../shape_assembly'))

from pytorch3d.transforms import quaternion_to_matrix
from mesh_to_sdf import sample_sdf_near_surface

def load_data(file_dir, cat_shape_dict):
    with open(file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            cat_shape_dict[cat].append(shape_id)
    return cat_shape_dict

class OurDataset(data.Dataset):

    def __init__(self, data_root_dir, data_csv_file, data_features=[], num_points = 1024 ,num_query_points = 1024 ,data_per_seg = 1):
        self.data_root_dir = data_root_dir
        self.data_csv_file = data_csv_file
        self.num_points = num_points
        self.num_query_points = num_query_points
        self.data_features = data_features
        self.data_per_seg = data_per_seg
        self.dataset = []
        
        with open(self.data_csv_file, 'r') as fin:
            self.category_list = [line.strip() for line in fin.readlines()]


    def transform_pc_to_rot(self, pcs):
        # zero-centered
        pc_center = (pcs.max(axis=0, keepdims=True) + pcs.min(axis=0, keepdims=True)) / 2
        pc_center = pc_center[0]
        new_pcs = pcs - pc_center

        # (batch_size, 2, 3)
        def bgs(d6s):
            bsz = d6s.shape[0]
            b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
            a2 = d6s[:, :, 1]
            b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
            b3 = torch.cross(b1, b2, dim=1)
            return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

        # randomly sample two rotation matrices
        rotmat = bgs(torch.rand(1, 6).reshape(-1, 2, 3).permute(0, 2, 1))
        new_pcs = (rotmat.reshape(3, 3) @ new_pcs.T).T

        gt_rot = rotmat[:, :, :2].permute(0, 2, 1).reshape(6).numpy()

        return new_pcs, pc_center, gt_rot


    def load_data(self):
        bar = ProgressBar()

        for category_i in bar(range(len(self.category_list))):
            category_id = self.category_list[category_i]
            instance_dir = os.path.join(self.data_root_dir, category_id)
            fileA = os.path.join(instance_dir, 'partA-pc.csv')
            fileB = os.path.join(instance_dir, 'partB-pc.csv')

            if not os.path.exists(fileA) or not os.path.exists(fileB):
                print("fileA is", fileA)
                print("fileB is", fileB)
                print("file not exists")
                continue

            dataframe_A = pd.read_csv(fileA, header=None)
            dataframe_B = pd.read_csv(fileB, header=None)
            gt_pcs_A = dataframe_A.to_numpy()
            gt_pcs_B = dataframe_B.to_numpy()
            
            for i in range(self.data_per_seg):
                gt_pcs_total = np.concatenate((gt_pcs_A, gt_pcs_B), axis=0)
                per = np.random.permutation(gt_pcs_total.shape[0])
                gt_pcs_total = gt_pcs_total[per]
             
                self.dataset.append([fileA,
                                     fileB,
                                    ])

    def __str__(self):
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        flag = 0
        while flag == 0:
            point_fileA, point_fileB, = self.dataset[index]

            dataframe_A = pd.read_csv(point_fileA, header=None)
            dataframe_B = pd.read_csv(point_fileB, header=None)
            gt_pcs_A = dataframe_A.to_numpy()
            gt_pcs_B = dataframe_B.to_numpy()

            if gt_pcs_A[0][0] != 0 and gt_pcs_A[0][1] != 0:
                flag = 1
            else:
                index += 1

        if gt_pcs_A[0][0] == 0 and gt_pcs_A[0][1] == 0:
            raise ValueError('getitem Zero encountered!')

        new_pcs_A, trans_A, rot_A = self.transform_pc_to_rot(gt_pcs_A)

        new_pcs_B, trans_B, rot_B = self.transform_pc_to_rot(gt_pcs_B)

        partA_symmetry_type = np.array([0,0,0,0,1,0])
        partB_symmetry_type = np.array([0,0,0,0,1,0])

        data_feats = dict()

        for feat in self.data_features:
            if feat == 'src_pc':
                data_feats['src_pc'] = new_pcs_A.T.float()

            elif feat == 'tgt_pc':
                data_feats['tgt_pc'] = torch.tensor(gt_pcs_B).T.float()

            elif feat == 'src_rot':
                data_feats['src_rot'] = rot_A.astype(np.float32)

            elif feat == 'tgt_rot':
                data_feats['tgt_rot'] = rot_B.astype(np.float32)

            elif feat == 'src_trans':
                data_feats['src_trans'] = trans_A.reshape(1, 3).T.astype(np.float32)

            elif feat == 'tgt_trans':
                data_feats['tgt_trans'] = trans_B.reshape(1, 3).T.astype(np.float32)

            elif feat == 'partA_symmetry_type':
                data_feats['partA_symmetry_type'] = partA_symmetry_type

            elif feat == 'partB_symmetry_type':
                data_feats['partB_symmetry_type'] = partB_symmetry_type

            elif feat == 'predicted_partB_position':
                data_feats['predicted_partB_position'] = np.array([0, 0, 0, 0]).astype(np.float32)

            elif feat == 'predicted_partB_rotation':
                data_feats['predicted_partB_rotation'] = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)

            elif feat == 'predicted_partA_position':
                data_feats['predicted_partA_position'] = np.array([0, 0, 0, 0]).astype(np.float32)

            elif feat == 'predicted_partA_rotation':
                data_feats['predicted_partA_rotation'] = np.array([0, 0, 0, 0, 0, 0]).astype(np.float32)

            elif feat == 'partA_mesh':
                data_feats['partA_mesh'] = self.load_mesh(data_feats, point_fileA)

            elif feat == 'partB_mesh':
                data_feats['partB_mesh'] = self.load_mesh(data_feats, point_fileB)

        return data_feats