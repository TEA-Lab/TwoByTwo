import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix
from transforms3d.quaternions import quat2mat
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))
from models.encoder.pointnet import PointNet
from models.encoder.dgcnn import DGCNN
from models.decoder.MLPDecoder import MLPDecoder
from models.encoder.vn_dgcnn import VN_DGCNN, VN_DGCNN_corr, VN_DGCNN_New, DGCNN_New 
from models.baseline.regressor_CR import Regressor_CR, Regressor_6d, VN_Regressor_6d
import utils
from pdb import set_trace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("BASE_DIR: ", BASE_DIR)
from ..ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
# from chamfer_distance import ChamferDistance as dist_chamfer_3D

def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

def debug_vis_input(batch_data, cfg, prd_data, iter_counts):
    for i in range(cfg.exp.batch_size):
        # print("i is,", i)
        save_dir = cfg.exp.vis_dir
        vis_dir = os.path.join(save_dir, 'vis_B_input')
        if not os.path.exists((vis_dir)):
            os.mkdir(vis_dir)
        # src_pc = batch_data['src_pc'][i]  # (3, 1024)
        tgt_pc = batch_data['tgt_pc'][i]  # (3, 1024)

        device = tgt_pc.device
        num_points = tgt_pc.shape[1]

        total_pc = np.array(tgt_pc.detach().cpu())
        color_mask_tgt = ['g'] * num_points

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(list(total_pc[0]), list(total_pc[1]), list(total_pc[2]), c=color_mask_tgt, s=10, alpha=0.9)
        ax.axis('scaled')
        ax.set_zlabel('Z', fontdict={'size': 20, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 20, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})

        fig.savefig(os.path.join(vis_dir, 'batch_{}_{}_input'.format(iter_counts,i)))
        plt.close(fig)

def debug_vis_output(batch_data, cfg, pred_data, iter_counts):

    # for every data in batch_size
    for i in range(cfg.exp.batch_size):
        save_dir = cfg.exp.vis_dir
        vis_dir = os.path.join(save_dir, 'vis_B_output')
        if not os.path.exists((vis_dir)):
            os.mkdir(vis_dir)

        tgt_pc = batch_data['tgt_pc'][i]  # (3, 1024)

        tgt_trans = pred_data['tgt_trans'][i].unsqueeze(1)  # (3)
        tgt_rot = pred_data['tgt_rot'][i]  # (6)

        device = tgt_pc.device
        num_points = tgt_pc.shape[1]
        tgt_rot_mat = bgs(tgt_rot.reshape(-1, 2, 3).permute(0, 2, 1)).reshape(3, 3)
  
        tgt_rot_mat = torch.linalg.inv(tgt_rot_mat)

        tgt_trans = tgt_trans.expand(-1, num_points)  # (3, 1024)

        tgt_pc = torch.matmul(tgt_rot_mat.double(), tgt_pc.double())  # (3, 1024)

        tgt_pc = tgt_pc + tgt_trans

        total_pc = np.array(tgt_pc.detach().cpu())
        color_mask_tgt = ['g'] * num_points
        total_color = color_mask_tgt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(list(total_pc[0]), list(total_pc[1]), list(total_pc[2]), c=total_color, s=10, alpha=0.9)
        ax.axis('scaled')
        ax.set_zlabel('Z', fontdict={'size': 20, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 20, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})

        fig.savefig(os.path.join(vis_dir, 'batch_{}_{}_output'.format(iter_counts,i)))
        plt.close(fig)


def debug_vis_gt(batch_data, cfg, pred_data, iter_counts):

    for i in range(cfg.exp.batch_size):
        save_dir = cfg.exp.vis_dir
        vis_dir = os.path.join(save_dir, 'vis_B_gt')
        if not os.path.exists((vis_dir)):
            os.mkdir(vis_dir)
        tgt_pc = batch_data['tgt_pc'][i]  # (3, 1024)
        tgt_trans = batch_data['tgt_trans'][i]  # (3, 1)
        tgt_rot = batch_data['tgt_rot'][i]  # (4)
        device = tgt_pc.device
        num_points = tgt_pc.shape[1]

        tgt_rot_mat = bgs(tgt_rot.reshape(-1, 2, 3).permute(0, 2, 1)).reshape(3, 3)
        tgt_rot_mat = torch.linalg.inv(tgt_rot_mat)

        tgt_trans = tgt_trans.expand(-1, num_points)  # (3, 1024)

        tgt_pc = torch.matmul(tgt_rot_mat, tgt_pc)  # (3, 1024)

        tgt_pc = tgt_pc + tgt_trans

        total_pc = np.array(tgt_pc.detach().cpu())
        color_mask_tgt = ['g'] * num_points
        total_color = color_mask_tgt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(list(total_pc[0]), list(total_pc[1]), list(total_pc[2]), c=total_color, s=10, alpha=0.9)
        ax.axis('scaled')
        ax.set_zlabel('Z', fontdict={'size': 20, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 20, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})

        fig.savefig(os.path.join(vis_dir, 'batch_{}_{}_gt'.format(iter_counts,i)))
        plt.close(fig)


class ShapeAssemblyNet_B_vnn(nn.Module):

    def __init__(self, cfg, data_features):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.init_encoder()

        self.encoder_dgcnn = DGCNN_New(feat_dim=cfg.model.pc_feat_dim)

        self.pose_predictor_rot = self.init_pose_predictor_rot()
        self.pose_predictor_trans = self.init_pose_predictor_trans()
        if self.cfg.model.recon_loss:
            self.decoder = self.init_decoder()
        self.data_features = data_features
        
        self.iter_counts = 0
        self.close_eps = 0.1
        self.L2 = nn.MSELoss() 
        self.R = torch.tensor([[0.26726124, -0.57735027,  0.77151675],
                  [0.53452248, -0.57735027, -0.6172134],
                  [0.80178373,  0.57735027,  0.15430335]], dtype=torch.float64).unsqueeze(0)
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()

        self.mlp_color = nn.Sequential(
            nn.Linear(512*2*3, 1024))


    def init_encoder(self):
        if self.cfg.model.encoderB == 'dgcnn':
            encoder = DGCNN(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoderB == 'vn_dgcnn':
            encoder = VN_DGCNN_New(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.model.encoderB == 'pointnet':
            encoder = PointNet(feat_dim=self.cfg.model.pc_feat_dim)
        return encoder

    def init_pose_predictor_rot(self):
        if self.cfg.model.encoderB == 'vn_dgcnn':
            pc_feat_dim = self.cfg.model.pc_feat_dim * 2 * 3
        if self.cfg.model.pose_predictor_rot == 'original':
            pose_predictor_rot = Regressor_CR(pc_feat_dim= pc_feat_dim, out_dim=6)
        elif self.cfg.model.pose_predictor_rot == 'vn':
            pose_predictor_rot = VN_equ_Regressor(pc_feat_dim= pc_feat_dim/3, out_dim=6)

        return pose_predictor_rot

    def init_pose_predictor_trans(self):
        if self.cfg.model.encoderB == 'vn_dgcnn':
            pc_feat_dim = self.cfg.model.pc_feat_dim * 2 * 3
        if self.cfg.model.pose_predictor_trans == 'original':
            pose_predictor_trans = Regressor_CR(pc_feat_dim=pc_feat_dim, out_dim=3)
        elif self.cfg.model.pose_predictor_trans == 'vn':
            pose_predictor_trans = VN_inv_Regressor(pc_feat_dim=pc_feat_dim/3, out_dim=3)
        return pose_predictor_trans

    def init_decoder(self):
        pc_feat_dim = self.cfg.model.pc_feat_dim
        decoder = MLPDecoder(feat_dim=pc_feat_dim, num_points=self.cfg.data.num_pc_points)
        return decoder


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        return optimizer


    def check_equiv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(torch.matmul(x, R) - xR))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return
    
    def check_inv(self, x, R, xR, name):
        mean_diff = torch.mean(torch.abs(x - xR))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return

    def check_network_property(self, gt_data, pred_data):
        with torch.no_grad():
            B, _, N = gt_data['src_pc'].shape
            R = self.R.float().repeat(B, 1, 1).to(gt_data['src_pc'].device)
            pcs_R = torch.matmul(gt_data['src_pc'].permute(0, 2, 1), R).permute(0, 2, 1)
            pred_data_R = self.forward(pcs_R, gt_data['tgt_pc'])

            equiv_feats = pred_data['Fa']
            equiv_feats_R = pred_data_R['Fa']
            self.check_equiv(equiv_feats, R, equiv_feats_R, 'equiv_feats')

            inv_feats = pred_data['Ga']
            inv_feats_R = pred_data_R['Ga']
            self.check_inv(inv_feats, R, inv_feats_R, 'inv_feats')

            if self.cfg.model.pose_predictor_rot == 'vn':
                rot = bgs(pred_data['src_rot'].reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
                rot_R = bgs(pred_data_R['src_rot'].reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
                self.check_equiv(rot, R, rot_R, 'rot')
        return

    def _recon_pts(self, Ga, Gb):
        global_inv_feat = torch.sum(torch.cat([Ga, Gb], dim=1), dim=1)
        recon_pts = self.decoder(global_inv_feat)
        return recon_pts

    def forward(self, tgt_pc):
        batch_size = tgt_pc.shape[0]
        num_points = tgt_pc.shape[2]
        if self.cfg.model.encoderB == 'dgcnn':
            tgt_point_feat = self.encoder(tgt_pc)  # (batch_size, pc_feat_dim(512), num_point(1024))

            tgt_feat = torch.mean(tgt_point_feat, dim=2)

        if self.cfg.model.encoderB == 'vn_dgcnn':
            Fb, Gb = self.encoder(tgt_pc)
            device = tgt_pc.device

            tgt_feat = Fb
            
        if self.cfg.model.pose_predictor_rot == 'original':
            tgt_rot = self.pose_predictor_rot(tgt_feat.reshape(batch_size, -1))
        else:
            tgt_rot = self.pose_predictor_rot(tgt_feat)
        
        if self.cfg.model.pose_predictor_trans == 'original':
            tgt_trans = self.pose_predictor_trans(tgt_feat.reshape(batch_size, -1))
        else:
            tgt_trans = self.pose_predictor_trans(tgt_feat)
        
        pred_dict = {
            'tgt_rot': tgt_rot,
            'tgt_trans': tgt_trans,
        }
        return pred_dict

    def compute_point_loss(self, batch_data, pred_data):
        tgt_pc = batch_data['tgt_pc'].float()  # batch x 3 x 1024
        tgt_rot_gt = self.recover_R_from_6d(batch_data['tgt_rot'].float())
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1
        tgt_rot_pred = self.recover_R_from_6d(pred_data['tgt_rot'].float())
        tgt_trans_pred = pred_data['tgt_trans'].float()

        tgt_trans_pred = tgt_trans_pred.unsqueeze(2)

        # Target point loss
        transformed_tgt_pc_pred = tgt_rot_gt @ tgt_pc + tgt_trans_pred  # batch x 3 x 1024
        with torch.no_grad():
            transformed_tgt_pc_gt = tgt_rot_pred @ tgt_pc + tgt_trans_gt  # batch x 3 x 1024
        tgt_point_loss = torch.mean(torch.sum((transformed_tgt_pc_pred - transformed_tgt_pc_gt) ** 2, axis=1))

        # Point loss
        point_loss =  tgt_point_loss
        return point_loss

    def compute_trans_loss(self, batch_data, pred_data):
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1ssssss
        tgt_trans_pred = pred_data['tgt_trans']  # batch x 3 x 1
        tgt_trans_pred = tgt_trans_pred.unsqueeze(dim=2)
        tgt_trans_loss = F.l1_loss(tgt_trans_pred, tgt_trans_gt)
        trans_loss = tgt_trans_loss
        return trans_loss

    def compute_rot_loss(self, batch_data, pred_data):
        tgt_R_6d = batch_data['tgt_rot']
        tgt_R_6d_pred = pred_data['tgt_rot']
        tgt_rot_loss = torch.mean(utils.get_6d_rot_loss(tgt_R_6d, tgt_R_6d_pred))
        rot_loss = tgt_rot_loss
        return rot_loss

    def compute_rot_loss_symmetry(self, batch_data, pred_data, device):
        
        partB_symmetry_type = batch_data['partB_symmetry_type']

        tgt_R_6d = batch_data['tgt_rot']

        tgt_R_6d_pred = pred_data['tgt_rot']
        tgt_rot_loss = torch.mean(utils.get_6d_rot_loss_symmetry(tgt_R_6d, tgt_R_6d_pred, partB_symmetry_type, device))

        rot_loss = tgt_rot_loss
        return rot_loss

    def compute_recon_loss(self, batch_data, pred_data):

        recon_pts = pred_data['recon_pts'] # batch x 1024 x 3

        src_pc = batch_data['src_pc'].float()  # batch x 3 x 1024
        tgt_pc = batch_data['tgt_pc'].float()  # batch x 3 x 1024
        src_quat_gt = batch_data['src_rot'].float()
        tgt_quat_gt = batch_data['tgt_rot'].float()

        src_Rs = utils.bgs(src_quat_gt.reshape(-1, 2, 3).permute(0, 2, 1)) # batch x 3 x 3
        tgt_Rs = utils.bgs(tgt_quat_gt.reshape(-1, 2, 3).permute(0, 2, 1)) # batch x 3 x 3

        src_trans_gt = batch_data['src_trans'].float()  # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1
        with torch.no_grad():
            transformed_src_pc_gt = src_Rs @ src_pc + src_trans_gt  # batch x 3 x 1024
            transformed_tgt_pc_gt = tgt_Rs @ tgt_pc + tgt_trans_gt  # batch x 3 x 1024
        gt_pts = torch.cat([transformed_src_pc_gt, transformed_tgt_pc_gt], dim=2).permute(0, 2, 1) # batch x 2048 x 3
        self.chamLoss = dist_chamfer_3D.chamfer_3DDist()
        dist1, dist2, idx1, idx2 = self.chamLoss(gt_pts, recon_pts)
        recon_loss = torch.mean(dist1) + torch.mean(dist2)
  
        return recon_loss

    def recover_R_from_6d(self, R_6d):
        R = utils.bgs(R_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        return R

    def quat_to_eular(self, quat):
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        r = R.from_quat(quat)
        euler0 = r.as_euler('xyz', degrees=True)

        return euler0

    def training_step(self, batch_data, device, batch_idx):
        self.iter_counts += 1
        partB_position, partB_rotation, total_loss, point_loss, rot_loss, trans_loss, recon_loss = self.forward_pass(batch_data, device, mode='train')
        return {"total_loss": total_loss,
                "point_loss": point_loss,
                "rot_loss": rot_loss,
                "trans_loss": trans_loss,
                "recon_loss": recon_loss,
                'predicted_partB_position': partB_position,
                'predicted_partB_rotation': partB_rotation
                }


    def calculate_metrics(self, batch_data, pred_data, device, mode):
        GD = self.compute_rot_loss(batch_data, pred_data)

        rot_error = self.compute_rot_loss(batch_data, pred_data)

        tgt_pc = batch_data['tgt_pc'].float()  # batch x 3 x 1024
        tgt_quat_gt = batch_data['tgt_rot'].float()

        tgt_Rs = utils.bgs(tgt_quat_gt.reshape(-1, 2, 3).permute(0, 2, 1)) # batch x 3 x 3

        tgt_trans_gt = batch_data['tgt_trans'].float()  # batch x 3 x 1
        with torch.no_grad():
            transformed_tgt_pc_gt = tgt_Rs @ tgt_pc + tgt_trans_gt  # batch x 3 x 1024
        gt_pts = transformed_tgt_pc_gt.permute(0, 2, 1) # batch x 1024 x 3

        pred_R_tgt = self.recover_R_from_6d(pred_data['tgt_rot'])
        pred_t_tgt = pred_data['tgt_trans'].view(-1, 3, 1)

        gt_euler_tgt = pytorch3d.transforms.matrix_to_euler_angles(tgt_Rs, convention="XYZ")

        pred_euler_tgt = pytorch3d.transforms.matrix_to_euler_angles(pred_R_tgt, convention="XYZ")

        with torch.no_grad():
            transformed_tgt_pc_pred = pred_R_tgt @ tgt_pc + pred_t_tgt  # batch x 3 x 1024

        recon_pts = transformed_tgt_pc_pred.permute(0, 2, 1) # batch x 2048 x 3

        dist1, dist2, idx1, idx2 = self.chamLoss(gt_pts, recon_pts)
        PA = torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1)

        thre = 0.0001
        acc = (PA < thre)
        PA_threshold = acc.sum(-1) / acc.shape[0]

        RMSE_T_2 = (pred_t_tgt - tgt_trans_gt).pow(2).mean(dim=-1) ** 0.5
        RMSE_T = RMSE_T_2

        dist_b1, dist_b2, idx_b1, idx_b2 = self.chamLoss(transformed_tgt_pc_gt.permute(0,2,1), transformed_tgt_pc_pred.permute(0,2,1))
        CD_2 = torch.mean(dist_b1, dim=-1) + torch.mean(dist_b2, dim=-1)

        return GD, rot_error, RMSE_T, PA_threshold, PA, CD_2

    def forward_pass(self, batch_data, device, mode, vis_idx=-1):

        pred_data = self.forward(batch_data['tgt_pc'].float())
        if self.cfg.model.point_loss:
            point_loss = self.compute_point_loss(batch_data, pred_data)
        else:
            point_loss = 0.0

        rot_loss = self.compute_rot_loss(batch_data, pred_data)
        trans_loss = self.compute_trans_loss(batch_data, pred_data)
        recon_loss = 0.0

        if vis_idx > -1:
            debug_vis_input(batch_data, self.cfg, pred_data, vis_idx)
            debug_vis_output(batch_data, self.cfg, pred_data, vis_idx)
            debug_vis_gt(batch_data, self.cfg, pred_data, vis_idx)

        total_loss = point_loss + rot_loss + trans_loss + recon_loss

        if mode == 'val':
            return (self.calculate_metrics(batch_data, pred_data, device, mode), pred_data['tgt_trans'], pred_data['tgt_rot'], total_loss, point_loss,rot_loss,trans_loss,recon_loss)

        # Total loss
        return pred_data['tgt_trans'], pred_data['tgt_rot'],total_loss, point_loss, rot_loss, trans_loss, recon_loss

