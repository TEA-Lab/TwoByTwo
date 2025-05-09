import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn.functional as F

def render_pts_label_png(fn, pc, color):
    # pc: (num_points, 3), color: (num_points,)
    new_color = []
    for i in range(len(color)):
        if color[i] == 1:
            new_color.append('#ab4700')
        else:
            new_color.append('#00479e')
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('point cloud')

    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=new_color, marker='.', s=5, linewidth=0, alpha=1)
    ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.savefig(fn+'.png')

def compute_distance_between_rotations(P,Q):
    #! input two rotation matrices, output the distance between them (unit is rad)
    #! P,Q are 3x3 numpy arrays
    P = np.asarray(P)
    Q = np.asarray(Q)
    R = np.matmul(P,Q.swapaxes(1,2))
    theta = np.arccos(np.clip((np.trace(R,axis1 = 1,axis2 = 2) - 1)/2,-1,1))
    return np.mean(theta)

def bgs(d6s):
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)


def bgdR(Rgts, Rps):
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    # print("\033[33m the theta in bgdR is", theta,"\033[0m")
    return torch.acos(theta)

def get_6d_rot_loss(gt_6d, pred_6d):
    #input pred_6d , gt_6d : batch * 2 * 3
    pred_Rs = bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_Rs = bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    theta = bgdR(gt_Rs, pred_Rs)
    theta_degree = theta * 180 / np.pi
    # print("\033[33m the theta is", theta,"\033[0m")
    return theta_degree

def get_6d_rot_loss_symmetry_new(batch_data, pred_data, device):

    batch_size = batch_data['src_rot'].shape[0]

    partA_symmetry_type = batch_data['partA_symmetry_type']
    partB_symmetry_type = batch_data['partB_symmetry_type']

    src_R_6d = batch_data['src_rot']
    tgt_R_6d = batch_data['tgt_rot']

    src_R_6d_pred = pred_data['src_rot']
    tgt_R_6d_pred = pred_data['tgt_rot']
    tgt_R_6d_init = batch_data['predicted_partB_rotation']

    src_Rs = bgs(src_R_6d_pred.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_src_Rs = bgs(src_R_6d.reshape(-1, 2, 3).permute(0, 2, 1))

    tgt_Rs = bgs(tgt_R_6d_pred.reshape(-1, 2, 3).permute(0, 2, 1))
    tgt_Rs_init = bgs(tgt_R_6d_init.reshape(-1, 2, 3).permute(0, 2, 1))

    tgt_Rs_new = torch.matmul(tgt_Rs, tgt_Rs_init)
    gt_tgt_Rs = bgs(tgt_R_6d.reshape(-1, 2, 3).permute(0, 2, 1))


    R1 = src_Rs / torch.pow(torch.det(src_Rs), 1/3).view(-1,1,1)
    R2 = gt_src_Rs / torch.pow(torch.det(gt_src_Rs), 1/3).view(-1,1,1)

    R3 = tgt_Rs_new/ torch.pow(torch.det(tgt_Rs_init), 1/3).view(-1,1,1)
    R4 = gt_tgt_Rs / torch.pow(torch.det(tgt_Rs), 1/3).view(-1,1,1)

    # R5 = gt_tgt_Rs / torch.pow(torch.det(gt_tgt_Rs), 1/3).view(-1,1,1)

    # cos_theta = torch.zeros(batch_size)
    z = torch.tensor([0.0,0.0,1.0], device=device).unsqueeze(0).repeat(batch_size,1)
    cos_theta = torch.zeros(batch_size)
    for i in range(batch_size):
        # for every data in batch_size
        symmetry_i = partA_symmetry_type[i]

        # if the data is z-axis symmetric
        if symmetry_i[4].item() == 1:               
            # cosidering symmetry when rotating around z-axis
            z1 = torch.matmul(R1[i], z[i])
            z2 = torch.matmul(R2[i], z[i])

            cos_theta[i] = torch.dot(z1,z2) / (torch.norm(z1) * torch.norm(z2))
            
        else:
            R_A = torch.matmul(R1[i], R2[i].transpose(1,0))
            cos_theta[i] = (torch.trace(R_A) - 1) /2

    theta_src = torch.acos(torch.clamp(cos_theta, -1.0+1e-6, 1.0-1e-6))*180 / np.pi

    cos_theta_B = torch.zeros(batch_size)
    for i in range(batch_size):
        # for every data in batch_size
        symmetry_i = partB_symmetry_type[i]

        # if the data is z-axis symmetric
        if symmetry_i[4].item() == 1:               
            # cosidering symmetry when rotating around z-axis
            
            # z4 = torch.matmul(R4[i], z[i])
            z3 = torch.matmul(R3[i], z[i])
            z4 = torch.matmul(R4[i], z[i])

            cos_theta_B[i] = torch.dot(z3,z4) / (torch.norm(z3) * torch.norm(z4))
            
        else:
            R_B = torch.matmul(R3[i], R4[i].transpose(1,0))
            cos_theta_B[i] = (torch.trace(R_B) - 1) /2

    theta_tgt = torch.acos(torch.clamp(cos_theta_B, -1.0+1e-6, 1.0-1e-6))*180 / np.pi

    src_rot_loss = torch.mean(theta_src)
    tgt_rot_loss = torch.mean(theta_tgt)

    # rot_loss = (src_rot_loss + tgt_rot_loss)/2.0

    return (src_rot_loss, tgt_rot_loss)

def get_6d_rot_loss_symmetry(gt_6d, pred_6d, symmetry, device):
    batch_size = gt_6d.shape[0]

    pred_Rs = bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_Rs = bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    R1 = pred_Rs / torch.pow(torch.det(pred_Rs), 1/3).view(-1,1,1)
    R2 = gt_Rs / torch.pow(torch.det(gt_Rs), 1/3).view(-1,1,1)

    # cos_theta = torch.zeros(batch_size)
    z = torch.tensor([0.0,0.0,1.0], device=device).unsqueeze(0).repeat(batch_size,1)
    cos_theta = torch.zeros(batch_size)
    for i in range(batch_size):
        # for every data in batch_size
        symmetry_i = symmetry[i]

        # if the data is z-axis symmetric
        if symmetry_i[4].item() == 1:               
            # cosidering symmetry when rotating around z-axis
            z1 = torch.matmul(R1[i], z[i])
            z2 = torch.matmul(R2[i], z[i])
            cos_theta[i] = torch.dot(z1,z2) / (torch.norm(z1) * torch.norm(z2))
        else:
            R = torch.matmul(R1[i], R2[i].transpose(1,0))
            cos_theta[i] = (torch.trace(R) - 1) /2

    theta = torch.acos(torch.clamp(cos_theta, -1.0+1e-6, 1.0-1e-6))*180 / np.pi
    return theta
            

def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')
