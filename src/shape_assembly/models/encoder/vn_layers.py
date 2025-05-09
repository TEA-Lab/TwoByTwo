import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

def conv1x1(in_channels, out_channels, dim):
    if dim == 3:
        return nn.Conv1d(in_channels, out_channels, 1, bias=False)
    elif dim == 4:
        return nn.Conv2d(in_channels, out_channels, 1, bias=False)
    elif dim == 5:
        return nn.Conv3d(in_channels, out_channels, 1, bias=False)
    else:
        raise NotImplementedError(f'{dim}D 1x1 Conv is not supported')


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                    mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNNewLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNNewLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                mask * x + (1 - mask) * (x - (d / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                    mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm',
                 negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0


class VNInFeature(nn.Module):
    """VN-Invariant layer."""

    def __init__(
            self,
            in_channels,
            dim=4,
            share_nonlinearity=False,
            negative_slope=0.2,
            use_rmat=False,
    ):
        super().__init__()

        self.dim = dim
        self.use_rmat = use_rmat
        self.vn1 = VNLinearBNLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearBNLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn_lin = conv1x1(
            in_channels // 4, 2 if self.use_rmat else 3, dim=dim)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]
        Returns:
            rotation invariant features of the same shape
        """
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z)  # [B, 3, 3, N] or [B, 2, 3, N]
        if self.use_rmat:
            z = z.flatten(1, 2).transpose(1, 2).contiguous()  # [B, N, 6]
            z = rot6d_to_matrix(z)  # [B, N, 3, 3]
            z = z.permute(0, 2, 3, 1)  # [B, 3, 3, N]
        z = z.transpose(1, 2).contiguous()

        if self.dim == 4:
            x_in = torch.einsum('bijm,bjkm->bikm', x, z)
        elif self.dim == 3:
            x_in = torch.einsum('bij,bjk->bik', x, z)
        elif self.dim == 5:
            x_in = torch.einsum('bijmn,bjkmn->bikmn', x, z)
        else:
            raise NotImplementedError(f'dim={self.dim} is not supported')

        return x_in


class VNLinearBNLeakyReLU(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            dim=5,
            share_nonlinearity=False,
            negative_slope=0.2,
    ):
        super().__init__()

        self.linear = VNLinear(in_channels, out_channels)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        self.leaky_relu = VNLeakyReLU(
            out_channels,
            # dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )

    def forward(self, x):
        # Linear
        p = self.linear(x)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        p = self.leaky_relu(p)
        return p
    
class NonEquivariantLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2, use_batchnorm=True, negative_slope=0.2):
        super(NonEquivariantLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.use_batchnorm = use_batchnorm

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        # Linear
        print("PRE x.shape is, ", x.shape)
        # [todo]
        x = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        x = self.map_to_feat(x)
        
        # BatchNorm
        if self.use_batchnorm:
            x = x.transpose(1, -1).contiguous()
            x = self.batchnorm(x)
            x = x.transpose(1, -1).contiguous()
        
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class NonEquivariantMaxPool(nn.Module):
    def __init__(self, dim=-1):
        super(NonEquivariantMaxPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        '''
        x: point features of shape [B, C, N]
        '''
        return torch.max(x, dim=self.dim, keepdim=True)[0]


class NonEquivariantStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, negative_slope=0.2):
        super(NonEquivariantStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.fc1 = NonEquivariantLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, negative_slope=negative_slope)
        self.fc2 = NonEquivariantLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, negative_slope=negative_slope)

    def forward(self, x):
        '''
        x: point features of shape [B, C_in, N]
        '''
        z0 = x
        z0 = self.fc1(z0)
        z0 = self.fc2(z0)
        
        # No need for orthogonalization or frame normalization in non-equivariant version
        return z0



class NonEquivariantLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=4, use_batchnorm=True, negative_slope=0.2):
        super(NonEquivariantLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        '''
        x: point features of shape [B, C_in, N]
        '''
        # Linear
        x = self.linear(x)
        # BatchNorm
        if self.use_batchnorm:
            x = self.batchnorm(x)
        # LeakyReLU
        x = self.leaky_relu(x)
        return x