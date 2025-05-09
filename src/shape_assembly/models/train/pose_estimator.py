import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, out_channels=(32, 64, 128), train_with_norm=True):
        super(PointNet, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 3  
        for out_channel in out_channels:
            self.layers.append(nn.Conv1d(in_channels, out_channel, 1))
            self.layers.append(nn.BatchNorm1d(out_channel) if train_with_norm else nn.Identity())
            self.layers.append(nn.ReLU())
            in_channels = out_channel
        self.global_pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x

class PoseClassifier(nn.Module):
    def __init__(self, pointnet_out_dim=128, pose_dim=6, hidden_dims=(512, 256, 128)):
        super(PoseClassifier, self).__init__()
        self.pointnet = PointNet(out_channels=(32, 64, pointnet_out_dim))
        input_dim = pointnet_out_dim + pose_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))  
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, point_cloud, poses):
        # Point cloud feature extraction
        point_cloud_features = self.pointnet(point_cloud)  # (batch_size, pointnet_out_dim)
        
        # Repeat point cloud features for each pose
        repeated_features = point_cloud_features.unsqueeze(1).repeat(1, poses.size(1), 1)  # (batch_size, num_poses, pointnet_out_dim)
        
        # Concatenate pose features with point cloud features
        combined_features = torch.cat((repeated_features, poses), dim=-1)  # (batch_size, num_poses, pointnet_out_dim + pose_dim)
        
        # Flatten the input for the classifier
        combined_features = combined_features.view(-1, combined_features.size(-1))  # (batch_size * num_poses, pointnet_out_dim + pose_dim)
        
        # Classification
        scores = self.classifier(combined_features)  # (batch_size * num_poses, 1)
        scores = scores.view(-1, poses.size(1))  # (batch_size, num_poses)
        
        return scores


batch_size = 8
num_points = 1024
num_poses = 1024
pose_dim = 6

point_cloud = torch.rand(batch_size, 3, num_points)  
poses = torch.rand(batch_size, num_poses, pose_dim)  

model = PoseClassifier(pointnet_out_dim=128, pose_dim=pose_dim)
scores = model(point_cloud, poses) 
print(scores)
print(scores.size())  
