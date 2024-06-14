import torch.nn as nn
# from models.pointnet_classifier import PointNetClassifier
from WayTu_Model.models.pointnet_classifier import PointNetClassifier
import torch
import torch.nn.functional as F
import numpy as np
from learning3d.models import PointNet

class WayTuBackbone(nn.Module):
    def __init__(self, feature_extractor_model_path, num_points, num_waypoints):
        super(WayTuBackbone, self).__init__()

        self.num_waypoints = num_waypoints

        # Load the pretrained feature extractor
        classifier = PointNetClassifier(num_points, 3).eval().cuda().double()
        classifier.load_state_dict(torch.load(feature_extractor_model_path))
        # Discard the classifier head and get the feature extractor as PointNet base 
        self.PointNet_base = classifier.base

        # Freeze initial layers of PointNet
        for param in self.PointNet_base.parameters():
            param.requires_grad = False
        
        for param in self.PointNet_base.mlp2.parameters():
            param.requires_grad = True
        
        self.attention = SelfAttention(1024,128)
        self.attention2 = SelfAttention(2048,128)

        self.fc1 = self.xavier_initialization(1024, 4096)
        self.fc2 = self.xavier_initialization(4096, 2048)
        self.fc3 = self.xavier_initialization(2048, 1024)
        self.fc4 = self.xavier_initialization(1024, 512)
        self.fc5 = self.xavier_initialization(512, 256)


        self.norm1 = nn.LayerNorm(4096)
        self.norm2 = nn.LayerNorm(2048)
        self.norm3 = nn.LayerNorm(1024)
        self.norm4 = nn.LayerNorm(512)


        self.res_fc2 = nn.Linear(1024, 2048)
        self.res_fc4 = nn.Linear(1024, 512)
    
    def forward(self, point_cloud):
        point_cloud = torch.permute(point_cloud, (0, 2, 1))
        x, _, _ = self.PointNet_base(point_cloud)

        input = x.to(torch.float32)

        x = self.attention(input)
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = F.relu(self.fc2(x))

        res_x = self.res_fc2(input)
        x = x + res_x

        x= self.attention2(x)

        x = self.norm2(x)
        x = F.relu(self.fc3(x))
        x = self.norm3(x)
        x = F.relu(self.fc4(x))

        

        res_x = self.res_fc4(input)
        x = x + res_x

        x = self.norm4(x)
        x = self.fc5(x)

        return x 

    def xavier_initialization(self, layer_in, layer_out):
        layer = nn.Linear(layer_in,layer_out)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

        return layer
    
class WayTuGeneratorHead(nn.Module):
    def __init__(self, num_waypoints):
        super(WayTuGeneratorHead, self).__init__()

        self.num_waypoints = num_waypoints

        self.fc5 = self.xavier_initialization(256, 1024)
        self.fc6 = self.xavier_initialization(1024, 512)
        self.fc7 = self.xavier_initialization(512, 128)
        self.fc8 = self.xavier_initialization(128, num_waypoints*7)

        self.norm4 = nn.LayerNorm(256)
        self.norm5 = nn.LayerNorm(1024)
        self.norm6 = nn.LayerNorm(512)
        self.norm7 = nn.LayerNorm(128)

    def forward(self, x):
        x = self.norm4(x)
        x = F.relu(self.fc5(x))
        x = self.norm5(x)
        x = F.relu(self.fc6(x))
        x = self.norm6(x)
        x = F.relu(self.fc7(x))
        x = self.norm7(x)
        x = self.fc8(x)

        x = x.view(-1, self.num_waypoints, 7) #.contiguous()
        positions = x[:, :, :3]
        quaternions = x[:, :, 3:]

        normalized_quaternions = F.normalize(quaternions, p=2, dim=-1)
        return positions, normalized_quaternions
    
    def xavier_initialization(self, layer_in, layer_out):
        layer = nn.Linear(layer_in,layer_out)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

        return layer
    

class WayTuSelectionHead(nn.Module):
    def __init__(self, num_waypoints):
        super(WayTuSelectionHead, self).__init__()

        self.num_waypoints = num_waypoints

        self.fc5 = self.xavier_initialization(256 + 7*num_waypoints, 128)
        self.fc6 = self.xavier_initialization(128, 1)

        self.bn4 = nn.BatchNorm1d(256+ 7*num_waypoints)
        self.bn5 = nn.BatchNorm1d(128)
    
    def forward(self, x, waypoint_positions, waypoint_quaternions):
        waypoints = torch.cat((waypoint_positions, waypoint_quaternions), dim=-1).view(waypoint_positions.size(0), -1)
        x = torch.cat((x, waypoints), 1)

        x = x.to(torch.float32)
        x.cuda()

        x = self.bn4(x)
        x = F.gelu(self.fc5(x))
        x=self.bn5(x)
        x = self.fc6(x)

        return x

    def xavier_initialization(self, layer_in, layer_out):
        layer = nn.Linear(layer_in,layer_out)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

        return layer
    
class WayTuModel(nn.Module):
    def __init__(self, feature_extractor_model_path, num_points, num_waypoints):
        super(WayTuModel, self).__init__()

        self.num_waypoints = num_waypoints

        self.backbone = WayTuBackbone(feature_extractor_model_path,num_points,num_waypoints)
        self.generator = WayTuGeneratorHead(num_waypoints)
        self.selection = WayTuSelectionHead(num_waypoints)
    
    def forward(self, point_clod):
        x = self.backbone(point_clod)
        pose, qua = self.generator(x)
        score = self.selection(x,pose,qua)

        return pose, qua, score


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, attention_dim)
        self.key_layer = nn.Linear(input_dim, attention_dim)
        self.value_layer = nn.Linear(input_dim, attention_dim)
        self.output_layer = nn.Linear(attention_dim, input_dim)
    
    def forward(self, x): 
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (keys.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        context_vector = torch.matmul(attention_weights, values)    
        attention_output = self.output_layer(context_vector)
        return attention_output