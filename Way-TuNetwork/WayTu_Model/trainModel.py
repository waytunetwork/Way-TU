
from WayTu_Model.WaytuModel import WayTuModel
from WayTu_Model.WayTu_Dataset import  WayTuDataset
import WayTu_Model.helpers as hlp

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


def end_to_end_model_io(cfg, dataset_root_path, feature_extractor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = cfg['e2e-batch']
    num_epoch = cfg['e2e-epoch']
    num_points = cfg['num-points']
    val_size = 0.2

    train_set, val_set,_ = hlp.get_all_samples(dataset_root_path, val_size)

    train_dataset = WayTuDataset(train_set,task=cfg['task'], normalize=cfg["e2e-normalize"])
    val_dataset = WayTuDataset(val_set, task=cfg['task'], normalize=cfg["e2e-normalize"])

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)
    val_dataloader  = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=4)

    model = WayTuModel(feature_extractor_model_path=feature_extractor, num_points= num_points, num_waypoints=3)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['e2e-lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    loss_dict = {"train_loss" : [], "val_loss" : []}
    best_val_loss = np.inf
    best_model = None

    for e in range(num_epoch):
        epoch_loss = 0
        model.train()
        for i, data in enumerate(train_dataloader):
            point_cloud, waypoint, score = data
            waypoint = waypoint.to(torch.float32)

            waypoint = waypoint.view(-1, 3, 7) # .contiguous()
            waypoint_positions = waypoint[:, :, :3].to(device) 
            waypoint_quaternions = waypoint[:, :, 3:].to(device)

            score = score.to(torch.float32).to(device)
            score = torch.unsqueeze(score,1)

            optimizer.zero_grad()

            pre_way_pos, pre_way_qua, pre_score = model(point_cloud.to(device))

            position_loss = F.mse_loss(pre_way_pos, waypoint_positions)
            quaternion_loss = quaternion_geodesic_loss(pre_way_qua, waypoint_quaternions)
            score_loss =F.mse_loss(pre_score, score)


            loss = 0.60 * position_loss + 0.30 * quaternion_loss + 0.1 * score_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss = epoch_loss/len(train_dataloader)

        val_epoch_loss = 0
        model.eval()
        for i,val_data in enumerate(val_dataloader):
            val_point_cloud, val_waypoint, val_score = val_data
            val_waypoint = val_waypoint.to(torch.float32)

            val_waypoint =val_waypoint.view(-1, 3, 7)
            val_waypoint_positions = val_waypoint[:, :, :3].to(device)
            val_waypoint_quaternions = val_waypoint[:, :, 3:].to(device)

            val_score = val_score.to(torch.float32).to(device)
            val_score = torch.unsqueeze(val_score,1)


            val_way_pos, val_way_qua, val_pre_score = model(val_point_cloud.to(device))

            val_position_loss = F.mse_loss(val_way_pos, val_waypoint_positions)
            val_quaternion_loss = quaternion_geodesic_loss(val_way_qua, val_waypoint_quaternions)
            val_score_loss =F.mse_loss(val_pre_score, val_score)

            val_loss = 0.65 * val_position_loss + 0.2 * val_quaternion_loss +  0.15 * val_score_loss

            val_epoch_loss +=  val_loss.item() 
        
        val_epoch_loss = val_epoch_loss/len(val_dataloader)

        if val_epoch_loss <= best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = model.state_dict()
        
        loss_dict["train_loss"].append(epoch_loss)
        loss_dict["val_loss"].append(val_epoch_loss)
        print(f"epoch {e}: train loss:{epoch_loss} val loss:{val_epoch_loss}")

        scheduler.step()
    torch.save(model.state_dict(), cfg['last-model'])
    torch.save(best_model, cfg['e2e-model'])

    return loss_dict["train_loss"], loss_dict["val_loss"]


def weighted_mse_loss(prediction,response, weight):
    weight = weight.reshape([weight.shape[0],1])
    weight = weight.expand(-1,response.shape[1])
    return (weight * (response - prediction)**2).mean()


def quaternion_geodesic_loss(prediction, target):
    dot_product = (prediction * target).sum(dim=-1).abs()
    dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6) 
    loss = 2 * torch.acos(dot_product)
    return loss.mean()

def angular_loss(pred, target):
    pred = pred / torch.norm(pred, dim=1, keepdim=True)
    target = target / torch.norm(target, dim=1, keepdim=True)
    
    dot_product = torch.sum(pred * target, dim=1)
    angular_loss = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0)) / torch.pi

    return torch.mean(angular_loss)

def weighted_quaternion_geodesic_loss(prediction, target, weight):
    weight = weight.reshape([weight.shape[0],1])

    dot_product = (prediction * target).sum(dim=-1).abs()
    dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6) 
    loss = 2 * torch.acos(dot_product)

    return (weight*loss).mean()
