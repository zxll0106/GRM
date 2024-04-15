from backbone.backbone import *
from utils import *
from roi_align.roi_align import RoIAlign

import torch
import torch.nn as nn
import torch.nn.functional as F

from infer_module.spatial_temporal_group_residual_module import Spatial_Temporal_Group_Residual_Module

class Model(nn.Module):

    def __init__(self,cfg,hidden_size):

        super(Model,self).__init__()

        self.cfg=cfg

        self.hidden_size=cfg.hidden_size

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        #
        self.images_embedding=nn.Sequential(nn.Linear(D*K*K,self.hidden_size),nn.ReLU())

        self.spatial_temporal_grms=Spatial_Temporal_Group_Residual_Module(v_len=self.hidden_size,sub_layers=cfg.num_layers,global_layers=cfg.num_temporal_layers)

        self.fc_acitivities=nn.Linear(self.hidden_size*(2**self.cfg.num_temporal_layers),self.cfg.num_activities)

        self.include_inter_group=True

        self.include_intra_group=True


    def forward(self,batch_data):
        images_in, boxes_in = batch_data

        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # batch_size,t,n,_=image_features.shape

        x_images = self.images_embedding(boxes_features)  # B,T,N,DIM

        x_features=self.spatial_temporal_grms(x_images)

        activities_scores=self.fc_acitivities(x_features).reshape(-1,self.cfg.num_activities)


        return {'activities': activities_scores}

class Model_collective(nn.Module):

    def __init__(self,cfg,hidden_size):

        super(Model_collective,self).__init__()

        self.cfg=cfg

        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.hidden_size=cfg.hidden_size

        self.images_embedding=nn.Sequential(nn.Linear(D*K*K,self.hidden_size),nn.ReLU())

        self.spatial_temporal_grms=Spatial_Temporal_Group_Residual_Module(v_len=self.hidden_size,sub_layers=cfg.num_layers,global_layers=cfg.num_temporal_layers)

        self.fc_acitivities = nn.Linear(self.hidden_size*(2**self.cfg.num_temporal_layers), self.cfg.num_activities)


    def forward(self,batch_data):

        images_in, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # batch_size,t,n,_=image_features.shape

        bboxes_num = bboxes_num_in[:, 0]

        x_images=self.images_embedding(boxes_features_all) #B,T,N,DIM

        x_features=[]

        for i in range(B):
            N = bboxes_num[i]
            x_images_b=x_images[i, :, :N, :].unsqueeze(0)
            x_features_b=self.spatial_temporal_grms(x_images_b)
            x_features.append(x_features_b)

        x_features=torch.cat(x_features,dim=0)
        x_features=x_features[:,-1,:]

        activities_scores=self.fc_acitivities(x_features).reshape(-1,self.cfg.num_activities)


        return activities_scores