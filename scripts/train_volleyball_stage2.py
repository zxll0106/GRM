import sys
sys.path.append(".")

import os
from train_net_dynamic import *
import torch

cfg=Config('volleyball')
cfg.inference_module_name = 'dynamic_tce_volleyball'

cfg.device_list = "1"
cfg.use_gpu = True
cfg.use_multi_gpu = False
cfg.training_stage = 2
cfg.train_backbone = False
cfg.test_interval_epoch = 1

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
cfg.out_size = 22, 40
cfg.emb_features = 512

# res18 setup
# cfg.backbone = 'res18'
# cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
# cfg.out_size = 23, 40
# cfg.emb_features = 512

cfg.hidden_size=1024
cfg.num_layers=1
cfg.num_temporal_layers=2

cfg.batch_size = 16
cfg.test_batch_size = 16
cfg.num_frames = 21
cfg.num_before=10
cfg.num_after=10
cfg.load_backbone_stage2 = False
cfg.train_learning_rate = 1e-5
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.max_epoch = 60
cfg.lr_plan = {11: 1e-5}
cfg.max_epoch = 100
cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

cfg.exp_note = 'Dynamic_TCE_Volleyball_stage2_vgg16'

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_net(cfg)
