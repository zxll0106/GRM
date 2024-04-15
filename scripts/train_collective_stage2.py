import sys
sys.path.append(".")
import os

from train_net_dynamic import *

cfg=Config('collective')

cfg.device_list="0"
cfg.training_stage=2
cfg.use_gpu = True
cfg.use_multi_gpu = False
cfg.train_backbone = False
cfg.load_backbone_stage2 = False

# ResNet18
# cfg.backbone = 'res18'
# cfg.image_size = 480, 720
# cfg.out_size = 15, 23
# cfg.emb_features = 512
# cfg.stage1_model_path = 'result/basemodel_CAD_res18.pth'

# VGG16
cfg.backbone = 'vgg16'
cfg.image_size = 480, 720
cfg.out_size = 15, 22
cfg.emb_features = 512
cfg.stage1_model_path = 'result/basemodel_CAD_vgg16.pth'

cfg.hidden_size=1024
cfg.num_layers=3
cfg.num_temporal_layers=1

cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10
cfg.num_graph = 4
cfg.tau_sqrt=True
cfg.batch_size = 2
cfg.test_batch_size = 32
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 1e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
cfg.max_epoch = 100
cfg.test_interval_epoch=1




cfg.exp_note='Dynamic_collective'
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    train_net(cfg)