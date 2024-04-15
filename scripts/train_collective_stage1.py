import sys
sys.path.append(".")

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from train_net import *

cfg=Config('collective')

cfg.device_list="1"
cfg.use_multi_gpu=False
cfg.training_stage=1
cfg.train_backbone=False

cfg.backbone = 'vgg16'
cfg.out_size = 22, 40
cfg.emb_features = 512
cfg.stage1_model_path='/extend/zxl/Group_Activity_Prediction/DIN-Group-Activity-Recognition-Benchmark-main/result/[Collective_stage1_stage1]<2022-09-09_15-59-24>/stage1_epoch69_87.80%.pth'

cfg.image_size=480, 720
# cfg.out_size=57,87
cfg.num_boxes=13
cfg.num_actions=6
cfg.num_activities=5
cfg.num_frames=10

cfg.batch_size=16
cfg.test_batch_size=8 
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=100

cfg.test_before_train=False

cfg.exp_note='Collective_stage1'
train_net(cfg)