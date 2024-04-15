import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.use_multi_gpu = False
cfg.device_list="0"
cfg.training_stage=1
cfg.stage1_model_path='/extend/zxl/Group_Activity_Prediction/DIN-Group-Activity-Recognition-Benchmark-main/result/[Volleyball_stage1_stage1]<2022-09-06_21-02-52>/stage1_epoch65_92.22%.pth'
cfg.train_backbone=True
cfg.test_before_train = True

# VGG16
cfg.backbone = 'vgg16'
cfg.image_size = 720, 1280
cfg.out_size = 22, 40
cfg.emb_features = 512

cfg.num_before = 5
cfg.num_after = 4

cfg.batch_size=8
cfg.test_batch_size=1
cfg.num_frames=1
# cfg.train_learning_rate=1e-5
# cfg.lr_plan={}
# cfg.max_epoch=200
cfg.train_learning_rate=1e-4
cfg.lr_plan={30:5e-5, 60:2e-5, 90:1e-5}
cfg.max_epoch=120
cfg.set_bn_eval = False
cfg.actions_weights=[[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]  

cfg.exp_note='Volleyball_stage1'
train_net(cfg)
