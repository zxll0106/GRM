import torch
import torch.optim as optim

import time
import random
import os
import sys
from thop import profile

from config import *
from volleyball import *
from collective import *
from dataset import *
from base_model import *
from GAP_model import *
from utils import *
from tqdm import tqdm

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)


    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 8, # 4,
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)


    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    basenet_list={'volleyball':Basenet_volleyball, 'collective':Basenet_collective}

    if cfg.training_stage==1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
    elif cfg.training_stage==2:
        # Load backbone
        if cfg.dataset_name=='volleyball':
            model=Model(cfg,hidden_size=1024)
        else:
            model=Model_collective(cfg,hidden_size=1024)
        if cfg.load_backbone_stage2:
            model.loadmodel(cfg.stage1_model_path)
        elif cfg.load_stage2model:
            # if cfg.use_multi_gpu:
            #     model = nn.DataParallel(model)
            state = torch.load(cfg.stage2model)
            model.load_state_dict(state['state_dict'])
            print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)
        else:
            print_log(cfg.log_path, 'Not loading stage1 or stage2 model.')
    else:
        assert(False)
    
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    train_list={'volleyball':train_volleyball, 'collective':train_collective}
    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    train=train_list[cfg.dataset_name]
    test=test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        # test_info=test_volleyball(validation_loader, model, device, 0, cfg)
        # show_epoch_info('Test', cfg.log_path, test_info)

        test_info=test_collective(validation_loader,model,device,0,cfg)
        show_epoch_info("Test",cfg.log_path,test_info)

        return

    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)
            
            if test_info['activities_acc']>best_result['activities_acc']:
                best_result=test_info
            print_log(cfg.log_path, 
                      'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
            
            # Save model
            if cfg.training_stage==2:
                # None
                # if test_info['activities_acc'] > 93.1:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                torch.save(state, filepath)
                print('model saved to:',filepath)
            elif cfg.training_stage==1:
                if test_info['activities_acc'] == best_result['activities_acc']:
                    for m in model.modules():
                        if isinstance(m, Basenet):
                            filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                            m.savemodel(filepath)
    #                         print('model saved to:',filepath)
            else:
                assert False
   
def train_volleyball(data_loader, model, device, optimizer, epoch, cfg):
    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    activities_meter_list=[AverageMeter() for i in range(10)]

    for batch_idx, batch_data in enumerate(tqdm(data_loader,desc='Train at Epoch '+str(epoch))):
        if batch_idx % 850 == 0 and batch_idx > 0:
            print('Training in processing {}/{}, group Activity Loss: {:.4f}'.format(batch_idx, len(data_loader), loss_meter.avg))

        model.train()
        if cfg.set_bn_eval:
            model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = batch_data[3].reshape((batch_size, num_frames))

        actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
        activities_in = activities_in[:, 0].reshape((batch_size,))

        # forward
        # actions_scores,activities_scores=model((batch_data[0],batch_data[1]))
        ret = model((batch_data[0], batch_data[1]))


        # Predict activities
        loss_list = []
        if 'activities' in list(ret.keys()):
            activities_scores = ret['activities']
            activities_loss = F.cross_entropy(activities_scores,activities_in)
            loss_list.append(activities_loss)
            activities_labels = torch.argmax(activities_scores,dim=1)
            activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_conf.add(activities_labels, activities_in)

            activities_labels=activities_labels.reshape(batch_size,num_frames,-1)
            activities_in=activities_in.reshape(batch_size,num_frames,-1)


            for i,activities_meter_i in enumerate(activities_meter_list):
                activities_labels_i=activities_labels[:,int(num_frames*(i+1)/10)-1]
                activities_in_i=activities_in[:,int(num_frames*(i+1)/10)-1]
                activities_correct_i = torch.sum(torch.eq(activities_labels_i.int(), activities_in_i.int()).float())
                activities_accuracy_i = activities_correct_i.item() / batch_size
                activities_meter_list[i].update(activities_accuracy_i,batch_size)


        if 'actions' in list(ret.keys()):
            # Predict actions
            actions_scores = ret['actions']
            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights) * cfg.actions_loss_weight
            loss_list.append(actions_loss)
            actions_labels = torch.argmax(actions_scores, dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            actions_meter.update(actions_accuracy, actions_scores.shape[0])

        if 'halting' in list(ret.keys()):
            loss_list.append(ret['halting']*cfg.halting_penalty)

        # print(loss_list)
        total_loss = sum(loss_list)
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        # Test max_clip_norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    activities_acc_list=[]
    for i in range(10):
        activities_acc_list.append(activities_meter_list[i].avg*100)
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf':activities_conf.value(),
        'activities_MPCA':MPCA(activities_conf.value()),
        'activities_acc_list':activities_acc_list,
    }
    
    return train_info
        
    
def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()
    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    epoch_timer=Timer()
    activities_meter_list = [AverageMeter() for i in range(10)]
    activities_conf_list=[ConfusionMeter(cfg.num_activities) for i in range(10)]

    with torch.no_grad():
        for batch_data_test in tqdm(data_loader,desc='Test at Epoch '+str(epoch)):
            # prepare batch data
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data[3].reshape((batch_size, num_frames))

            actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            # forward
            # actions_scores,activities_scores=model((batch_data[0],batch_data[1]))
            ret = model((batch_data[0], batch_data[1]))
            
            # Predict actions
            actions_in=actions_in.reshape((batch_size*num_frames*cfg.num_boxes,))
            activities_in=activities_in.reshape((batch_size*num_frames,))

            # Predict activities
            loss_list = []
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores,activities_in)
                loss_list.append(activities_loss)
                activities_labels = torch.argmax(activities_scores,dim=1)


                activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)

                activities_labels = activities_labels.reshape(batch_size, num_frames, -1)
                activities_in = activities_in.reshape(batch_size, num_frames, -1)


                for i, activities_meter_i in enumerate(activities_meter_list):
                    activities_labels_i = activities_labels[:, int(num_frames * (i + 1) / 10) - 1]
                    activities_in_i = activities_in[:, int(num_frames * (i + 1) / 10) - 1]
                    activities_correct_i = torch.sum(torch.eq(activities_labels_i.int(), activities_in_i.int()).float())
                    activities_accuracy_i = activities_correct_i.item() / batch_size
                    activities_meter_list[i].update(activities_accuracy_i,batch_size)
                    activities_conf_list[i].add(activities_labels_i.squeeze(-1),activities_in_i.squeeze(-1))


            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions']
                actions_weights=torch.tensor(cfg.actions_weights).to(device=device)
                actions_loss=F.cross_entropy(actions_scores,actions_in,weight=actions_weights)
                loss_list.append(actions_loss)
                actions_labels=torch.argmax(actions_scores,dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])

            if 'halting' in list(ret.keys()):
                loss_list.append(ret['halting'])

            # Total loss
            total_loss = sum(loss_list)
            loss_meter.update(total_loss.item(), batch_size)

    activities_acc_list = []
    for i in range(10):
        activities_acc_list.append(activities_meter_list[i].avg * 100)
        print('ratio:',(i+1)*0.1)
        print(activities_conf_list[i].value())

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'activities_acc_list': activities_acc_list,
    }
    
    return test_info


def test_volleyball_ratio(data_loader, model, device, epoch, cfg,ratio):
    model.eval()
    train_with_action = False
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    epoch_timer = Timer()

    with torch.no_grad():
        for batch_data_test in tqdm(data_loader, desc='Test at Epoch ' + str(epoch)):
            # prepare batch data
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data[3].reshape((batch_size, num_frames))

            actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            # forward
            # actions_scores,activities_scores=model((batch_data[0],batch_data[1]))
            ret = model((batch_data[0], batch_data[1]))


            # Predict actions
            actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            # Predict activities
            loss_list = []
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_scores=activities_scores.reshape(batch_size,num_frames,-1)[:,-1]
                activities_loss = F.cross_entropy(activities_scores, activities_in)
                loss_list.append(activities_loss)
                activities_labels = torch.argmax(activities_scores, dim=1)

                activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)


            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions']
                actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
                actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
                loss_list.append(actions_loss)
                actions_labels = torch.argmax(actions_scores, dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])

            if 'halting' in list(ret.keys()):
                loss_list.append(ret['halting'])

            # Total loss
            total_loss = sum(loss_list)
            loss_meter.update(total_loss.item(), batch_size)

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
    }

    return test_info


def train_collective(data_loader, model, device, optimizer, epoch, cfg):
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)

    for batch_data in tqdm(data_loader,desc='Train at Epoch '+str(epoch)):
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        # forward
        # actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
        activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))
        activities_in = batch_data[3].reshape((batch_size, num_frames))
        bboxes_num = batch_data[4].reshape(batch_size, num_frames)

            
        if cfg.training_stage==1:
            activities_in = activities_in.reshape(-1,)
        else:
            # activities_in = activities_in.reshape(batch_size*num_frames,)
            activities_in = activities_in[:, 0].reshape(batch_size, )


        # Predict activities
        activities_loss=F.cross_entropy(activities_scores,activities_in)
        activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
        activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
        activities_accuracy=activities_correct.item()/activities_scores.shape[0]
        activities_meter.update(activities_accuracy, activities_scores.shape[0])
        activities_conf.add(activities_labels, activities_in)

        # Total loss
        total_loss = activities_loss # + cfg.actions_loss_weight*actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
    }
    
    return train_info


def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    # flag = 0
    # wrong = []
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            # forward
            # actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
            activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))
            activities_in = batch_data[3].reshape((batch_size, num_frames))
            bboxes_num = batch_data[4].reshape(batch_size, num_frames)


            if cfg.training_stage == 1:
                activities_in = activities_in.reshape(-1, )
            else:
                activities_in = activities_in[:, 0].reshape(batch_size, )

            # Predict activities
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            activities_labels = torch.argmax(activities_scores, dim=1)  # B,
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_conf.add(activities_labels, activities_in)


            # Total loss
            total_loss = activities_loss  # + cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
    }

    return test_info


def test_collective_ratio(data_loader, model, device, epoch, cfg, ratio):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()

    epoch_timer = Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    # flag = 0
    # wrong = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc='Test at Epoch ' + str(epoch)):
            # prepare batch data
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            # forward
            activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))
            activities_in = batch_data[3].reshape((batch_size, num_frames))
            bboxes_num = batch_data[4].reshape(batch_size, num_frames)

            if cfg.training_stage == 1:
                activities_in = activities_in.reshape(-1, )
            else:
                activities_in = activities_in[:, 0].reshape(batch_size, )

            # Predict activities
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            activities_labels = torch.argmax(activities_scores, dim=1)  # B,
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_conf.add(activities_labels, activities_in)


            # Total loss
            total_loss = activities_loss  # + cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
    }  # 'actions_acc':actions_meter.avg*100

    return test_info