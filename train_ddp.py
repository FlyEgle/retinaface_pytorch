# -*- coding:utf-8 -*-
# Trainig the RetinaFace with the torch ddp instead of the dataparallel
from __future__ import print_function

# system
import os
import math
import json
import time
import random
import torch
import datetime
import argparse
import numpy as np 

# torch
import torch 
import torch.optim as optim 
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp 
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

# retinaface models
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from collections import OrderedDict

# use for inner data 
from data import CommonFaceDetectionDataSet, detection_collate, preproc, cfg_mnet, cfg_re50

# apex
import apex 
from apex import amp 

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_file',
                    default='/data/remote/dataset/wider_face/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None,
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_ckpt', default='/data/remote/github_code/face_detection/Pytorch_Retinaface/checkpoints',
                    help='Location to save checkpoint models')
parser.add_argument('--num_classes', default=2, type=int, 
                    help="The num classes for face detection")
parser.add_argument('--image_size', default=640, type=int, 
                    help='image size for training')
parser.add_argument('--warmup_epochs', default=0, type=int, 
                    help='image size for training')

parser.add_argument('--pretrain', default=0, help='widerface pretrain models')
parser.add_argument('--pretrain_model', default="/data/remote/github_code/face_detection/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth", help="widerface pretrain cchekpoints")
parser.add_argument('--log_writer', default=1, help="write the training log")
parser.add_argument('--log_dir', default="/data/remote/code/sex_image_classification/output_dir/output_log/2020_4_13_sample_train_logdir", help="tensorboard log directory")
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ngpu', type=int, default=1)

parser.add_argument('--seed', default=100, type=int, help='seed for initializing training.')

# ddp 
parser.add_argument('--world-size', type=int, default=-1,
                    help="number of nodes for distributed training")
parser.add_argument('--rank', default=-1, type=int,
                help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                help='distributed backend')
parser.add_argument('--multiprocessing-distributed', default=1, type=int,
                help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
parser.add_argument('--local_rank', default=1)

# apex
parser.add_argument('--use_apex', type=int, default=1, help='use the apex for mixup traininig!!!')


def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 100:
        # fast the training loop and no stable on the value 
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False   

def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value 
        else:
            new_state_dict[key] = value
    return new_state_dict


def main_worker(gpu, ngpus_per_node, args):
    # each gpu is like a rank -
    args.gpu = gpu 

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu 
    args.model = "{}-{}".format("Retinaface", args.network)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print('rank: {} / {}'.format(args.rank, args.world_size))
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)

    if args.rank == 0:
        if not os.path.exists(args.save_ckpt):
            os.mkdir(args.save_ckpt)
    # create model 
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    model = RetinaFace(cfg=cfg)
    if args.rank == 0:
        print("================{}=============".format(args.model))
        # print(model)
    # resume 
    if args.resume_net is not None:
        if args.rank == 0:
            print("Loading resume network....")
        state_dict = torch.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
    
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    # pretrain
    if args.pretrain:
        if args.rank == 0:
            print("Loading the pretrain model!!!")
        state_dict = torch.load(args.pretrain_model, map_location="cpu")
        if "state_dict" in state_dict.keys():
            new_state_dict = translate_state_dict(state_dict['state_dict'])
            model.load_state_dict(new_state_dict)
        else:
            new_state_dict = translate_state_dict(state_dict)
            model.load_state_dict(new_state_dict)
        if args.rank == 0:
            print("Load the pretrain model Finish!!!")
    
    model.cuda(args.gpu)
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
    # use the ddp data model
    if args.use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    # loss function
    criterion = MultiBoxLoss(args.num_classes, 0.35, True, 0, True, 7, 0.35, False)

    # anchor box
    priorbox = PriorBox(cfg, image_size=(args.image_size, args.image_size))
    
    # generate the anchor
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda(args.gpu)
    print("generate priors anchor ", args.rank)
    # traindataset
    rgb_mean = (104, 117, 123)  # bgr order
    start_time = time.time()
    train_dataset = CommonFaceDetectionDataSet(args.training_file, preproc(args.image_size, rgb_mean))
    end_time = time.time()
    # print("build dataset waste time is {}".format(end_time - start_time))
    if args.rank == 0:
        print("dataset", len(train_dataset))
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None 
    
    log_writer = SummaryWriter(args.log_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=detection_collate
    )

    if args.rank == 0:
        information_kwargs = {
            "Epoch ": args.epochs,
            "Model ": args.network,
            "BatchSize ": args.batch_size * ngpus_per_node,
            "Classes ": args.num_classes,
            "Optimizer_learning_rate ": args.lr,
            "Log dir ": args.log_dir,
            "apex ": args.use_apex,
            "Save Checkpoint ": args.save_ckpt,
            "Train_data ": args.training_file,
            "num_wrokers ": args.num_workers,
        }
        Information_data = json.dumps(
            information_kwargs, sort_keys=True, indent=4, separators=(',', ':'))
        print("==============**********==============")
        print(Information_data)

    batch_iter = 0
    train_batch = math.ceil(len(train_dataset) / (args.batch_size * ngpus_per_node))
    total_batch = train_batch * args.epochs
    # training loop 
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for epoch
        batch_iter = train(train_loader, model, criterion, optimizer, epoch, args, batch_iter, priors, total_batch, train_batch, log_writer)    

        if (epoch + 1) % 5 == 0:
            if args.rank == 0:
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                if args.use_apex:
                    state_dict['amp'] = amp.state_dict()
                torch.save(state_dict, args.save_ckpt + '/'  + args.network  + '_epoch_{}'.format(epoch+1) + '.pth')
        
    if args.rank == 0:
        state_dict = translate_state_dict(model.state_dict())
        state_dict = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }
        if args.use_apex:
            state_dict['amp'] = amp.state_dict()
            torch.save(model.state_dict(), args.save_ckpt + '/'  + args.network  + '_final.pth')
    

def record_log(log_writer, bbox_loss, class_loss, lr, batch_idx, batch_time):
    log_writer.add_scalar("train/bbox_loss", bbox_loss.data.item(), batch_idx)
    log_writer.add_scalar("train/class_loss", class_loss.data.item(), batch_idx)
    log_writer.add_scalar("learning_rate", lr, batch_idx)
    log_writer.add_scalar("train/batch_time", batch_time, batch_idx)

def train(train_loader, model, criterion, optimizer, epoch, args, batch_iter, priors, total_batch, train_batch, log_writer):
    model.train() 
    for batch_idx, (images, targets) in enumerate(train_loader):
        
        batch_start = time.time()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = [anno.cuda(args.gpu, non_blocking=True) for anno in targets]
        
        lr = adjust_learning_rate(epoch, args, batch_idx+1, optimizer)
        
        # forward
        optimizer.zero_grad()
        output = model(images)
        loss_l, loss_c, loss_landm = criterion(output, priors, targets)
        loss = 2.0 * loss_l + loss_c + loss_landm

        if args.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # backprop
        else:
            loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_start

        batch_iter += 1
        batch_idx += 1
    
        if args.rank == 0:
            print("Training Epoch: [{}/{}] batchidx:[{}/{}] batchiter: [{}/{}] Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} LearningRate: {:.6f} Batchtime: {:.4f}s".format(
                epoch+1, args.epochs, batch_idx, train_batch, batch_iter, total_batch, loss_l, loss_c, loss_landm, lr, batch_time
            ))

        if args.log_writer:
            if args.rank == 0:
                record_log(log_writer, loss_l, loss_c, lr, batch_iter, batch_time)

    return batch_iter


def adjust_learning_rate(epoch, args, batch_idx, optimizer):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / ngpus_per_node * (epoch * (ngpus_per_node - 1) / args.warmup_epochs + 1)
    elif epoch < 190:
        lr_adj = 1.
    elif epoch < 220:
        lr_adj = 1e-1
    # elif epoch < int(args.epochs):
    #     lr_adj = 1e-2
    else:
        lr_adj = 1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


if __name__ == "__main__":
    args = parser.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("ngpus_per_node", ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("ngpus_per_node", ngpus_per_node)
        main_worker(args.gpu, ngpus_per_node, args)