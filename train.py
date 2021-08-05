# coding:utf-8

from __future__ import print_function

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import math
import time
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import datetime
import argparse
import torch.optim as optim
import horovod.torch as hvd 
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
# wider face dataset for trainig wider face 
# from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
# use for inner data 
from data import CommonFaceDetectionDataSet, detection_collate, preproc, cfg_mnet, cfg_re50

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset',
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
parser.add_argument('--save_folder', default='/data/remote/github_code/face_detection/Pytorch_Retinaface/checkpoints',
                    help='Location to save checkpoint models')

parser.add_argument('--pretrain', default=0, help='widerface pretrain models')
parser.add_argument('--pretrain_model', default="/data/remote/github_code/face_detection/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth", help="widerface pretrain cchekpoints")
parser.add_argument('--log_writer', default=1, help="write the training log")
parser.add_argument('--log_dir', default="/data/remote/code/sex_image_classification/output_dir/output_log/2020_4_13_sample_train_logdir", help="tensorboard log directory")


args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

# model 
net = RetinaFace(cfg=cfg)

print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.pretrain:
    print("Loading the pretrain model!!!")
    state_dict = torch.load(args.pretrain_model, map_location="cpu")
    net.load_state_dict(state_dict)
    print("Load the pretrain model Finish!!!")


# need to change the hvd or ddp
if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), lr=initial_lr,
                      momentum=momentum, weight_decay=weight_decay)

# loss function
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

# anchor box
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))

# generate the anchor
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    # prepare the dataset
    dataset = CommonFaceDetectionDataSet(training_dataset, preproc(img_dim, rgb_mean))
    # dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    print("dataset", len(dataset))
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(
                dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + '/' +
                           cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(
            optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        if args.log_writer:
            record_log(log_writer, loss_l, loss_c, lr, iteration+1)
        


    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# add the tensorboard log
def record_log(log_writer, bbox_loss, class_loss, lr, batch_idx):
    log_writer.add_scalar("train/bbox_loss", bbox_loss.data.item(), batch_idx)
    log_writer.add_scalar("train/class_loss", class_loss.data.item(), batch_idx)
    log_writer.add_scalar("learning_rate", lr, batch_idx)


if __name__ == '__main__':
    log_writer = SummaryWriter(args.log_dir)
    train()
