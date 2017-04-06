from __future__ import print_function
import argparse
import os
import sys
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import models.DTNet_32 as net
from misc import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='svhn',  help='')
parser.add_argument('--dataroot', required=False,
  default='')
parser.add_argument('--valDataroot', required=False,
  default='')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=64, help='input batch size')
parser.add_argument('--originalSize', type=int, 
  default=36, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int, 
  default=32, help='the height / width of the cropped input image to network')
parser.add_argument('--nClasses', type=int, 
  default=10, help='size of the input channels')
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--lrE', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--rmsprop', action='store_true', help='Whether to use adam (default is adam)')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay, default=0.000')
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--exp', default='recog', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=50, help='Where to store samples and models')

opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
#opt.manualSeed = 10
opt.manualSeed = random.randint(1, 10000)
torch.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get dataloader and preprocessor
#import pdb; pdb.set_trace()
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform_trn = transforms.Compose([transforms.Scale(opt.originalSize),
                                    transforms.RandomCrop(opt.imageSize),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                    ])
transform_tst = transforms.Compose([transforms.Scale(opt.originalSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                    ])
dataloader, _ = getLoader(opt.dataset, 
                       opt.dataroot, 
                       opt.originalSize, 
                       opt.imageSize, 
                       opt.batchSize, 
                       opt.workers,
                       mean=mean, std=std, 
                       #split='extra', #SVHN
                       split='train', #MNIST
                       transform_fn=transform_trn)
valDataloader, _ = getLoader(opt.dataset, 
                          opt.valDataroot, 
                          opt.imageSize,
                          opt.imageSize, 
                          opt.valBatchSize, 
                          opt.workers,
                          mean=mean, std=std, 
                          split='test',
                          transform_fn=transform_tst)

ngf = 64
inputChannelSize = 3

netE = net.E(inputChannelSize, ngf, opt.nClasses)
netE.apply(weights_init)
if opt.netE != '':
  netE.load_state_dict(torch.load(opt.netE))
print(netE)

criterionCE = nn.CrossEntropyLoss()

netE.cuda()
criterionCE.cuda()

if opt.rmsprop:
  optimizerE = optim.RMSprop(netE.parameters(), lr = opt.lrE)
else:
  optimizerE = optim.Adam(netE.parameters(), lr = opt.lrE, betas = (opt.beta1, 0.999), weight_decay=opt.wd)


def validate(val_loader, model, criterion, display=100):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  model.eval()

  end = time.time()
  for i, (input_, target) in enumerate(val_loader):
    input = torch.FloatTensor(input_.size(0), 3, input_.size(2), input_.size(3))
    if opt.dataset == 'mnist':
      input[:,0,:,:].copy_(input_)
      input[:,1,:,:].copy_(input_)
      input[:,2,:,:].copy_(input_)
    else:
      target -= 1
      input.copy_(input_)
    input = input.cuda(async=True)
    target = torch.LongTensor(target.size(0)).copy_(target).cuda(async=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    output,_ = model(input_var)
    loss = criterion(output.squeeze(3).squeeze(2), target_var)

    prec1, prec5 = accuracy(output.data.squeeze(3).squeeze(2), target, topk=(1, 5))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

    batch_time.update(time.time() - end)
    end = time.time()

    if i % display == 0:
      print('Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses,
             top1=top1, top5=top5))

  print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))
  sys.stdout.flush()
  return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, display=100):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  model.train()

  end = time.time()
  for i, (input_, target) in enumerate(train_loader):
    data_time.update(time.time() - end)

    input = torch.FloatTensor(input_.size(0), 3, input_.size(2), input_.size(3))
    if opt.dataset == 'mnist':
      input[:,0,:,:].copy_(input_)
      input[:,1,:,:].copy_(input_)
      input[:,2,:,:].copy_(input_)
    else:
      target -= 1
      input.copy_(input_)
    input = input.cuda(async=True)
    target = torch.LongTensor(target.size(0)).copy_(target).cuda(async=True)
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    output,_ = model(input_var)
    loss = criterion(output.squeeze(3).squeeze(2), target_var)

    prec1, prec5 = accuracy(output.data.squeeze(3).squeeze(2), target, topk=(1, 5))
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    if i % display == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
             epoch, i, len(train_loader), batch_time=batch_time,
             data_time=data_time, loss=losses, top1=top1, top5=top5))
      sys.stdout.flush()

best_prec1 = -1
best_epoch = -1
#import pdb; pdb.set_trace()
for epoch in range(opt.niter):
  train(dataloader, netE, criterionCE, optimizerE, epoch)
  prec1 = validate(valDataloader, netE, criterionCE)

  is_best = prec1 > best_prec1
  if is_best:
    best_prec1 = prec1
    best_epoch = epoch
    torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opt.exp, epoch))
print('Best accuracy: %f in epoch %d' % (best_prec1, best_epoch))
