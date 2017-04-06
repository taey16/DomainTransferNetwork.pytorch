from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.DTNet_32 as net
from misc import *

parser = argparse.ArgumentParser()
parser.add_argument('--datasetA', default='svhn',  help='dataset name for A domain')
parser.add_argument('--datarootA', default='', help='root-path for datasetA')
parser.add_argument('--valDatarootA', default='', help='root-path for val. datasetA')

parser.add_argument('--datasetB', default='mnist',  help='dataset name for B domain')
parser.add_argument('--datarootB', default='', help='root-path for datasetA')
parser.add_argument('--valDatarootB', default='', help='root-path for val. datasetB')

parser.add_argument('--batchSize', type=int, default=4, help='train batch size')
parser.add_argument('--valBatchSize', type=int, default=256, help='validation batch size')
parser.add_argument('--originalSize', type=int, 
  default=36, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int, 
  default=32, help='the height / width of the cropped input image to network')
parser.add_argument('--nClasses', type=int, 
  default=10, help='# of classes in source domain')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE', default='', help="path to netE trained on source domain")
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--betaTID', type=float, default=15, help='beta weight')
parser.add_argument('--alphaCONST', type=float, default=15, help='alpha weight')
parser.add_argument('--crossentropy', action='store_true', 
  help='Whether to use crossentropy loss in computing L_CONST(default: L1Loss')
# TODO implement L_TID
#parser.add_argument('--gammaTV', type=float, default=15, help='gamma weight')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay for Descriminator, default=0.0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--rmsprop', action='store_true', help='Whether to use rmsprop (default is adam)')
# NOTE Does not support CPU mode
#parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=50, help='interval for console display')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) val. images')

opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
#opt.manualSeed = 100
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get datasets for source and target domains
# NOTE datasetA: source domain(e.g. SVHN), datasetB: target domain(e.g. MNIST)
#import pdb; pdb.set_trace()
dataloaderA, datasetA = getLoader(opt.datasetA, 
                       opt.datarootA, 
                       opt.originalSize, 
                       opt.imageSize, 
                       opt.batchSize, 
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                       split='train')
valDataloaderA, valDatasetA = getLoader(opt.datasetA, 
                          opt.valDatarootA, 
                          opt.imageSize,
                          opt.imageSize, 
                          opt.valBatchSize, 
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                          split='test')
dataloaderB, datasetB = getLoader(opt.datasetB, 
                       opt.datarootB, 
                       opt.originalSize, 
                       opt.imageSize, 
                       opt.batchSize, 
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                       split='train')
valDataloaderB, valDatasetB = getLoader(opt.datasetB, 
                          opt.valDatarootB, 
                          opt.imageSize,
                          opt.imageSize, 
                          opt.valBatchSize, 
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
                          split='test')

# file pointer for lodding
trainLogger = open('%s/train.log' % opt.exp, 'w')

ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = 3
outputChannelSize= 3

# NOTE
# loading Generator, G, Discriminator D, pretrained encoder E(f in paper)
netG = net.G(outputChannelSize, ngf)
netG.apply(weights_init)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)
netE = net.E(inputChannelSize, ngf, opt.nClasses)
netE.apply(weights_init)
if opt.netE != '':
  netE.load_state_dict(torch.load(opt.netE))
print(netE)
netD = net.D(inputChannelSize, 3, ndf)
netD.apply(weights_init)
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))
print(netD)

netG.train()
netE.eval() # freezing f
netD.train()
criterionCE = nn.CrossEntropyLoss()
criterionCAE = nn.MSELoss() # nn.L1Loss()

# declear input, output tensors
target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
source = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_source = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
label_d = torch.LongTensor(opt.batchSize)
label_c = torch.LongTensor(opt.batchSize)
# NOTE:supervision for D_1, D_2, and D_3
# D has multi-class classification loss(i.e. cross-entropy) on top of it.
fake_source_label = 0
fake_target_label = 1
real_target_label = 2 

# get alpha beta
betaTID = opt.betaTID
alphaCONST = opt.alphaCONST

# on to CUDA
netD.cuda()
netG.cuda()
netE.cuda()
criterionCE.cuda()
criterionCAE.cuda()
target, source, label_d = target.cuda(), source.cuda(), label_d.cuda()
label_c = label_c.cuda()
val_target, val_source = val_target.cuda(), val_source.cuda()

# set torch autogradient Variable
target = Variable(target)
source = Variable(source)
label_d = Variable(label_d)
label_c = Variable(label_c)

# visualization samples in source, and target domains
val_source_iter = iter(valDataloaderA)
val_target_iter = iter(valDataloaderB)
val_source_cpu, _ = val_source_iter.next()
val_target_cpu, _ = val_target_iter.next()
val_source_cpu = val_source_cpu.cuda()
val_target_cpu = val_target_cpu.cuda()
val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_source.resize_as_(val_source_cpu).copy_(val_source_cpu)
vutils.save_image(val_target, '%s/samples_real_target.png' % opt.exp, nrow=16, normalize=True)
vutils.save_image(val_source, '%s/samples_real_source.png' % opt.exp, nrow=16, normalize=True)

# set optimizer
if opt.rmsprop:
  optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
  optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)
else:
  optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999), weight_decay=opt.wd)
  optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.0)

# freezing f
for p in netE.parameters():
  p.requires_grad = False

ganIterations = 0 # iteration counter for generator-updates
for epoch in range(opt.niter):
  # reorganize datasets with random shuffle
  # It is due to the fact that # of train samples in source and target domain is different
  dataloaderA = torch.utils.data.DataLoader(datasetA, 
                                            batch_size=opt.batchSize, 
                                            shuffle=True, 
                                            num_workers=opt.workers)
  dataloaderB = torch.utils.data.DataLoader(datasetB, 
                                            batch_size=opt.batchSize, 
                                            shuffle=True, 
                                            num_workers=opt.workers)
  source_iter = iter(dataloaderA)
  target_iter= iter(dataloaderB)

  i, j = 0, 0
  while i < len(source_iter) and j < len(target_iter):
    if opt.datasetB == 'mnist':
      target_cpu_1c, _ = target_iter.next()
      source_cpu, label_source = source_iter.next()
      label_source -= 1 # torch.vision.svhn returns labels ranged from 1 to 10
      # replacate 1-channel 2D array in MNIST in 3 times
      target_cpu = torch.FloatTensor(target_cpu_1c.size(0), 
                                     outputChannelSize, 
                                     target_cpu_1c.size(2), 
                                     target_cpu_1c.size(3))
      target_cpu[:,0,:,:].copy_(target_cpu_1c)
      target_cpu[:,1,:,:].copy_(target_cpu_1c)
      target_cpu[:,2,:,:].copy_(target_cpu_1c)
    else:
      target_cpu, _ = target_iter.next()
      source_cpu_1c, label_source = source_iter.next()
      source_cpu = torch.FloatTensor(source_cpu_1c.size(0), 
                                     inputChannelSize, 
                                     source_cpu_1c.size(2), 
                                     source_cpu_1c.size(3))
      source_cpu[:,0,:,:].copy_(source_cpu_1c)
      source_cpu[:,1,:,:].copy_(source_cpu_1c)
      source_cpu[:,2,:,:].copy_(source_cpu_1c)
    batch_size = target_cpu.size(0)
    i+=1
    j+=1

    # check terminal state in dataloader(iterator)
    if batch_size <> source_cpu.size(0): continue

    # on the CUDA
    target_cpu, source_cpu = target_cpu.cuda(), source_cpu.cuda()
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    source.data.resize_as_(source_cpu).copy_(source_cpu)

    # NOTE: max_D first
    for p in netD.parameters(): 
      p.requires_grad = True 
    netD.zero_grad()

    # NOTE: compute L_D in Eq.(3)
    # complute loss for D_3 and then compute grads.
    label_d.data.resize_(batch_size).fill_(real_target_label)
    output_real_target = netD(target)
    errD_real_target = criterionCE(output_real_target, label_d)
    errD_real_target.backward()
    # complute loss for D_2 and then compute grads.
    _, h_target = netE(target)
    x_hat_target = netG(h_target)
    fake_target = x_hat_target.detach()
    label_d.data.fill_(fake_target_label)
    output_fake_target = netD(fake_target)
    errD_fake_target = criterionCE(output_fake_target, label_d)
    errD_fake_target.backward()
    # complute loss for D_1 and then compute grads.
    _, h_source = netE(source)
    x_hat_source = netG(h_source)
    fake_source = x_hat_source.detach()
    label_d.data.fill_(fake_source_label)
    output_fake_source = netD(fake_source)
    errD_fake_source = criterionCE(output_fake_source, label_d)
    errD_fake_source.backward()
    # compute for loss D(i.e. L_D in eq.(3))
    L_d = errD_real_target + errD_fake_target + errD_fake_source 
    # update parameters (max_D first)
    optimizerD.step()

    # prevent computing gradients of weights in Discriminator
    for p in netD.parameters():
      p.requires_grad = False
    netG.zero_grad() # start to learning G

    # NOTE: compute L_CONST and then compute grads. (Eq.(5))
    if opt.crossentropy:
      label_c.data.resize_(batch_size).copy_(label_source)
      pred_x_hat_source, _ = netE(x_hat_source)
      L_const_ = criterionCE(pred_x_hat_source.squeeze(3).squeeze(2), label_c)
      #prec1, _ = accuracy(pred_x_hat_source.data.squeeze(3).squeeze(2), label_c.data, topk=(1,1))
    else:
      # MSE loss between f(x) and f(g(f(x))) as described in Eq.(5)
      _, output_h_source = netE(x_hat_source)
      L_const_ = criterionCAE(output_h_source, h_source.detach())
    L_const = alphaCONST * L_const_
    if alphaCONST <> 0:
      L_const.backward(retain_variables=True)

    # NOTE: compute L_TID and then compute grads. (Eq.(6))
    L_tid_ = criterionCAE(x_hat_target, target)
    L_tid = betaTID * L_tid_
    if betaTID <> 0:
      L_tid.backward(retain_variables=True)

    # NOTE: compute L_GANG and then compute grads. (Eq.(4))
    label_d.data.fill_(real_target_label)
    output_x_hat_source = netD(x_hat_source)
    errG_x_hat_source = criterionCE(output_x_hat_source, label_d)
    errG_x_hat_source.backward()
    output_x_hat_target = netD(x_hat_target)
    errG_x_hat_target = criterionCE(output_x_hat_target, label_d)
    errG_x_hat_target.backward()
    # compute L_g
    L_g = errG_x_hat_source + errG_x_hat_target

    # update parameters 
    optimizerG.step()
    ganIterations += 1

    # logging
    if ganIterations % opt.display == 0:
      print('[%d/%d][%d/%d][%d/%d] L_D: %f(rt: %f ft: %f fs: %f) L_g: %f(ft: %f fs: %f) L_const: %f L_tid: %f'
          % (epoch, opt.niter, i, len(dataloaderA), j, len(dataloaderB),
             L_d.data[0], errD_real_target.data[0], errD_fake_target.data[0], errD_fake_source.data[0], 
             L_g.data[0], errG_x_hat_target.data[0], errG_x_hat_source.data[0], 
             L_const.data[0],
             L_tid.data[0]))
      sys.stdout.flush()
      trainLogger.write('%d\t%f\t%f\t%f\t%f\n' % \
                        (i, L_d.data[0], L_g.data[0], L_const.data[0], L_tid.data[0]))
      trainLogger.flush()

    # generating images
    if ganIterations % opt.evalIter == 0:
      # NOTE: instance normalization
      val_batch_output = torch.FloatTensor(val_source.size(0), 
                                           inputChannelSize, 
                                           val_source.size(2), 
                                           val_source.size(3)).fill_(0)
      for idx in range(val_source.size(0)):
        if opt.datasetA == 'mnist':
          single_img = torch.FloatTensor(1, inputChannelSize, val_source.size(2), val_source.size(3))
          single_img[0,0,:,:].copy_(val_source[idx,:,:,:])
          single_img[0,1,:,:].copy_(val_source[idx,:,:,:])
          single_img[0,2,:,:].copy_(val_source[idx,:,:,:])
        else:
          single_img = val_source[idx,:,:,:].unsqueeze(0)
        val_sourcev = Variable(single_img.cuda(), volatile=True)
        _, h_val = netE(val_sourcev)
        x_hat_val = netG(h_val)
        val_batch_output[idx,:,:,:].copy_(x_hat_val.data)
      vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' % \
        (opt.exp, epoch, ganIterations), nrow=16, normalize=True)

  torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
  torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.exp, epoch))
trainLogger.close()
