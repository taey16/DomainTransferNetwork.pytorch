import torch
import os 
import sys

def create_exp_dir(exp):
  try:
    os.makedirs(exp)
    print('Creating exp dir: %s' % exp)
  except OSError:
    pass
  return True


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def getLoader(datasetName, dataroot, originalSize, imageSize, batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', transform_fn=None):
  import torchvision.transforms as transforms
  if transform_fn is None and (split=='train' or split=='extra'):
    transform_fn = transforms.Compose([transforms.Scale(originalSize),
                                        transforms.RandomCrop(imageSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                      ])
  elif transform_fn is None and split=='test':
    transform_fn = transforms.Compose([transforms.Scale(imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std),
                                      ])
    
  if datasetName == 'svhn':
    from torchvision.datasets.svhn import SVHN as commonDataset
    if split=='train': split = 'extra'
    dataset = commonDataset(root=dataroot, 
                            download=True, 
                            split=split, 
                            transform=transform_fn)
  elif datasetName == 'mnist':
    from torchvision.datasets.mnist import MNIST as commonDataset
    flag_trn = split=='train'
    dataset = commonDataset(root=dataroot, 
                            download=True, 
                            train=flag_trn, 
                            transform=transform_fn)

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batchSize, 
                                           shuffle=True, 
                                           num_workers=int(workers))
  return dataloader, dataset


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()
  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
