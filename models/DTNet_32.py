import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class D(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(D, self).__init__()

    # input is 32
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, 64, 4, 2, 1, bias=False))
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(64, 128, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(128, 256, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = nn.Sequential()
    layer4.add_module('%s.realyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    layer4.add_module('%s.conv' % name, nn.Conv2d(256, 128, 4, 1, 0, bias=False))
    # input is 1

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = nn.Sequential()
    # NOTE: Do not add batchNorm
    #layer5.add_module('%s.bn' % name, nn.BatchNorm2d(128))
    layer5.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    layer5.add_module('%s.conv' % name, nn.Conv2d(128, 3, 1, 1, 0, bias=False))
    # 1

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    return out5.squeeze(3).squeeze(2)


class E(nn.Module):
  def __init__(self, input_nc, nf, nclasses):
    super(E, self).__init__()

    # input is 32
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, 64, 4, 2, 1, bias=False))
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(64, 128, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(128, 256, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = nn.Sequential()
    layer4.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    layer4.add_module('%s.conv' % name, nn.Conv2d(256, 128, 4, 1, 0, bias=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = nn.Sequential()
    layer5.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    layer5.add_module('%s.conv' % name, nn.Conv2d(128, nclasses, 1, 1, 0, bias=False))

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    return out5, out4


class G(nn.Module):
  def __init__(self, output_nc, nf):
    super(G, self).__init__()

    # input is 1
    layer_idx = 4
    name = 'dlayer%d' % layer_idx
    d_inc = 128
    dlayer4 = nn.Sequential()
    #dlayer4.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    dlayer4.add_module('%s.conv' % name, nn.ConvTranspose2d(d_inc, 256, 4, 1, 0, bias=False))

    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = 256
    dlayer3 = blockUNet(d_inc, 128, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = 128
    dlayer2 = blockUNet(d_inc, 64, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = 64
    dlayer1 = nn.Sequential()
    dlayer1.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s.tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
    dlayer1.add_module('%s.tanh' % name, nn.Tanh())

    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, dout5):
    dout4 = self.dlayer4(dout5)
    dout3 = self.dlayer3(dout4)
    dout2 = self.dlayer2(dout3)
    dout1 = self.dlayer1(dout2)
    return dout1
