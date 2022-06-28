import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model.utils.defconv2d import DeformConv2d
from model.utils.bitestimate import Joint

class resblock(nn.Module):
  def __init__(self, In, N, K):
    super().__init__()
    self.convs = nn.Sequential(
        nn.Conv2d(In, N, K, 1, 1),
        nn.ReLU(),
        nn.Conv2d(N, N, K, 1, 1)
    )
  def forward(self, x):
    x_ = self.convs(x)
    x = x + x_
    return x

class FeatureExtraction(nn.Module):
  def __init__(self, In):
    super().__init__()
    self.conv = nn.Conv2d(In, 64, 5, 2, 2)
    self.reses = nn.Sequential(
        resblock(64,64,3),
        resblock(64,64,3),
        resblock(64,64,3)
    )
  def forward(self, x):
    x = self.conv(x)
    x = x + self.reses(x)
    return x

class MotionEstimation(nn.Module):
  def __init__(self, In):
    super().__init__()
    self.convs = nn.Sequential(
        nn.Conv2d(In, 64, 3, 1, 1),
        nn.Conv2d(64, 64, 3, 1, 1)
    )
  def forward(self, target, anchor):
    x = torch.cat([target, anchor], dim=1)
    x = self.convs(x)
    return x

class Compresser(nn.Module):
  def __init__(self, In):
    super().__init__()
    self.conv = nn.ModuleList([nn.Conv2d(In, 128, 3, 2, 1), nn.Conv2d(128, 128, 3, 2, 1), nn.Conv2d(128, 128, 3, 2, 1)])
    self.reses = nn.ModuleList([nn.Sequential(
        resblock(128,128,3),
        resblock(128,128,3),
        resblock(128,128,3)
    ) for i in range(3)])
    self.deconv = nn.ModuleList([nn.ConvTranspose2d(128, 128, 2, 2), nn.ConvTranspose2d(128, 128, 2, 2), nn.ConvTranspose2d(128, 64, 2, 2)])

  def quantize(self, x):
    """
    Quantize function:  The use of round function during training will cause the gradient to be 0 and will stop encoder from training.
    Therefore to immitate quantisation we add a uniform noise between -1/2 and 1/2
    :param x: Tensor
    :return: Tensor
    """
    uniform = -1 * torch.rand(x.shape) + 1/2
    return x + uniform.to(x.device)

  def forward(self, x):
    #encoder
    for i in range(3):
      x = self.conv[i](x)
      x = x + self.reses[i](x)
	#compressed
    compressed = self.quantize(x)
    x = self.quantize(x)
    #decoder
    for i in range(3):
      x_ = self.reses[i](x)
      x = self.deconv[i](x + x_)
    return x, compressed

class MotionCompensation(nn.Module):
  def __init__(self, G=8):
    super().__init__()
    self.G = G
    self.dconvs = nn.ModuleList([DeformConv2d(int(64/G), int(64/G), 3, 0, 1, modulation=True) for i in range(G)])
    self.convs = nn.Sequential(
        nn.Conv2d(128, 64, 3, 1, 1),
        nn.Conv2d(64, 64, 3, 1, 1)
    )
  def forward(self, o, anchor):
    tensorlist = []
    for i in range(self.G):
        tensorlist.append( self.dconvs[i](anchor[:, int(64/self.G*i):int(64/self.G*(i+1))], o[:, int(64/self.G*i):int(64/self.G*(i+1))]) )
    target = torch.cat(tensorlist, dim=1)
    target_ = torch.cat([target, anchor], dim=1)
    target = target + self.convs(target_)
    return target

class FrameReconstruction(nn.Module):
  def __init__(self, In):
    super().__init__()
    self.reses = nn.Sequential(
        resblock(In,64,3),
        resblock(64,64,3),
        resblock(64,64,3)
    )
    self.deconv = nn.ConvTranspose2d(64, 3, 2, 2)
  def forward(self, x):
    x_ = self.reses(x)
    x = self.deconv(x + x_)
    return x


class FVC(nn.Module):
  def __init__(self, In):
    super().__init__()
    self.fext = FeatureExtraction(In)
    self.mest = MotionEstimation(128)
    self.mcompr = Compresser(64)
    self.mcompe = MotionCompensation()
    self.rcompr = Compresser(64)
    self.frec = FrameReconstruction(64)

    self.bit1 = Joint()
    self.bit2 = Joint()
    # self.loss = RateDistortionLoss()

    # self.lam = 0.025

  def forward(self, img_target, img_anchor):

	# feature extract
    f_target = self.fext(img_target)
    f_anchor = self.fext(img_anchor)

	# motion
    offset = self.mest(f_target, f_anchor)
    offset_, comp_off = self.mcompr(offset)
    f_target_ = self.mcompe(offset_, f_anchor)

	# residual
    f_residual = f_target - f_target_
    f_residual_, comp_res = self.rcompr(f_residual)
    f_target__ = f_target_ + f_residual_

	# reconstruct
    img_target_ = self.frec(f_target__)

    # bit-rate estimate
    sigma_o, mu_o, z_hat_o = self.bit1(comp_off)
    sigma_r, mu_r, z_hat_r = self.bit2(comp_res)
    
    # loss calcurate
    # loss, self.lam = self.loss(img_target, img_target_, mu_o, sigma_o, comp_off, z_hat_o, self.lam)

    return img_target_,  mu_o, sigma_o, comp_off, z_hat_o,  mu_r, sigma_r, comp_res, z_hat_r

