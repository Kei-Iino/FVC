"""
Implementation of the model from the paper
Minnen, David, Johannes Ballé, and George D. Toderici.
["Joint autoregressive and hierarchical priors for learned image compression."](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
) Advances in Neural Information Processing Systems. 2018.
"""

import torch
from torch import nn


class HyperEncoder(nn.Module):
	def __init__(self, dim_in):
		super(HyperEncoder, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=192, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=1)
		self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2)
	
	def forward(self, x):
		x = self.conv1(x)
		x = nn.LeakyReLU()(x)
		x = self.conv2(x)
		x = nn.LeakyReLU()(x)
		x = self.conv3(x)
		return x


class HyperDecoder(nn.Module):
	def __init__(self, dim_in):
		super(HyperDecoder, self).__init__()
		
		self.deconv1 = nn.ConvTranspose2d(in_channels=dim_in, out_channels=192, kernel_size=5, stride=2)
		self.deconv2 = nn.ConvTranspose2d(in_channels=192, out_channels=288, kernel_size=5, stride=2, output_padding=1, padding=1)
		self.deconv3 = nn.ConvTranspose2d(in_channels=288, out_channels=256, kernel_size=3, stride=1)
	
	def forward(self, x):
		x = self.deconv1(x)
		x = nn.LeakyReLU()(x)
		x = self.deconv2(x)
		x = nn.LeakyReLU()(x)
		x = self.deconv3(x)
		return x


class EntropyParameters(nn.Module):
	def __init__(self, dim_in):
		super(EntropyParameters, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=640, kernel_size=1, stride=1)
		self.conv2 = nn.Conv2d(in_channels=640, out_channels=512, kernel_size=1, stride=1)
		self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
	
	def forward(self, x):
		x = self.conv1(x)
		x = nn.LeakyReLU()(x)
		x = self.conv2(x)
		x = nn.LeakyReLU()(x)
		x = self.conv3(x)
		return x


class Joint(nn.Module):
	def __init__(self):
		super(Joint, self).__init__()
		self.hyper_encoder = HyperEncoder(128)
		self.hyper_decoder = HyperDecoder(192)
		self.entropy = EntropyParameters(128+256)
		
	def quantize(self, x):
		"""
		Quantize function:  The use of round function during training will cause the gradient to be 0 and will stop encoder from training.
		Therefore to immitate quantisation we add a uniform noise between -1/2 and 1/2
		:param x: Tensor
		:return: Tensor
		"""
		uniform = -1 * torch.rand(x.shape) + 1/2
		return x + uniform.to(x.device)

	def forward(self, comp):
		
		z = self.hyper_encoder(comp)
		z_hat = self.quantize(z)
		psi = self.hyper_decoder(z_hat)
		phi_psi = torch.cat([comp, psi], dim=1)
		sigma_mu = self.entropy(phi_psi)

		sigma, mu = torch.split(sigma_mu, comp.shape[1], dim=1)

		# clip sigma so it's larger than 0 - to make sure it satisfies statistical requirement of sigma >0 and not too close to 0 so it doesn't cause computational issues
		sigma = torch.clamp(sigma, min = 1e-6)

		return sigma, mu, z_hat