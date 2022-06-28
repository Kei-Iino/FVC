import torch
import torch.nn as nn


class RateDistortionLoss(nn.Module):
	def __init__(self, type="sigmoid", constant_lambda=True):
		"""
		Initialise Rate-Distortion Loss function
		constant_lambda:: whether to keep lambda as constant or dynamically optimise it
		type:: which of the simplified hyperlatent rate calcuations to use [normal, sigmoid]
			normal - assumes that hyperlatens are normally distributed N(0,1) convolved with U[-1/2, 1/2]
			sigmoid - treats sigmoid of hyperlatents as their CDF,  convolved with U[-1/2, 1/2]
		"""
		super(RateDistortionLoss, self).__init__()

		self.criterion = nn.MSELoss()

		if type == "normal":
			self.hyper_cumulative = self.simple_cumulative
		elif type == "sigmoid":
			self.hyper_cumulative = self.sigmoid_cumulative
		
		if constant_lambda:
			self.assign_lambda = self.constant_lambda
		else:
			self.assign_lambda = self.lambda_update
			self.epsilon = 1e-2
	
	def cumulative(self, mu, sigma, x):
		"""
		Calculates CDF of Normal distribution with parameters mu and sigma at point x
		"""
		half = 0.5
		const = (2 ** 0.5)
		return half * (1 + torch.erf((x - mu) / (const * sigma)))

	def simple_cumulative(self, x):
		"""
		Calculates CDF of Normal distribution with mu = 0 and sigma = 1
		"""
		half = 0.5
		const = -(2 ** -0.5)
		return half * torch.erf(const * x)

	def sigmoid_cumulative(self, x):
		"""
		Calculates sigmoid of the tensor to use as a replacement of CDF
		"""
		return torch.sigmoid(x)
	
	def lambda_update(self, lam, distortion):
		"""
		Updates Lagrangian multiplier lambda at each step
		"""
		return self.epsilon * distortion + lam

	def constant_lambda(self, lam, distortion):
		"""
		Assigns Lambda the same in the case lambda is constant
		"""
		return lam
	
	def latent_rate(self, mu, sigma, y):
		"""
		Calculate latent rate
		
		Since we assume that each latent is modelled a Gaussian distribution convolved with Unit Uniform distribution we calculate latent rate
		as a difference of the CDF of Gaussian at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
		
		See apeendix 6.2
		J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
		“Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
		Available: https://openreview.net/forum?id=rkcQFMZRb.
		"""
		upper = self.cumulative(mu, sigma, (y + .5))
		lower = self.cumulative(mu, sigma, (y - .5))
		return -torch.sum(torch.log2(torch.abs(upper - lower)+(1e-6)), dim=(1, 2, 3))
	
	def hyperlatent_rate(self, z):
		"""
		Calculate hyperlatent rate
		Since we assume that each latent is modelled a Non-parametric convolved with Unit Uniform distribution we calculate latent rate
		as a difference of the CDF of the distribution at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
		See apeendix 6.2
		J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
		“Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
		Available: https://openreview.net/forum?id=rkcQFMZRb.
		"""
		upper = self.hyper_cumulative(z + .5)
		lower = self.hyper_cumulative(z - .5)
		return -torch.sum(torch.log2(torch.abs(upper - lower)+(1e-6)), dim=(1, 2, 3))
	
	def forward(self, x, x_hat, mu, sigma, y_hat, z, mu_r, sigma_r, y_hat_r, z_r, lam):
		"""
		Calculate Rate-Distortion Loss
		"""
		distortion = torch.mean((x - x_hat).pow(2))
		if int(distortion)>1000:
			print("out")

		latent_rate = torch.mean(self.latent_rate(mu, sigma, y_hat))
		hyperlatent_rate = torch.mean(self.hyperlatent_rate(z))
	
		latent_rate_r = torch.mean(self.latent_rate(mu_r, sigma_r, y_hat_r))
		hyperlatent_rate_r = torch.mean(self.hyperlatent_rate(z_r))
		
		loss =  distortion + latent_rate/10000000 + hyperlatent_rate/100000 + latent_rate_r/10000000 + hyperlatent_rate_r/100000

		if torch.isinf(loss):
			print("warn")

		lam = self.assign_lambda(lam, distortion)

		return loss, distortion, latent_rate, hyperlatent_rate, latent_rate_r, hyperlatent_rate_r