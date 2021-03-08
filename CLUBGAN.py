import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		#self.linear1 = nn.Linear(74, 1024)
		self.linear1 = nn.Linear(62, 1024)
		self.bn1 = nn.BatchNorm1d(1024)

		self.linear2 = nn.Linear(1024, 7*7*128)
		self.bn2 = nn.BatchNorm1d(7*7*128)
		self.relu1 = nn.ReLU()
		

		self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu2 = nn.ReLU()
		

		self.upconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
		self.tanh1 = nn.Tanh()

	def forward(self, x):
		x = self.linear1(x)
		x = self.bn1(x)
		
		x = self.linear2(x)
		x = self.bn2(x)
		x = self.relu1(x)
		
		x = x.view(-1, 128, 7, 7)
		
		x = self.upconv1(x)
		x = self.bn3(x)
		x = self.relu2(x)
		

		x = self.upconv2(x)
		x = self.tanh1(x)

		return x

class Q_estimator(nn.Module):
	def __init__(self):
		super(Q_estimator, self).__init__()
		self.linear1 = nn.Linear(1024, 128) 
		self.bn1 = nn.BatchNorm1d(128)
		self.lrelu1 = nn.LeakyReLU(0.1, inplace=True)

		self.linear2 = nn.Linear(128, 10)
		self.linear3 = nn.Linear(128, 2)
		self.linear4 = nn.Linear(128, 2)

		self.softmax1 = nn.Softmax()
		#self.softmax2 = nn.Softmax()

	def forward(self, x):
		x = self.linear1(x)
		x = self.bn1(x)
		x = self.lrelu1(x)

		disc_logits = self.linear2(x)
		disc_logits = self.softmax1(disc_logits)

		cont_mu = self.linear3(x)
		cont_logvar = self.linear4(x)

		return disc_logits, cont_mu, cont_logvar


class D_estimator(nn.Module):
	def __init__(self):
		super(D_estimator, self).__init__()
		self.linear2 = nn.Linear(1024, 1)
		#self.softmax1 = nn.Softmax()
		self.sigmoid1 = nn.Sigmoid()

	def forward(self, x):
		x = self.linear2(x)
		#x = self.softmax1(x)
		x = self.sigmoid1(x)

		return x

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
		self.lrelu1 = nn.LeakyReLU(0.1)

		self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		self.lrelu2 = nn.LeakyReLU(0.1) 
		
		
		self.linear1 = nn.Linear(7*7*128, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.lrelu3 = nn.LeakyReLU(0.1)
		

	def forward(self, x):
		x  = self.conv1(x)
		x = self.lrelu1(x)

		x = self.conv2(x)
		x = self.bn1(x)
		x = self.lrelu2(x)

		x = x.view(-1, 7*7*128)
		
		x = self.linear1(x)
		x = self.bn2(x)
		x = self.lrelu3(x)

		return x


