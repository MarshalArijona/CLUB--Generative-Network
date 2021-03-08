from dataloader import get_data_MNIST
from GAN import *
from utils import *

import torch
import torch.optim as optim
import os
import random
import numpy as np 
import matplotlib.pyplot as plt

params = { "batch_size" : 200,
		   "epochs" : 100,
		   "checkpoint" : 10,
		   "lr_D" : 2e-4,
		   "lr_G" : 1e-3,
		   "beta1" : 0.5,
		   "beta2" : 0.999,
		   "noise_dim" : 62,
		   "log_step" : 100,
		   "sample_step" : 100,
		   "model_path" : "./models",
		   "sample_path" : "./results"	   
		 }

seed = 123
random.seed(seed)
torch.manual_seed(seed)

if not os.path.exists(params["model_path"]):
	os.makedirs(params["model_path"])

if not os.path.exists(params["sample_path"]):
	os.makedirs(params["sample_path"])

train_loader, test_loader = get_data_MNIST(params["batch_size"])

total_step = len(train_loader)

#list of loss
gen_losses = []
dis_losses = []

#network
generator = Generator()
generator.apply(weights_init)

discriminator = Discriminator()
discriminator.apply(weights_init)

#optimizer
g_optimizer = optim.Adam(generator.parameters(), lr = params["lr_G"], betas = (params["beta1"], params["beta2"]))
d_optimizer = optim.Adam(discriminator.parameters(), lr = params["lr_D"], betas = (params["beta1"], params["beta2"]))

#criterion
bce_loss = nn.BCELoss()
ones_label = torch.ones(params["batch_size"], 1)
zeros_label = torch.zeros(params["batch_size"], 1)

if torch.cuda.is_available():
	ones_label = ones_label.cuda()
	zeros_label = zeros_label.cuda()
	generator = generator.cuda()
	discriminator = discriminator.cuda()

for epoch in range(params["epochs"]):
	for i, (data, target) in enumerate(train_loader):
		generator.train()
		discriminator.train()

		d_optimizer.zero_grad()

		#noise
		noise = torch.randn(params["batch_size"], params["noise_dim"])

		if torch.cuda.is_available():
			data = data.cuda()
			noise = noise.cuda()

		fake_images = generator(noise)

		output_real = discriminator(data)
		output_fake = discriminator(fake_images.detach())

		dis_loss_real = bce_loss(output_real, ones_label)
		dis_loss_real.backward()

		dis_loss_fake = bce_loss(output_fake, zeros_label)
		dis_loss_fake.backward()

		dis_loss = dis_loss_real + dis_loss_fake
		d_optimizer.step()

		del noise, fake_images, output_fake

		g_optimizer.zero_grad()

		noise = torch.randn(params["batch_size"], params["noise_dim"])

		if torch.cuda.is_available():
			noise = noise.cuda()
			
		fake_images = generator(noise)
		output_fake = discriminator(fake_images)

		gen_loss =  bce_loss(output_fake, ones_label)
		gen_loss.backward()
		g_optimizer.step()

		if i == 0:
			gen_losses.append(gen_loss.detach().cpu().item())
			dis_losses.append(dis_loss.detach().cpu().item())

		if (i + 1) % params["log_step"] == 0:
			print('Epoch [%d/%d], Step[%d/%d], dis_loss: %.4f, gen_loss: %.4f'
                      % (epoch + 1, params["epochs"], i + 1, total_step, dis_loss.item(), gen_loss.item()))

	if epoch % 10 == 9:
		with torch.no_grad():
			generator.eval()

			noise = torch.randn(100, params["noise_dim"])
		
			if torch.cuda.is_available():
				noise = noise.cuda()

			sample_images = generator(noise)
			sample_images = sample_images.view(100, 1, 28, 28)

			GAN_type = "GAN"
			save_plot(sample_images, epoch, GAN_type, params["sample_path"])

			del noise, sample_images


	if epoch % params["checkpoint"] == params["checkpoint"] - 1:
		gen_path = os.path.join(params["model_path"], "generator-%d.pkl" % (epoch + 1))
		torch.save(generator.state_dict(), gen_path)
		dis_path = os.path.join(params["model_path"], "discriminator-%d.pkl" % (epoch + 1))
		torch.save(discriminator.state_dict(), dis_path)

#plot training loss
try:
	axis = len(gen_losses) + 1
	plt.plot(axis, dis_losses)
	plt.plot(axis, gen_losses)
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(["discriminator loss", "generator loss"])
	plt.savefig("training_loss_GAN.png")
except:
	pass
		




