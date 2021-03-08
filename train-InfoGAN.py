from dataloader import get_data_MNIST
from InfoGAN import *
from utils import *

import pickle
import random
import torch
import torch.optim as optim
import os
import numpy as np 
import matplotlib.pyplot as plt

params = { "batch_size" : 200,
		   "epochs" : 100,
		   "checkpoint" : 10,
		   "lambda" : 0.5,
		   "lr_D" : 2e-4,
		   "lr_G" : 1e-3,
		   "beta1" : 0.5,
		   "beta2" : 0.999,
		   "noise_dim" : 62,
		   "disc_dim" : 10,
		   "cont_dim" : 2,
		   "log_step" : 100,
		   "sample_step" : 100,
		   "model_path" : "./models_infogan",
		   "sample_path" : "./results_infogan",
		   "err" : 1e-4	   
		 }

seed = 123
random.seed(seed)
torch.manual_seed(seed)

def generate_dc(batch_size, dim):
	prob = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
	samples = prob.sample((batch_size, ))
	return samples
	
def generate_cc(batch_size, dim):
	return torch.randn(batch_size, dim) * 0.5


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

d_estimator = D_estimator()
d_estimator.apply(weights_init)

q_estimator = Q_estimator()
q_estimator.apply(weights_init)

#optimizer
g_optimizer = optim.Adam(generator.parameters(), lr=params["lr_G"], betas = (params["beta1"], params["beta2"]))
d_optimizer = optim.Adam([{"params" : discriminator.parameters()}, {"params" : d_estimator.parameters()}, {"params" : q_estimator.parameters()}], lr=params["lr_D"], betas = (params["beta1"], params["beta2"]))

#criterion
bce_loss = nn.BCELoss()
ones_label = torch.ones(params["batch_size"], 1)
zeros_label = torch.zeros(params["batch_size"], 1)

if torch.cuda.is_available():
	ones_label = ones_label.cuda()
	zeros_label = zeros_label.cuda()
	generator = generator.cuda()
	discriminator = discriminator.cuda()
	d_estimator = d_estimator.cuda()
	q_estimator = q_estimator.cuda()

for epoch in range(params["epochs"]):
	for i, (data, target) in enumerate(train_loader):
		generator.train()
		discriminator.train()
		q_estimator.train()
		d_estimator.train()

		#optimizer discriminator
		d_optimizer.zero_grad()

		noise = torch.randn(params["batch_size"], params["noise_dim"])
		one_hot_disc = generate_dc(params["batch_size"], params["disc_dim"])
		cont_code = generate_cc(params["batch_size"], params["cont_dim"]) 

		latent = torch.cat((noise, cont_code, one_hot_disc), 1)

		if torch.cuda.is_available():
			data = data.cuda()
			latent = latent.cuda()
			one_hot_disc = one_hot_disc.cuda()
			cont_code = cont_code.cuda()

		real_disc_output = discriminator(data)
		output_real = d_estimator(real_disc_output)
		
		fake_images = generator(latent)
		fake_disc_output = discriminator(fake_images.detach())
		
		output_fake = d_estimator(fake_disc_output)

		#mi_loss
		softmax, mu_c, logvar_c = q_estimator(fake_disc_output)
		stddev = torch.exp(logvar_c)		

		logli_cat = logli_categorical(softmax, one_hot_disc)
		logli_gauss = logli_gaussian(mu_c, stddev, cont_code)

		mean_logli_cat = torch.mean(logli_cat)
		mean_logli_gauss = torch.mean(logli_gauss)

		disc_loss = -1.0 * params["lambda"] * mean_logli_cat
		cont_loss = -1.0 * params["lambda"] * mean_logli_gauss
		
		dis_loss_real = bce_loss(output_real, ones_label)
		dis_loss_fake = bce_loss(output_fake, zeros_label)

		dis_loss = dis_loss_real + dis_loss_fake + disc_loss + cont_loss
		dis_loss.backward()

		d_optimizer.step()

		del noise, one_hot_disc, cont_code, latent, fake_images, fake_disc_output, output_fake

		#optimize generator
		g_optimizer.zero_grad()
		noise = torch.randn(params["batch_size"], params["noise_dim"])
		disc_code = generate_dc(params["batch_size"], params["disc_dim"])
		cont_code = generate_cc(params["batch_size"], params["cont_dim"])

		latent = torch.cat((noise, cont_code, disc_code), 1)

		if torch.cuda.is_available():
			latent = latent.cuda()
			disc_code = disc_code.cuda()
			cont_code = cont_code.cuda()

		fake_images = generator(latent)
		fake_disc_output = discriminator(fake_images)
		output_fake = d_estimator(fake_disc_output)

		#mi loss
		softmax, mu_c, logvar_c = q_estimator(fake_disc_output)
		stddev = torch.exp(0.5 * logvar_c)		

		logli_cat = logli_categorical(softmax, disc_code)
		logli_gauss = logli_gaussian(mu_c, stddev, cont_code)

		mean_logli_cat = torch.mean(logli_cat)
		mean_logli_gauss = torch.mean(logli_gauss)

		disc_loss = -1.0 * params["lambda"] * mean_logli_cat
		cont_loss = -1.0 * params["lambda"] * mean_logli_gauss

		gen_loss =  bce_loss(output_fake, ones_label) + disc_loss + cont_loss
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

			tmp = np.zeros((100, params["cont_dim"]))
			for k in range(10):
				tmp[k * 10 : (k + 1)* 10, 0] = np.linspace(-2, 2, 10)
			cont = torch.tensor(tmp).float()

			tmp = np.zeros((100, params["disc_dim"]))
			for k in range(10):
				tmp[k * 10 : (k + 1)*10, k] = 1
			disc = torch.tensor(tmp).float()

			latent = torch.cat((noise, cont, disc), 1).float()

			if torch.cuda.is_available():
				latent = latent.cuda()

			samples = generator(latent)
			samples = samples.view(100, 1, 28, 28)

			GAN_type = "InfoGAN"
			save_plot(samples, epoch, GAN_type, params["sample_path"])

			del samples, latent, disc, cont

	if epoch % params["checkpoint"] == params["checkpoint"] - 1:
		gen_path = os.path.join(params["model_path"], "generator-%d.pkl" % (epoch + 1))
		torch.save(generator.state_dict(), gen_path)
		
		dis_path = os.path.join(params["model_path"], "discriminator-%d.pkl" % (epoch + 1))
		torch.save(discriminator.state_dict(), dis_path)
		
		d_path = os.path.join(params["model_path"], "d_estimator-%d.pkl" % (epoch + 1))
		torch.save(d_estimator.state_dict(), d_path)
		
		q_path = os.path.join(params["model_path"], "q_estimator-%d.pkl" % (epoch + 1))
		torch.save(q_estimator.state_dict(), q_path)

#plot training loss
try:

	file_gen = "gen_losses.pkl"
	file_dis = "dis_losses.pkl"

	open_file_gen = open(file_gen, "wb")
	open_file_dis = open(file_dis, "wb")

	pickle.dump(gen_losses, open_file_gen)
	pickle.dump(dis_losses, open_file_dis)
	
	open_file_gen.close()
	open_file_dis.close()

	axis = len(gen_losses) + 1
	plt.plot(axis, dis_losses)
	plt.plot(axis, gen_losses)
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(["discriminator loss", "generator loss"])
	plt.savefig("training_loss_InfoGAN.png")
except:
	pass