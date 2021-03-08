from dataloader import get_data_MNIST
from CLUBGAN import *
from utils import *

import random
import torch
import torch.optim as optim
import os
import numpy as np 
import matplotlib.pyplot as plt

params = { "batch_size" : 300,
		   "epochs" : 100,
		   "checkpoint" : 10,
		   "lambda" : 0.25,
		   "lr_D" : 2e-4,
		   "lr_G" : 1e-3,
		   "beta1" : 0.5,
		   "beta2" : 0.999,
		   "noise_dim" : 62,
		   "disc_dim" : 10,
		   "cont_dim" : 2,
		   "log_step" : 100,
		   "sample_step" : 100,
		   "model_path" : "./models_clubgan",
		   "sample_path" : "./results_clubgan",
		   "sample_path_2" : "./results_clubgan_2",
		   "err" : 1e-4,
		   "sampling" : False	   
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

if not os.path.exists(params["sample_path_2"]):
	os.makedirs(params["sample_path_2"])

train_loader, test_loader = get_data_MNIST(params["batch_size"])

total_step = len(train_loader)

#list of loss
mi_losses = []

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
generator_optimizer = optim.Adam([{"params" : generator.parameters()}], lr=params["lr_G"], betas = (params["beta1"], params["beta2"]))
discriminator_optimizer = optim.Adam([{"params" : discriminator.parameters()}, {"params": d_estimator.parameters()}, {"params" : q_estimator.parameters()}], lr=params["lr_D"], betas = (params["beta1"], params["beta2"]))

#d_optimizer = optim.Adam(d_estimator.parameters(), lr=params["lr_D"], betas = (params["beta1"], params["beta2"]))

if torch.cuda.is_available():
	generator = generator.cuda()
	discriminator = discriminator.cuda()
	d_estimator = d_estimator.cuda()
	q_estimator = q_estimator.cuda()

for epoch in range(params["epochs"]):
	for i, (data, target) in enumerate(train_loader):
		discriminator_optimizer.zero_grad()
		generator_optimizer.zero_grad()

		one_labels = torch.tensor([1] * params["batch_size"]).view(-1, 1) 
		zero_labels = torch.tensor([0] * params["batch_size"]).view(-1, 1)
		
		noise = torch.randn(params["batch_size"], params["noise_dim"])
		disc_code = generate_dc(params["batch_size"], params["disc_dim"])
		cont_code = generate_cc(params["batch_size"], params["cont_dim"])

		latent = torch.cat((noise, cont_code, disc_code), 1)

		if torch.cuda.is_available():
			data = data.cuda()
			latent = latent.cuda()
			disc_code = disc_code.cuda()
			cont_code = cont_code.cuda()
			one_labels = one_labels.cuda()
			zero_labels = zero_labels.cuda()
			
		fake_output = generator(latent)

		#merged_data = torch.cat((data, fake_output.detach()))

		discriminator_fake = discriminator(fake_output.detach())
		fake_given_x = d_estimator(discriminator_fake)

		logits, mu, logvar = q_estimator(discriminator_fake)
		var = torch.exp(logvar)

		logli_disc = logli_categorical(logits, disc_code) 
		logli_cont = logli_gaussian(mu, var, cont_code)

		mean_logli_disc = torch.mean(logli_disc)
		mean_logli_cont = torch.mean(logli_cont)

		loss_disc = -1.0 * params["lambda"] * mean_logli_disc
		loss_cont = -1.0 * params["lambda"] * mean_logli_cont
		
		fake_logli_loss = logli_bernoulli(fake_given_x, zero_labels)			
		fake_logli_loss = fake_logli_loss.mean()

		discriminator_real = discriminator(data)
		real_given_x = d_estimator(discriminator_real)
		real_logli_loss = logli_bernoulli(real_given_x, one_labels)
		real_logli_loss = real_logli_loss.mean()

		logli_loss = -1.0 * (fake_logli_loss + real_logli_loss)  + loss_disc + loss_cont
		logli_loss.backward()

		discriminator_optimizer.step()
		discriminator_optimizer.zero_grad()

		discriminator_fake = discriminator(fake_output)
		fake_given_x = d_estimator(discriminator_fake)

		logits, mu, logvar = q_estimator(discriminator_fake)
		var = torch.exp(logvar)

		logli_disc = logli_categorical(logits, disc_code) 
		logli_cont = logli_gaussian(mu, var, cont_code)

		mean_logli_disc = torch.mean(logli_disc)
		mean_logli_cont = torch.mean(logli_cont)

		loss_disc = -1.0 * params["lambda"] * mean_logli_disc
		loss_cont = -1.0 * params["lambda"] * mean_logli_cont

		fake_logli_loss = logli_bernoulli(fake_given_x, zero_labels)
		positive_sample = fake_logli_loss

		if params["sampling"]:
			negative_sample = logli_bernoulli(fake_given_x, one_labels)

		else:
			negative_sample = logli_bernoulli(fake_given_x, one_labels)
			negative_sample = (negative_sample + positive_sample) / 2.0


		contrastive = positive_sample - negative_sample
		contrastive_mean = contrastive.mean()	

		loss = contrastive_mean + loss_disc + loss_cont
		loss.backward()
		
		generator_optimizer.step()
		
		if (i + 1) % params["log_step"] == 0:
			print('Epoch [%d/%d], Step[%d/%d], club_loss: %.4f'
                      % (epoch + 1, params["epochs"], i + 1, total_step, contrastive_mean.item()))

	mi_losses.append(contrastive_mean.item())

	if epoch == 0 or epoch % 10 == 9:
		with torch.no_grad():
			noise = torch.randn(100, params["noise_dim"])
			disc = generate_dc(100, params["disc_dim"])
			cont = generate_cc(100, params["cont_dim"]) 

			latent = torch.cat((noise, cont, disc), 1)
			if torch.cuda.is_available():
				latent = latent.cuda()

			samples = generator(latent)
			samples = samples.view(100, 1, 28, 28)

			GAN_type = "CLUBGAN"
			save_plot(samples, epoch, GAN_type, params["sample_path"])

			del samples, latent, disc, cont

			tmp = np.zeros((100, params["cont_dim"]))
			for k in range(10):
				tmp[k * 10 : (k + 1)* 10, 0] = np.linspace(-2, 2, 10)
			cont = torch.tensor(tmp).float()

			tmp = np.zeros((100, params["disc_dim"]))
			for k in range(10):
				tmp[k * 10 : (k + 1)*10, k] = 1
			disc = torch.tensor(tmp).float()

			latent_2 = torch.cat((noise, cont, disc), 1)

			if torch.cuda.is_available():
				latent_2 = latent_2.cuda()

			samples_2 = generator(latent_2)
			samples_2 = samples_2.view(100, 1, 28, 28)

			save_plot(samples_2, epoch, GAN_type, params["sample_path_2"])

			del samples_2, latent_2, disc, cont


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
	len_axis = len(mi_losses)
	axis = np.arange(len_axis)
	plt.plot(axis, mi_losses)
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(["MI loss"])
	plt.savefig("training_loss_CLUBGAN.png")
except:
	pass
