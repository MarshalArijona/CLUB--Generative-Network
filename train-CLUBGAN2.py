from dataloader import get_data_MNIST
from CLUBGAN import *
from utils import *

import random
import torch
import torch.optim as optim
import os
import numpy as np 
import matplotlib.pyplot as plt

params = { "batch_size" : 200,
		   "epochs" : 200,
		   "checkpoint" : 10,
		   "lambda" : 0.1,
		   "lr_D" : 2e-4,
		   "lr_G" : 1e-3,
		   "beta1" : 0.5,
		   "beta2" : 0.999,
		   "noise_dim" : 62,
		   "log_step" : 100,
		   "sample_step" : 100,
		   "model_path" : "./models_clubgan",
		   "sample_path" : "./results_clubgan",
		   "err" : 1e-4,
		   "sampling" : False
		 }

seed = 123
random.seed(seed)
torch.manual_seed(seed)

'''
def generate_dc(batch_size, dim):
	prob = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
	samples = prob.sample((batch_size, ))
	return samples
	
def generate_cc(batch_size, dim):
	return torch.randn(batch_size, dim) * 0.5
'''

if not os.path.exists(params["model_path"]):
	os.makedirs(params["model_path"])

if not os.path.exists(params["sample_path"]):
	os.makedirs(params["sample_path"])

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

#optimizer
generator_optimizer = optim.Adam(generator.parameters(), lr=params["lr_G"], betas = (params["beta1"], params["beta2"]))
discriminator_optimizer = optim.Adam([{"params" : discriminator.parameters()}, {"params": d_estimator.parameters()}], lr=params["lr_D"], betas = (params["beta1"], params["beta2"]))

#d_optimizer = optim.Adam(d_estimator.parameters(), lr=params["lr_D"], betas = (params["beta1"], params["beta2"]))

if torch.cuda.is_available():
	generator = generator.cuda()
	discriminator = discriminator.cuda()
	d_estimator = d_estimator.cuda()

for epoch in range(params["epochs"]):
	for i, (data, target) in enumerate(train_loader):
		generator.train()
		discriminator.train()

		discriminator_optimizer.zero_grad()
		generator_optimizer.zero_grad()

		noise = torch.randn(params["batch_size"], params["noise_dim"])
		one_labels = torch.tensor([1] * params["batch_size"]).float().view(-1, 1)
		zero_labels = torch.tensor([0] * params["batch_size"]).float().view(-1, 1)

		if torch.cuda.is_available():
			data = data.cuda()
			noise = noise.cuda()
			one_labels = one_labels.cuda()
			zero_labels = zero_labels.cuda()
			
		fake_output = generator(noise)
		
		fake_discriminator = discriminator(fake_output.detach())
		fake_given_x = d_estimator(fake_discriminator)
		logli_fake_given_x = logli_bernoulli(fake_given_x, zero_labels)
		logli_fake_given_x = logli_fake_given_x.mean()

		real_discriminator = discriminator(data)
		real_given_x = d_estimator(real_discriminator)
		logli_real_given_x = logli_bernoulli(real_given_x, one_labels)
		logli_real_given_x = logli_real_given_x.mean()

		logli_loss = -1.0 * (logli_fake_given_x + logli_real_given_x)
		logli_loss.backward()
		discriminator_optimizer.step()

		fake_discriminator = discriminator(fake_output)
		fake_given_x = d_estimator(fake_discriminator)
		logli_fake_given_x = logli_bernoulli(fake_given_x, zero_labels)

		positive_sample = logli_fake_given_x
		positive_sample = positive_sample.view(-1, 1)

		'''
		if params["sampling"]:
			negative_sample = logli_bernoulli(fake_given_x, one_labels)
		
		else:
			negative_sample = logli_bernoulli(fake_given_x, one_labels)
			negative_sample = (negative_sample + positive_sample) / 2.0
	
		contrastive = positive_sample - negative_sample
		contrastive_mean = contrastive.mean()

		loss = contrastive_mean
		'''
		
		#com_fake_given_x = 1.0 - fake_given_x
		#log_com_fake_given_x = torch.log(com_fake_given_x + 1e-4)
		

		sum_fake_given_x = torch.sum(fake_given_x)
		sum_fake_given_x = (sum_fake_given_x - fake_given_x) / (params["batch_size"] - 1)
		
		#sum_real_given_x = torch.sum(real_given_x)
		#sum_real_given_x = (sum_real_given_x - real_given_x) / (params["batch_size"] - 1)

		#positive_sample = torch.cat((logli_real_given_x, logli_fake_given_x))
		#sum_given_x = torch.cat((sum_real_given_x, sum_fake_given_x))

		term1 = positive_sample - torch.log(sum_fake_given_x + 1e-4)
		
		l1_out = term1
		l1_out_mean = l1_out.mean()
		loss = l1_out.mean()
		
		#print(loss)
		

		'''
		vub = positive_sample - torch.log(torch.tensor(0.9))
		vub_mean = vub.mean()
		loss = vub_mean
		'''

		loss.backward()

		generator_optimizer.step()


		if (i + 1) % params["log_step"] == 0:
			print('Epoch [%d/%d], Step[%d/%d], mi_loss: %.4f'
                      % (epoch + 1, params["epochs"], i + 1, total_step, loss.item()))

	#mi_losses.append(contrastive_mean.item())
	mi_losses.append(loss.item())

	if epoch == 0 or epoch % 10 == 9:
		with torch.no_grad():
			generator.eval()

			noise = torch.randn(100, params["noise_dim"])

			if torch.cuda.is_available():
				noise = noise.cuda()

			samples = generator(noise)
			samples = samples.view(100, 1, 28, 28)

			GAN_type = "CLUBGAN"
			save_plot(samples, epoch, GAN_type, params["sample_path"])

			del samples, noise

	if epoch % params["checkpoint"] == params["checkpoint"] - 1:
		gen_path = os.path.join(params["model_path"], "generator-%d.pkl" % (epoch + 1))
		torch.save(generator.state_dict(), gen_path)
		
		dis_path = os.path.join(params["model_path"], "discriminator-%d.pkl" % (epoch + 1))
		torch.save(discriminator.state_dict(), dis_path)
		
		d_path = os.path.join(params["model_path"], "d_estimator-%d.pkl" % (epoch + 1))
		torch.save(d_estimator.state_dict(), d_path)
		

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
