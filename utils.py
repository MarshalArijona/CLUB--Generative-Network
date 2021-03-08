import torch
import torchvision
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn


def logli_bernoulli(prob, samples, tiny=1e-4):
	term = samples * torch.log(prob + tiny) + (1.0 - samples) * torch.log(1 - prob + tiny)
	logli = torch.sum(term, 1)

	return logli

def logli_continuous_bernoulli(prob, samples, tiny=1e-4):
	term = 1 - 2 * prob
	cb = 2 * torch.atanh(term) / (term + tiny)
	logli = cb * prob ** samples * (1 - prob) ** (1 - samples)
	return logli

def logli_categorical(prob, samples, tiny=1e-4):
	term = torch.log(prob + tiny) * samples
	logli = torch.sum(term, 1)

	return logli


def logli_gaussian(mean, stddev, samples, tiny=1e-4):
	eps = (samples - mean) / (stddev + tiny)
	term = - 0.5 * torch.log(torch.tensor(2 * 3.14)) - torch.log(stddev + tiny) - 0.5 * torch.square(eps)
	logli = torch.sum(term, 1)

	return logli 

def save_plot(examples, epoch, gan, path, nrow=10):
	filename = path + "/" + gan + "-samples-%d.png" % (epoch + 1)
	#torchvision.utils.save_image(examples, nrow=nrow, fp=filename, padding=2, normalize=True)
	torchvision.utils.save_image(examples, nrow=nrow, fp=filename, padding=2)


#already in cuda
def vCLUB(data_x, data_y, sup_data_y, logli, discriminator, d_estimator, discriminator_optimizer, batch_size, sampling=False):
	#update q(y|x)
	mean_logli = -1.0 * torch.mean(logli)
	mean_logli.backward()
	
	discriminator_optimizer.step()

	discriminator_optimizer.zero_grad()

	#postive sample
	y_probs = discriminator(data_x)
	y_probs = d_estimator(y_probs)

	positive_sample = logli_bernoulli(y_probs, data_y)
	positive_sample = positive_sample.view(-1, 1)


	if torch.cuda.is_available():
		positive_sample = positive_sample.cuda()

	if sampling:
		#dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.tensor([0.5, 0.5]))
		#smpl = dist.sample(())
		negative_sample = logli_bernoulli(y_probs, sup_data_y)

	else:
		negative_sample = logli_bernoulli(y_probs, sup_data_y)
		negative_sample = (positive_sample + negative_sample) / 2.0 

	if torch.cuda.is_available():
		negative_sample = negative_sample.cuda()

	contrastive = positive_sample - negative_sample
	contrastive_mean = contrastive.mean()

	return contrastive_mean


def vcbCLUB(data_x, data_y, logli, discriminator, d_estimator, discriminator_optimizer, batch_size, sampling=False):
	#update q(y|x)
	mean_logli = -1.0 * torch.mean(logli)
	mean_logli.backward()
	
	discriminator_optimizer.step()

	discriminator_optimizer.zero_grad()

	#postive sample
	y_probs = discriminator(data_x)
	y_probs = d_estimator(y_probs)

	positive_sample = logli_bernoulli(y_probs, data_y)
	positive_sample = positive_sample.view(-1, 1)
	
	shuffled_index = torch.randperm(batch_size)
	positive_sample = positive_sample[shuffled_index, :]

	if torch.cuda.is_available():
		positive_sample = positive_sample.cuda()

	if sampling:
		sampling_index = torch.randint(0, batch_size, (batch_size, ))
		negative_sample = positive_sample[sampling_index, :]

	else:
		negative_sample = torch.mean(positive_sample)

	if torch.cuda.is_available():
		negative_sample = negative_sample.cuda()

	contrastive = positive_sample - negative_sample
	contrastive_mean = contrastive.mean()

	return contrastive_mean

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def merge_data(fake_data, real_data, batch_size):
	#ones_label = torch.ones(params["batch_size"], 1)
	#zeros_label = torch.zeros(params["batch_size"], 1)
	ones_label = [1] * batch_size
	zeros_label = [0] * batch_size

	labels = zeros_label + ones_label

	labels = torch.tensor(labels)
	#labels = torch.nn.functional.one_hot(labels)
	#labels = labels.view(2 * batch_size, 2)
	labels = labels.view(2 * batch_size, 1)

	merged_data = torch.cat((fake_data, real_data))
	
	index = torch.randperm(2 * batch_size)
	
	shuffled_merged_data = merged_data[index, :]
	shuffled_labels = labels[index, :]

	return shuffled_merged_data, shuffled_labels


def vCLUB3(data_x, data_y, logli, discriminator, d_estimator, discriminator_optimizer, d_optimizer, batch_size, sampling=True, tiny=1e-4):
	#if torch.cuda.is_available():
	#	data_x = data_x.cuda()
	#	data_y = data_y.cuda()
	#	logli = logli.cuda()

	#neg_samples = []
	#negative_sample = None

	#update q(y|x)
	mean_logli = -1.0 * torch.mean(logli)
	mean_logli.backward(retain_graph=True)
	
	d_optimizer.step()
	discriminator_optimizer.step()

	#d_optimizer.zero_grad()
	#discriminator_optimizer.zero_grad()

	#postive sample
	y_probs = discriminator(data_x)
	y_probs = d_estimator(y_probs)

	#print(y_probs)

	#positive_sample = logli_categorical(y_probs, data_y)
	#positive_sample = logli_bernoulli(y_probs, data_y)
	positive_sample = logli_categorical(y_probs, data_y)
	positive_sample = positive_sample.view(-1, 1)
	
	if torch.cuda.is_available():
		positive_sample = positive_sample.cuda()

	if sampling:
		idx_smpl_real = torch.randint(0, batch_size // 2, (batch_size // 2, ))
		idx_smpl_fake = torch.randint(batch_size // 2, batch_size, (batch_size // 2, ))

		real_neg_sample = positive_sample[idx_smpl_fake, :]
		fake_neg_sample = positive_sample[idx_smpl_real, :]

		negative_sample = torch.cat((real_neg_sample, fake_neg_sample))
		negative_sample = negative_sample.view(-1, 1)

	else:
		mean_real_logli = positive_sample[:batch_size // 2, :].mean()
		mean_fake_logli = positive_sample[batch_size // 2:, :].mean()

		real_neg_sample = positive_sample[:batch_size // 2, :] - mean_fake_logli
		fake_neg_sample = positive_sample[batch_size // 2:, :] - mean_real_logli

		negative_sample = torch.cat((real_neg_sample, fake_neg_sample))

	if torch.cuda.is_available():
		negative_sample = negative_sample.cuda()

	#negative sample
	#if sampling:
		#index = torch.randint(0, batch_size, (batch_size, ))
		#sampled_y = data_y[index, :]
		
		#negative_sample = torch.log(y_probs + tiny) * data_y
		#negative_sample = data_y * torch.log(y_probs + tiny) + (1.0 - data_y) * torch.log(1 - y_probs + tiny)
		#negative_sample = torch.sum(negative_sample, 1)
		#negative_sample = negative_sample.view(-1, 1)
		
	#else:
	

		#for i in range(y_probs.shape[0]):
			#negative_i = torch.log(y_probs[i] + tiny) * data_y
		#	negative_i = data_y * torch.log(y_probs[i] + tiny) + (1.0 - data_y) * torch.log(1 - y_probs[i] + tiny)
		#	negative_i = torch.sum(negative_i, 1)
		#	negative_i = negative_i.mean()
		#	neg_samples.append(negative_i)

		#neg_samples = torch.tensor(neg_samples).view(-1, 1)
		#negative_sample = neg_samples
		#negative_sample

	#if torch.cuda.is_available():
	#	negative_sample = negative_sample.cuda()

	contrastive = positive_sample - negative_sample

	#print("positive_sample")
	#print(positive_sample)

	#print("negative_sample")
	#print(negative_sample)

	#print("contrastive")
	#print(contrastive)

	contrastive_mean = contrastive.mean()

	return contrastive_mean