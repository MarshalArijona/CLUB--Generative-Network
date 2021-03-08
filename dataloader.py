import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

root_mnist = './dataset-MNIST'

if not os.path.exists(root_mnist):
	os.makedirs(root_mnist)

def get_data_MNIST(batch_size):
	transform = transforms.Compose([
			transforms.Resize(28),
			transforms.CenterCrop(28),
			transforms.ToTensor()
		])

	train_dataset = datasets.MNIST(root_mnist, train=True, download=True, transform=transform) 
	test_dataset = datasets.MNIST(root_mnist, train=False, download=True, transform=transform)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

	return train_loader, test_loader