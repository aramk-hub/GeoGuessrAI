import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Use CUDA device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# Define transform to Normalize images (input is PIL image)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# Define training and test set
trainset = torchvision.datasets.Country211(root='./data', train=True,
                                           download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.Country211(root='./data', train=False,
                                          download=True, transform=transform)
testloader = torch._utils.data.DataLoader(testset, batch_size=batch_size, 
                                        shuffle=False, num_workers=0)

# classes = ()

class Net(nn.Module):
    def __init__(self):

    def forward(self, x):
