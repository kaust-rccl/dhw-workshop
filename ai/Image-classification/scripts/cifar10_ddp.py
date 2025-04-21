import os, sys, tempfile, argparse
import torch

import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


import time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


def dataloader(gpu,world_size,batch_size,num_workers):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    ])
    data_dir=os.path.join(os.environ['COMMON_DIR'], "datasets", "cifar10")
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=False, transform=transform)
    trainSampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                               num_replicas=world_size,
                                                               rank=gpu)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          num_workers=num_workers,
                                          pin_memory=False,
                                          sampler=trainSampler)
                                         

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=False, transform=transform)
    testSampler = torch.utils.data.distributed.DistributedSampler(testset,
                                                                  num_replicas=world_size,
                                                                  rank=gpu)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=batch_size,
                                             shuffle=False, 
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             sampler=testSampler)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,testloader


# Lets check the shape of the training dataloader

# THe above shows that we have a total of 50,000 pictures of 10 classes in training dataset. 
# 
# Setting the batch_size=4 means we that our input will be 4 pictures i.e. 4*(3x32x32) pixels fed to our model at a time.
# This implies that our training loop will do 50000/4 = 12500 trips across the PCIe bus. 

# Let us show some of the training images

# ### 2. Define a Convolutional Neural Network
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).
# 
# 



# Our naive model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Layers
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # Activations    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# an alternative model from the torchvision catalouge
# net=torchvision.models.convnext_large()


def setup():
    
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)



def train (net,args):
    batch_size=args.batch_size
    epochs=args.epochs
    num_workers=args.num_workers
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    gpus=world_size
    start_timer()
    gpu_id=rank
    trainloader,testloader = dataloader(gpu_id,world_size,batch_size,num_workers)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        net.cuda(gpu_id)
    device = torch.device('cuda:%d'%(gpu_id) if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
                          momentum=args.momentum)
    
    # Wrap model as DDP
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[rank])
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i % 2000 == 1999) and (gpu_id == 0):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    end_timer_and_print('Finished Training')

def main(net,args):
    setup()
    train(net,args)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=1,
                        help="number of dataloaders", type=int)
    parser.add_argument("--batch-size", default=4,
                        help="mini batch size per GPU", type=int)
    parser.add_argument("--epochs", default=2,
                        help="total epochs", type=int)
    parser.add_argument("--lr", default=1e-3,
                        help="Learning rate",type=float)
    parser.add_argument("--momentum", default=0.9,
                        help="Momentum", type=float)
    parser.add_argument("--master-ip", default='localhost',
                        help="MASTER ADDR", type=str)
    parser.add_argument("--master-port", default=29500,
                        help="MASTER PORT", type=int)

    args = parser.parse_args()
    net=Net()
    main(net,args)








