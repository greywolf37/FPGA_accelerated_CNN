# This template should be followed to benchmark your solution

# Imports
import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from pathlib import Path

# Initialization
# Perform any initialization procedures such as 
#   Binding
#   Flashing FPGA
#   Instantiating your FPGA accelerated VGG16 net
#   Any other initialization that shouldn't be timed

# Get datasets
data_dir = Path('~/data').expanduser()
os.mkdir(data_dir)
os.system('wget https://s3.amazonaws.com/ese680.imagenet/ILSVRC2012_devkit_t12.tar.gz -P ~/data/ --show-progress -nc')
os.system('wget https://s3.amazonaws.com/ese680.imagenet/ILSVRC2012_devkit_t3.tar.gz -P ~/data/ --show-progress -nc')
os.system('wget https://s3.amazonaws.com/ese680.imagenet/ILSVRC2012_img_val.tar -P ~/data/ --show-progress -nc')

# Create dataset
input_size = 224
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageNet(data_dir, split='val', transform=test_transform)

# Create dataloaders
batch1_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)
batch32_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=32)

########################################################################
# Batch size 1 benchmarking
# Get a single (input, label) tuple of batch size 1 from your dataloader
# TODO

# Run inference on the single (input, label) tuple
# TODO

# Start timer
tic = time.perf_counter()

# Run loop that performs 1024 inferences of batch size 1
for i, data in zip(range(1024), batch1_loader):
    # TODO
    pass

# end timer
toc = time.perf_counter()

# Print results
runtime_1 = toc - tic
print(f'Runtime(1) = {runtime_1:0.4f} seconds')
# TODO Print accuracy of network

########################################################################
# Batch size 32 benchmarking
# Get a single (input, label) tuple of batch size 32 from your dataloader
# TODO

# Run inference on the single (input, label) tuple
# TODO

# Start timer
tic = time.perf_counter()

# Run loop that performs 1024 inferences of batch size 32
for i, data in zip(range(1024), batch32_loader):
    # TODO
    pass

# end timer
toc = time.perf_counter()

# Print results
runtime_32 = toc - tic
print(f'Runtime(32) = {runtime_32:0.4f} seconds')
# TODO Print accuracy of network

########################################################################

# Print score
print(f'FOM = {((runtime_1 + runtime_32) / 2):0.4f}')
