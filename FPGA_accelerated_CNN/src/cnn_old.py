# Skip to content
# Search or jump to…

# Pull requests
# Issues
# Marketplace
# Explore
 
# @jhanv 
# greywolf37
# /
# FPGA_accelerated_CNN
# 2
# 00
# Code
# Issues
# Pull requests
# Actions
# Projects
# Wiki
# Security
# Insights
# FPGA_accelerated_CNN/src/cnn.py /
# @jhanv
# jhanv Done VGG integration
# Latest commit e089917 7 hours ago
#  History
#  1 contributor
# 223 lines (167 sloc)  7.23 KB
  
# IN CPP
# include <torch/extension.h>
# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#   m.def("forward_sw", &forward_sw, "Forward");
#   m.def("backward_sw", &backward_sw, "Backward");
# }

import time
import os
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from pathlib import Path

# import sklearn
# from sklearn.metrics import accuracy_score

from torch.utils.cpp_extension import load
lltm_cpp = load(name="host", sources=["host.cpp"])
import host

# This template should be followed to benchmark your solution

# Imports


# Initialization
# Perform any initialization procedures such as 
#   Binding
#   Flashing FPGA
#   Instantiating your FPGA accelerated VGG16 net
#   Any other initialization that shouldn't be timed

# Get datasets
data_dir = Path('~/data').expanduser()
if not os.path.isdir(data_dir) :
    
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

# Downloading the pretrained model
vgg16 = models.vgg16(pretrained=True)


class CNN_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weights):
        return lltm_cpp.forward_sw(input_tensor, weights)

        
class CNN_cpp(torch.nn.Module):
    def __init__(self):
        super(CNN_cpp, self).__init__()
        self.inputs_size = 4

    def forward(self, input_tensor, weights, bias):
        input_tensor = F.pad(input_tensor, (1,1,1,1))
        bias = bias.reshape((1, bias.shape[0], 1,1))
        return CNN_function.apply(input_tensor, weights) + bias


class VGG16_custom(nn.Module):
    def __init__(self, pretrained_features):
        super(VGG16_custom, self).__init__()
        self.features = pretrained_features
        self.fpga_conv2d = CNN_cpp()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        self.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=1000, bias=True) 
        )

    def forward(self, x):
        x = self.relu(self.fpga_conv2d(x, self.features[0].weight, self.features[0].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[2].weight, self.features[2].bias))
        x = self.maxpool(x)
        x = self.relu(self.fpga_conv2d(x, self.features[5].weight, self.features[5].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[7].weight, self.features[7].bias))
        x = self.maxpool(x)
        x = self.relu(self.fpga_conv2d(x, self.features[10].weight, self.features[10].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[12].weight, self.features[12].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[14].weight, self.features[14].bias))
        x = self.maxpool(x)
        x = self.relu(self.fpga_conv2d(x, self.features[17].weight, self.features[17].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[19].weight, self.features[19].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[21].weight, self.features[21].bias))
        x = self.maxpool(x)
        x = self.relu(self.fpga_conv2d(x, self.features[24].weight, self.features[24].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[26].weight, self.features[26].bias))
        x = self.relu(self.fpga_conv2d(x, self.features[28].weight, self.features[28].bias))
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

for param in vgg16.parameters():
    param.requires_grad = False

pretrained_features = vgg16.features
model = VGG16_custom(pretrained_features)

with torch.no_grad():
    for j in range(len(vgg16.classifier)):
        if str(vgg16.classifier[j]).startswith("Linear"):
            model.classifier[j].weight = vgg16.classifier[j].weight
            model.classifier[j].bias = vgg16.classifier[j].bias

model.eval()

# ########################################################################
# # Batch size 1 benchmarking
# # Get a single (input, label) tuple of batch size 1 from your dataloader
# # TODO


# input, label = next(iter(batch1_loader))
# # print(input)
# # Run inference on the single (input, label) tuple
# # TODO
# with torch.no_grad():
#   output = model(input)
#   output_vgg = vgg16(input)


# print("*"*100)
# # Start timer
# tic = time.perf_counter()

# # Run loop that performs 1024 inferences of batch size 1
# labels, outputs = [], []
# ti = 0
# for i, data in zip(range(10), batch1_loader):
#     # TODO
#     input, label = data
#     start = time.perf_counter()
#     output = model(input)
#     end = time.perf_counter()
#     ti += end - start

#     print(f'Runtime(1) = {ti:0.4f} seconds for {i} instances')
#     # print(output)
#     labels.append(label.item())
#     outputs.append(torch.argmax(output).item())
# # end timer
# toc = time.perf_counter()

# # Print results
# runtime_1 = toc - tic

# print(f'Runtime(1) = {runtime_1:0.4f} seconds')
# # TODO Print accuracy of network
# print(labels, outputs)
# print("Accuracy", sum([a == b for a, b in zip(labels, outputs)])/ len(labels))

print("*"*100)
########################################################################
# Batch size 32 benchmarking
# Get a single (input, label) tuple of batch size 32 from your dataloader
# TODO
# input, label = load_item(1, batch1_loader) 

# Run inference on the single (input, label) tuple
# TODO


input, label = next(iter(batch32_loader))
# print(input)
# Run inference on the single (input, label) tuple
# TODO
with torch.no_grad():
  output = model(input)
print(output)

print("*"*100)

# Start timer
tic = time.perf_counter()
outputs = []
labels = []
ti = 0
# Run loop that performs 1024 inferences of batch size 32
for i, data in zip(range(10), batch32_loader):
    # TODO
    input, label = data
    start = time.perf_counter()
    output = model(input)
    end = time.perf_counter()
    ti += end - start
    print(f'Runtime(32) = {ti:0.4f} seconds for {i} inferences')
    for lab, out in zip(label,output):
      outputs.append(torch.argmax(out).item())
      labels.append(lab.item())
      # print(len(outputs), "************", len(labels))
    
# end timer
toc = time.perf_counter()

# Print results
runtime_32 = toc - tic
print(f'Runtime(32) = {runtime_32:0.4f} seconds')
# TODO Print accuracy of network

########################################################################

# Print score
print(f'FOM = {((runtime_1 + runtime_32) / 2):0.4f}')
# print(len(labels), len(outputs))
print("Accuracy", sum([a == b for a, b in zip(labels, outputs)])/ len(labels))







# class CNN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 5, kernel_size = 2, padding = (1,1), stride=1, bias = True)
    
#     def get_bias(self, layer):
#         if layer == 1:
#             return self.conv1.bias
    
#     def get_kernel(self, layer):
#         if layer == 1:
#             return self.conv1.weight

#     def forward(self, input_tensor):
#         return self.conv1(input_tensor), self.conv1.weight


# stars = '*' * 100
# model = CNN()
# input_tensor = torch.rand((1,3,3,3), requires_grad=True)
# output_torch = model.forward(input_tensor)[0]
# kernel = model.get_kernel(1)
# bias = model.get_bias(1)
# model_cpp = CNN_cpp()
# output_cpp = model_cpp.forward(input_tensor, kernel, bias)



# print(stars, "\n Weight Check ", stars, "\n", torch.equal(model.forward(input_tensor)[1], kernel ))
# print(stars, "\n output of Torch Forward", stars ,"\n", output_torch)
# print(stars, "\n Output of cpp Forward", stars, "\n", output_cpp)
# print(stars, "\n Output Check ", stars, "\n", torch.isclose(output_torch, output_cpp))
# print(stars, "\n Input grad of Torch Forward", stars ,"\n", input_grad)
# print(stars, "\n Input Grad of cpp Forward", stars, "\n", input_grad_cpp)
# print(stars, "\n Weight Check ", stars, "\n", torch.isclose(input_grad, input_grad_cpp ))
# 
# © 2021 GitHub, Inc.
# Terms
# Privacy
# Security
# Status
# Docs
# Contact GitHub
# Pricing
# API
# Training
# Blog
# About
