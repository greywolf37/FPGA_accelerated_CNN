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
print("1")
from torch.utils.cpp_extension import load
lltm_cpp = load(name="host", sources=["host.cpp"])
import host
print("2")
# This template should be followed to benchmark your solution

# Imports


# Initialization
# Perform any initialization procedures such as 
#   Binding
#   Flashing FPGA
#   Instantiating your FPGA accelerated VGG16 net
#   Any other initialization that shouldn't be timed
print("3")
# Get datasets
data_dir = Path('~/data').expanduser()
if not os.path.isdir(data_dir) :
    
    os.mkdir(data_dir)

    os.system('wget https://s3.amazonaws.com/ese680.imagenet/ILSVRC2012_devkit_t12.tar.gz -P ~/data/ --show-progress -nc')
    os.system('wget https://s3.amazonaws.com/ese680.imagenet/ILSVRC2012_devkit_t3.tar.gz -P ~/data/ --show-progress -nc')
    os.system('wget https://s3.amazonaws.com/ese680.imagenet/ILSVRC2012_img_val.tar -P ~/data/ --show-progress -nc')

print("4")
# Create dataset
input_size = 224
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageNet(data_dir, split='val', transform=test_transform)
print("5")
# Create dataloaders
batch1_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)
batch32_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=32)
print("6")
# Downloading the pretrained model
vgg16 = models.vgg16(pretrained=True)
stars = "*" * 100
print("7")
class fpga_cnn_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weights, bias):
        input_tensor = F.pad(input_tensor, (1,1,1,1))
        bias = bias.reshape((1,bias.shape[0],1,1))
        return lltm_cpp.forward_sw(input_tensor, weights) + bias

        
class fpga_conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias = True):
        super(fpga_conv2d, self).__init__( )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = torch.empty(out_channels)

    def forward(self, input_tensor):
        return fpga_cnn_function.apply(input_tensor, self.weight, self.bias)


class VGG16_custom(nn.Module):
    def __init__(self):
        super(VGG16_custom, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.features = nn.Sequential(
        fpga_conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        fpga_conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        fpga_conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        fpga_conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        fpga_conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        fpga_conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

print("8")
for param in vgg16.parameters():
    param.requires_grad = False
print("9")
model = VGG16_custom()
stars = '*' * 100
print(stars)
print(model)

stars = "*" *100
with torch.no_grad():
    for i in range(len(vgg16.features)):
        if str(vgg16.features[i]).startswith("Conv2d"):
            model.features[i].weight = vgg16.features[i].weight
            model.features[i].bias = vgg16.features[i].bias

    for j in range(len(vgg16.classifier)):
        if str(vgg16.classifier[j]).startswith("Linear"):
            model.classifier[j].weight = vgg16.classifier[j].weight
            model.classifier[j].bias = vgg16.classifier[j].bias

print("10")
model.eval()

#######################################################################
# Batch size 1 benchmarking
# Get a single (input, label) tuple of batch size 1 from your dataloader
# TODO


input, label = next(iter(batch1_loader))
# print(input)
# Run inference on the single (input, label) tuple
# TODO
print(stars, "\n"," BATCH 1 inference")
# print(stars, "\n", vgg16.features[10].weight, stars, "\n", vgg16.features[12].weight, stars, "\n", vgg16.features[14].weight, stars, "\n",)
with torch.no_grad():
  output = model(input)
  output_vgg = vgg16(input)
  print(output)

print(stars, "\n"," BACH 1")


# Start timer
tic = time.perf_counter()

# Run loop that performs 1024 inferences of batch size 1
labels, outputs = [], []
for i, data in zip(range(10), batch1_loader):
    # TODO
    input, label = data
    start = time.perf_counter()
    output = model(input)
    stop = time.perf_counter()
    labels.append(label.item())
    outputs.append(torch.argmax(output).item())
# end timer
toc = time.perf_counter()

# Print results
runtime_1 = toc - tic

print(f'Runtime(1) = {runtime_1:0.4f} seconds')
# TODO Print accuracy of network
# print(labels, outputs)
print("Accuracy", sum([a == b for a, b in zip(labels, outputs)])/ len(labels))


# stars = "*" *100
# print(stars, "\n"," BACH 31")


########################################################################
# Batch size 32 benchmarking
# Get a single (input, label) tuple of batch size 32 from your dataloader
# TODO
# input, label = load_item(1, batch1_loader) 
# model.eval()

# Run inference on the single (input, label) tuple
# TODO
print("Batch 32")
# # Start timer
# tic = time.perf_counter()
# outputs = []
# labels = []
# # Run loop that performs 1024 inferences of batch size 32
# for i, data in zip(range(2), batch32_loader):
#     # TODO
#     input, label = data
#     output = model(input)
#     for lab, out in zip(label,output):
#       outputs.append(torch.argmax(out).item())
#       labels.append(lab.item())
#       # print(len(outputs), "************", len(labels))
    
# # end timer
# toc = time.perf_counter()

# # Print results
# runtime_32 = toc - tic
# print(f'Runtime(32) = {runtime_32:0.4f} seconds')
# # TODO Print accuracy of network

# ########################################################################

# # Print score
# print(f'FOM = {((runtime_1 + runtime_32) / 2):0.4f}')
# # print(len(labels), len(outputs))
# print("Accuracy", sum([a == b for a, b in zip(labels, outputs)])/ len(labels))


##############################################################################################################
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