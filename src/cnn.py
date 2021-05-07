import math
import torch
from torch import nn
from torch.utils.cpp_extension import load

print("1")
lltm_cpp = load(name="host", sources=["host.cpp"])
print("2")
import host
print("3")
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size = 2, stride=1, bias = False)
    
    def get_kernel(self, layer):
        if layer == 1:
            return self.conv1.weight
    def forward(self, input_tensor):
        return self.conv1(input_tensor), self.conv1.weight

class CNN_cpp(torch.nn.Module):
    def __init__(self):
        super(CNN_cpp, self).__init__()
        self.inputs_size = 4

    def forward(self, input_tensor, kernel):
        return lltm_cpp.forward_sw(input_tensor, kernel)

model = CNN()
input_tensor = torch.rand((1,3,3,3))
ouput_torch = model.forward(input_tensor)[0]
kernel = model.get_kernel(1)
print("5")
model_cpp = CNN_cpp()
ouput_cpp = model_cpp.forward(input_tensor, kernel)
print("6")
stars = '*' * 100
print(stars, "\n Weight Check ", stars, "\n", torch.equal(model.forward(input_tensor)[1], kernel ))
print(stars, "\n Ouput of Torch Forward", stars ,"\n", ouput_torch)
print(stars, "\n Output of cpp Forward", stars, "\n", ouput_cpp)
print(stars, "\n Output Check ", stars, "\n", torch.isclose(ouput_torch, ouput_cpp))