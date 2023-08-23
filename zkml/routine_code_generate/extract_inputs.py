import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from zkml.routine_code_generate.utils import *
from zkml.quantization.quantize import calc_scale, calc_zero_point, quantize, dequantize, quantize_all
import math

"""
in_channels = m
out_channels = n
image_shape = [x, x, in_channels]
filter_shape = [y, y, in_channels]
padding = p
stride = s
"""

num_epochs = 5
num_classes = 10
batch_size = 1
learning_rate = 0.001

DATA_PATH = "/mnt/code/zkml-noir/static/DL/CNN/dataset"
MODEL_STORE_PATH = "/mnt/code/zkml-noir/static/DL/CNN/model"

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module):
    # Initializing the network structure: defining the layers of the network
    def __init__(self):
        super(ConvNet, self).__init__()

        # Using Sequential to create ordered layers
        # Conv2d is a method of nn.Module, which creates a set of convolution filters
        # The first argument is the number of input channels, and the second argument is the number of output channels
        # kernel_size: size of the convolution filter, here it is 5*5, so the parameter is set to 5
        # If the convolution filter is x*y, the parameter is a tuple (x, y)
        self.layer1 = nn.Sequential(
            # Dimensional change during convolution operation & pooling operation: width_of_output = (width_of_input - filter_size + 2*padding) / stride + 1
            # Dimensional change during convolution: 28 - 5 + 2*2 + 1 = 28, I want the output of the convolution to have the same dimensions as the input, so I added 2 padding
            nn.Conv2d(1, 5, kernel_size=3, stride=3, padding=1),
            # Activation function
            nn.ReLU(),
            # kernel_size: pooling size, stride: down-sample
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))

        self.drop_out = nn.Dropout()
        # The next two fully connected layers, each with 1000 nodes, and 10 nodes corresponding to 10 categories
        # The purpose of the connected layers is to add the rich information output by the neural network to the standard classifier
        self.fc1 = nn.Linear(4 * 4 * 2, 10)
        self.fc2 = nn.Linear(10, 10)

    # Defining the forward pass of the network, this function will override the forward function in nn.Module
    # Input x goes through the layers of the network, and the output is out
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # Flattening the data dimensions from 7 x 7 x 64 into 3164 x 1
        # Fixing the side with -1 to have only one column
        out = out.reshape(out.size(0), -1)
        # Drop out some neural units with a certain probability to prevent overfitting
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()
model.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))  # Loading model weights

# for key in model.state_dict().keys():
#     print(key)
#     print(model.state_dict()[key])

inputs = {}
all_values = [0]

for i, key in enumerate(model.state_dict().keys()):
    layer_num = int(i / 2)
    name = "inputs_" + str(layer_num)
    value = model.state_dict()[key]
    shape = value.shape
    if len(shape) == 4:
        inputs[name] = [0 for n in range(shape[1] * shape[2] * shape[3])]
    elif len(shape) == 2:
        inputs[name] = [0 for n in range(shape[1])]
    print(value.shape)
    value_list = value.view(-1).numpy().tolist()
    if 'weight' in key:
        inputs[name].extend(value_list)
    if 'bias' in key:
        inputs[name].extend(value_list)
        # for j in range(len(inputs[name])):
        #     inputs[name][j] = int(inputs[name][j]*2**8+2**8)
        #     if inputs[name][j]<0 or inputs[name][j]>2**16:
        #         raise Exception('---')
    all_values.extend(value_list)
    print(len(inputs[name]))

# print(inputs)

for images in test_loader:
    name = 'image'
    inputs[name] = images[0].view(-1).numpy().tolist()
    # for j in range(len(inputs[name])):
    #     inputs[name][j] = int(inputs[name][j] * 2**8 + 2**8)
    #     if inputs[name][j] < 0 or inputs[name][j] > 2**16:
    #         raise Exception('---')
    outputs = model(images[0])
    _, predicted = torch.max(outputs.data, 1)
    print(outputs.data, images[1])
    break

_type = "uint32"
scale_molecule, scale_denominator = calc_scale(all_values, _type)
scale = math.ceil(scale_molecule / scale_denominator)
_zero_point = calc_zero_point(all_values, scale, _type)
scale = int(scale_molecule / scale_denominator)
# print("quantize", quantize(all_values, scale, _zero_point, _type))
scale = 2**32
_zero_point = 2**64
print("quantize", scale, _zero_point, _type)
for k in inputs:
    v = inputs[k]
    # v = quantize(v, scale, _zero_point, _type)
    v = [int(a * scale + _zero_point) for a in v]
    # v = array_int2str(v)
    inputs[k] = v
prover_str = '\n'.join([k + ' = ' + array_int2str(inputs[k]) for k in inputs])

path = '/mnt/code/noir_project/cnn2'

with open(os.path.join(path, "Prover.toml"), "w+") as file:
    file.write(prover_str)

hex_list = ["0xffffffffc93a26a5","0x0100000000397d8638","0x01000000001d8a1fd3","0x01000000006243f801","0x010000000072280992","0x010000000035275ad9","0xffffffffae7fff60","0x0100000000215b2dc9","0x010000000009f5e4ea","0xffffffffca4332f9"]

decimal_list = [int(hex_string, 16) for hex_string in hex_list]
dequantized_list = [(v-_zero_point)/scale for v in decimal_list]
print([(v-_zero_point)/scale for v in decimal_list])
print(dequantized_list[0]*dequantized_list[1])
exponent = 0
target = decimal_list[-1]

while 2 ** exponent < target:
    exponent += 1

print("2的%d次方等于%d" % (exponent, target))
print("Decimal list:", decimal_list, decimal_list.index(max(decimal_list)))
