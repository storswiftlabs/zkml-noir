import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np

"""
in_channels = m
out_channels = n
image_shape = [x, x, in_channels]
filter_shape = [y, y, in_channels]
padding = p
stride = s
"""

# PART 1: Load Data and Set Hyperparameters

# Set hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# PyTorch will download MNIST data and save it to DATA_PATH
# The trained model will also be saved to MODEL_STORE_PATH
DATA_PATH = "/mnt/code/zkml-noir/static/DL/CNN/dataset"
MODEL_STORE_PATH = "/mnt/code/zkml-noir/static/DL/CNN/model"

# Transforms to apply to the data
# The Compose function from torchvision allows various transforms to be sequentially applied to the data
# First, specify the transform transforms.ToTensor() to convert data to PyTorch tensor
# PyTorch tensor is a special data type in PyTorch used for operations on data and weights in the network, essentially a multi-dimensional matrix
# Next, use transforms.Normalize to normalize the data, with parameters as the mean and standard deviation of the data
# MNIST data is single-channel, for multiple channels, you would need to provide mean and variance for each channel
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset, create train_dataset and test_dataset objects here
# root: the location of train.pt and test.pt data files; train: specify whether to get train.pt or test.pt data
# transform: apply transformations to the created data; download: download MNIST data from online
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# DataLoader object in PyTorch, can shuffle data, batch data, and load data in parallel using multiprocessing
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# PART 2: Create CNN Class

# Neural network structure:
# Input image is 28 x 28 single-channel
# First convolution: 32 channels of 5 x 5 convolutional filters, followed by ReLU activation
# and 2 x 2 max pooling (stride = 2, producing a 14 x 14 output)
# Second convolution: 64 channels of 5 x 5 convolutional filters, followed by ReLU activation
# and 2 x 2 max pooling (stride = 2, producing a 7 x 7 output)
# Flattened to 7 x 7 x 64 = 3164 nodes and connected to fully connected layers (1000 nodes)
# Finally, softmax operation on 10 output nodes to produce class probabilities

class ConvNet(nn.Module):
    # Initializing the network structure: defining the layers of the network
    def __init__(self):
        super(ConvNet, self).__init__()

        # Sequential method to create ordered layers
        # Conv2d method of nn.Module, creates a set of convolution filters
        # First argument is the number of input channels, second argument is the number of output channels
        # kernel_size: size of the convolution filter, here it is 5 x 5, so parameter is set to 5
        # For convolution filter of x * y, the parameter is a tuple (x, y)
        self.layer1 = nn.Sequential(
            # Dimensional change during convolution and pooling operation formula: width_of_output = (width_of_input - filter_size + 2 * padding) / stride + 1
            # Dimensional change during convolution: 28 - 5 + 2 * 2 + 1 = 28, added 2 padding to make output dimensions same as input, hence 2 padding
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
        # Two subsequent fully connected layers, each with 1000 nodes, 10 nodes correspond to 10 classes
        # The purpose of connecting these layers is to add the rich information output by the neural network to the standard classifier
        self.fc1 = nn.Linear(4 * 4 * 2, 10)
        self.fc2 = nn.Linear(10, 10)

    # Define the forward propagation of the network, this function will override the forward function in nn.Module
    # Input x goes through the layers of the network, and the output is out
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # Flattens the data dimensions from 7 x 7 x 64 into 3164 x 1
        # Fixed the side with -1 to have only one column
        out = out.reshape(out.size(0), -1)
        # Drop out some neural units with a certain probability to prevent overfitting
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# PART 3: Create an instance of the CNN
model = ConvNet()
# model.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))  # Load model weights


# This function includes SoftMax activation and cross entropy, so softmax activation does not need to be defined in the network structure definition
criterion = nn.CrossEntropyLoss()
# First argument: the parameters we want to train.
# In the nn.Module class, the method nn.parameters() allows PyTorch to track all the model parameters that need training in the CNN, so it knows which parameters to optimize
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# PART 4: Train the Model

# # Length of training dataset
# total_step = len(train_loader)
# loss_list = []
# acc_list = []
# for epoch in range(num_epochs):
#     # Loop through the training data (images, labels)
#     for i, (images, labels) in enumerate(train_loader):
#         # Input images to the network, get output, in this step the model.forward(images) function is automatically called by the model
#         outputs = model(images)
#         # Calculate loss
#         loss = criterion(outputs, labels)
#         loss_list.append(loss.item())

#         # Backpropagation, optimize using Adam
#         # First, zero the gradients of all parameters, otherwise they will accumulate
#         optimizer.zero_grad()
#         # Perform backpropagation
#         loss.backward()
#         # Update gradients
#         optimizer.step()

#         # Record accuracy
#         total = labels.size(0)
#         # torch.max(x, 1) takes the maximum along the rows
#         # The maximum value in each row is stored in _, and the index of the maximum value in each row is stored in predicted
#         # The value of each element in each row of the output represents the probability of that class, and the class corresponding to the maximum probability is the classification result
#         # In other words, find the index of the maximum probability
#         _, predicted = torch.max(outputs.data, 1)
#         # .sum() calculates the number of elements that are the same in predicted and labels, it returns a tensor, .item() gets the value of this tensor (int type)
#         correct = (predicted == labels).sum().item()
#         acc_list.append(correct / total)

#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
#                 .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

# PART 5: Test the Model

# Set the model to evaluation mode, disabling dropout or batch normalization layers in the model
model.eval()
# Disable autograd in the model, speeding up computation
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # print(outputs, predicted, labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # break
    print('Test Accuracy of the model on 10,000 test images: {} %'.format((correct / total) * 100))

# Save the model
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
