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
pading = p
stride = s
"""


# PART 1：加载数据，设置超参

#超参设置
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

#pytorch 将会下载MNIST数据保存到DATA_PATH
#也会将训练好的模型保存到MODEL_STORE_PATH
DATA_PATH = "/mnt/code/zkml-noir/static/DL/CNN/dataset"
MODEL_STORE_PATH = "/mnt/code/zkml-noir/static/DL/CNN/model"

# transforms to apply to the data
# transforms.Compose函数来自于torchvision包
# 用Compose可以将各种transforms有序组合到一个list中
# 首先指定一个转换transforms.ToTensor()，将数据转换为pytorch tensor
# pytorch tensor是pytorch中特殊的数据类型，用于网络中数据和权重的操作，本质上是多维矩阵
# 接下来用transforms.Normalize对数据进行归一化，参数为数据的平均数和标准差
# MNIST数据是单通道的，多通道就需要提供每个通道的均值和方差
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset，这里创建了train_dataset和test_dataset对象
# root：train.pt和test.pt数据文件位置；train：指定获取train.pt或者test.pt数据
# tranform：对创建的数据进行操作transform操作；download：从线上下载MNIST数据
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# pytorch中的DataLoader对象，可以对数据洗牌，批处理数据，多处理来并行加载数据
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# PART 2：创建CNN类

# 神经网络的结构：
# 输入图片为 28*28 单通道
# 第一次卷积：32 channels of 5 x 5 convolutional filters，a ReLU activation
# followed by 2 x 2 max pooling(stride = 2，this gives a 14 x 14 output)
# 第二次卷积：64 channels of 5 x 5 convolutional filters
# 2 x 2 max pooling (stride = 2，produce a 7 x 7 output) 
# 展开需要节点：7 x 7 x 64 = 3164 个，接上全连接层（含1000个节点）
# 最后对10个输出节点进行softmax操作，产生分类概率

class ConvNet(nn.Module):
    # 初始化定义网络的结构：也就是定义网络的层
    def __init__(self):
        super(ConvNet,self).__init__()

        # Sequential方法使我们有序的创建网络层
        # Conv2d nn.Module的方法，该方法创建一组卷积滤波器，
        # 第一个参数为输入的channel数，第二个参数为输出的channel数
        # kernel_size：卷积滤波器的尺寸，这里卷积滤波器为5*5，所以参数设置为5
        # 如果卷积滤波器为 x*y，参数就是一个元组(x,y)
        self.layer1 = nn.Sequential(
            # 卷积操作&池化操作的维度变化公式: width_of_output = (width_of_input - filter_size + 2*padding)/stride + 1
            # 卷积操作时维度变化：28-5+2*2+1 =28，我希望卷积的输出和输出维度一样，所以加了2 padding
            nn.Conv2d(1,5,kernel_size=3,stride=3,padding=1),
            # 激活函数
            nn.ReLU(),
            # kernel_size：pooling size，stride：down-sample
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(5,2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1))

        self.drop_out = nn.Dropout()
        # 后面两个全连接层，分别有1000个节点，10个节点对应10种类别
        # 接全连接层的意义是：将神经网络输出的丰富信息加到标准分类器中
        self.fc1 = nn.Linear(4*4*2,10)
        self.fc2 = nn.Linear(10,10)

    # 定义网络的前向传播,该函数会覆盖 nn.Module 里的forward函数
    # 输入x,经过网络的层层结构，输出为out
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        # flattens the data dimensions from 7 x 7 x 64 into 3164 x 1
        # 左行右列，-1在哪边哪边固定只有一列
        out = out.reshape(out.size(0),-1)
        # 以一定概率丢掉一些神经单元，防止过拟合
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# PART 3：创建一个CNN实例
model = ConvNet()
# model.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))  # 加载模型权重


# 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
criterion = nn.CrossEntropyLoss()
# 第一个参数:我们想要训练的参数。
# 在nn.Module类中，方法 nn.parameters()可以让pytorch追踪所有CNN中需要训练的模型参数，让他知道要优化的参数是哪些
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# PART 4：训练模型

#训练数据集长度
total_step = len(train_loader) 
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    # 遍历训练数据(images,label)
    for i,(images,labels) in enumerate(train_loader):
        # 向网络中输入images，得到output,在这一步的时候模型会自动调用model.forward(images)函数
        outputs = model(images)
        # 计算这损失
        loss = criterion(outputs,labels)
        loss_list.append(loss.item())

        # 反向传播，Adam优化训练
        # 先清空所有参数的梯度缓存，否则会在上面累加
        optimizer.zero_grad()
        # 计算反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()

        # 记录精度
        total = labels.size(0)
        # torch.max(x,1) 按行取最大值
        # output每一行的最大值存在_中，每一行最大值的索引存在predicted中
        # output的每一行的每个元素的值表示是这一类的概率，取最大概率所对应的类作为分类结果
        # 也就是找到最大概率的索引
        _,predicted = torch.max(outputs.data,1)
        # .sum()计算出predicted和label相同的元素有多少个，返回的是一个张量，.item()得到这个张量的数值(int型)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct/total)

        if (i+1) % 100 == 0:
            print('Epoch[{}/{}],Step[{},{}],Loss:{:.4f},Accuracy:{:.2f}%'
            .format(epoch+1,num_epochs,i+1,total_step,loss.item(),(correct/total)*100))

# PART 5：测试模型

#将模型设为评估模式，在模型中禁用dropout或者batch normalization层
model.eval()
# 在模型中禁用autograd功能，加快计算
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        # print(outputs, predicted, labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # break
    print('Test Accuracy of the model on the 1w test images:{} %'.format((correct / total) * 100))

# save the model
torch.save(model.state_dict(),MODEL_STORE_PATH + 'conv_net_model.ckpt')
