import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms

import torchvision.datasets as datasets

import torch.nn.functional as F


# 配置设备

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置超参数

batch_size = 64

num_epochs = 10


# 数据集加载和预处理

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))

])


trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# 定义2种不同的网络结构

class SimpleCNN(nn.Module):#简单的卷积神经网络，包含一个卷积层和两个全连接层。

    def __init__(self):

        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        self.fc1 = nn.Linear(32 * 26 * 26, 128)

        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = x.view(-1, 32 * 26 * 26)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

class ComplexCNN(nn.Module):#复杂的卷积神经网络，包含两个卷积层和两个全连接层。

    def __init__(self):
        super(ComplexCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)

        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 12 * 12)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x




# 定义训练和测试函数

def train_model(model, trainloader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:

                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')

                running_loss = 0.0


def test_model(model, testloader):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for inputs, labels in testloader:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# 定义不同学习率和网络结构的组合

learning_rates = [0.001, 0.0005 ,0.0001]

network_structures = [SimpleCNN, ComplexCNN]


# 训练和评估每种组合

results = {}

for lr in learning_rates:

    for net in network_structures:

        print(f'Training {net.__name__} with learning rate {lr}')

        model = net().to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_model(model, trainloader, criterion, optimizer, num_epochs)

        accuracy = test_model(model, testloader)

        print(f'Accuracy of {net.__name__} with learning rate {lr}: {accuracy:.2f}%')

        results[(net.__name__, lr)] = accuracy


# 打印最终结果

for (net_name, lr), acc in results.items():

    print(f'Network: {net_name}, Learning Rate: {lr}, Accuracy: {acc:.2f}%')

