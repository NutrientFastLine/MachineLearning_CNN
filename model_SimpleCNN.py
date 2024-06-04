# -*- coding: UTF-8 -*- #
"""
@filename:model_SimpleCNN.py
@author:‘非瑜’
@time:2024-06-03
"""
import torch
from torch import nn


# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        """
            模型分为特征提取部分和分类部分
            特征提取部分包含所有的卷积层和池化层，用于从输入中提取特征
            分类部分包含全连接层，用于对提取的特征进行分类
        """
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷积层提取特征，输出32*28*28
            nn.ReLU(),  # 非线性化激活，使模型能够学习更复杂的特征
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 池化层对特征图进行下采样，减少特征图尺寸，同时保留重要特征，输出32*14*14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 输出64*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 输出64*7*7
            nn.Flatten()  # 展平
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=128),  # 全连接层对提取的特征进行分类
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 测试模块输入输出是否正确
if __name__ == '__main__':
    model = SimpleCNN()
    x1 = torch.ones((64, 1, 28, 28))
    output = model(x1)
    print(output.shape)
