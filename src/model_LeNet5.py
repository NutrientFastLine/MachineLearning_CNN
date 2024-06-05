# -*- coding: UTF-8 -*- #
"""
@filename:model_LeNet5.py
@author:‘非瑜’
@time:2024-06-04
"""
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 特征提取部分
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 16 * 5 * 5)  # 展平特征图
        x = self.classifier(x)
        return x


# 测试模块输入输出是否正确
if __name__ == '__main__':
    model = LeNet5()  # 实例化模型
    x1 = torch.ones((64, 1, 28, 28))
    output = model(x1)  # 调用实例的方法
    print(output.shape)  # 应该输出 torch.Size([64, 10])
