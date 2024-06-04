# -*- coding: UTF-8 -*- #
"""
@filename:model_vgg16.py
@author:‘非瑜’
@time:2024-06-04
"""
import torch
import torchvision
import torch.nn as nn


# 定义自定义的 VGG16 模型以适应灰度图像和10个类别
class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        self.features = torchvision.models.vgg16().features  # 加载特征层
        self.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 修改输入层
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 保持自适应平均池化层
        self.classifier = torchvision.models.vgg16().classifier  # 加载分类层
        self.classifier[-1] = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 测试模块输入输出是否正确
if __name__ == '__main__':
    model = CustomVGG16()  # 实例化模型
    x1 = torch.ones((64, 1, 224, 224))  # 使用灰度图像
    output = model(x1)  # 调用实例的方法
    print(output.shape)  # 应该输出 torch.Size([64, 10])
    print(type(model).__name__)
