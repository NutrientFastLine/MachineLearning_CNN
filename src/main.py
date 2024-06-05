# -*- coding: UTF-8 -*- #
"""
@filename:main.py
@author:‘非瑜’
@time:2024-06-03
"""
import os
import torch
from train import ModelTrainer
from test import ModelTester
from model_SimpleCNN import SimpleCNN, SimpleCNNDropout
from model_LeNet5 import LeNet5
from model_vgg16 import CustomVGG16

# 选择使用的模型
# model = SimpleCNN()
model = SimpleCNNDropout()
# model = LeNet5()
# model = CustomVGG16()

# 选择学习率
# learning_rate = 1e-3
learning_rate = 1e-3

# 选择批次大小
batch_size = 64
# batch_size = 128

# 选择优化方法
optimizer_class = torch.optim.Adam
# optimizer_class = torch.optim.SGD

# 开始训练模型
trainer = ModelTrainer(model, learning_rate, batch_size, optimizer_class)
trainer.train()

# 获取模型保存路径
save_models_dir = '../saved_models'
model_name = type(model).__name__
filename_format = "{}_lr{}_batch{}_{}".format(model_name,
                                              learning_rate,
                                              batch_size,
                                              optimizer_class.__name__)
model_save_path = os.path.join(save_models_dir,  f'{filename_format}.pth')

# 开始测试模型
tester = ModelTester(model, model_save_path, batch_size)
tester.test()
