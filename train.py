# -*- coding: UTF-8 -*- #
"""
@filename:train.py
@author:‘非瑜’
@time:2024-06-03
"""
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model, learning_rate, batch_size, optimizer_class):
        """
        初始化模型训练类

        参数:
        model (torch.nn.Module): 要训练的模型
        learning_rate (float): 学习率
        batch_size (int): 批次大小
        optimizer_class (torch.optim.Optimizer): 优化器类
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_class = optimizer_class

        self.model_name = type(self.model).__name__
        self.filename_format = "{}_lr{}_batch{}_{}".format(self.model_name,
                                                           self.learning_rate,
                                                           self.batch_size,
                                                           self.optimizer_class.__name__)

        # 数据预处理
        self.trainloader = self._load_data()

        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数，这里使用常用的交叉熵损失函数
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)  # 定义优化器

        self.total_train_step = 0  # 训练的总次数
        self.epochs = 10  # 训练的轮数

        # 创建模型保存以及可视化图形保存路径
        self.save_models_dir = './saved_models'
        self._create_save_models_dir()
        self.save_images_dir = './images'
        self._create_save_images_dir()

    def _load_data(self):
        """
        加载训练数据

        返回:
        DataLoader: 训练数据加载器
        """
        if self.model_name == 'SimpleCNN':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            print(self.model_name)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        trainset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True,
                                                     transform=transform)

        return DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

    def _create_save_models_dir(self):
        """
        创建保存路径
        """
        if not os.path.exists(self.save_models_dir):
            os.makedirs(self.save_models_dir)

    def _create_save_images_dir(self):
        """
        创建保存图片的文件夹
        """
        if not os.path.exists(self.save_images_dir):
            os.makedirs(self.save_images_dir)

    def train(self):
        """
        训练模型
        """
        epoch_avrglosses = []  # 记录每一轮次平均的损失，用于绘制损失随轮次变化的曲线

        for epoch in range(self.epochs):
            print("-------第 {} 轮训练开始-------".format(epoch + 1))
            running_loss = 0.0  # 每轮中的总损失
            self.model.train()
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()  # 清空梯度
                outputs = self.model(images)  # 前向传播
                loss = self.criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 优化器更新参数

                running_loss += loss.item()
                self.total_train_step += 1
                if self.total_train_step % 100 == 0:
                    print("训练次数：{}, 当前Loss: {:.4f}".format(self.total_train_step, loss.item()))

            epoch_avrglosses.append(running_loss / len(self.trainloader))
            print("第 {} 轮训练结束, 平均Loss: {:.4f}".format(epoch + 1, running_loss / len(self.trainloader)))

        print('Finished Training')
        self._plot_results(epoch_avrglosses)  # 绘制损失曲线
        self._save_model()  # 保存当前模型训练参数

    def _plot_results(self, epoch_avrglosses):
        """
        可视化损失和准确率随批次变化的曲线，并保存图片

        参数:
        epoch_losses (list): 每一轮次的损失
        """
        plt.figure(figsize=(12, 5))

        plt.plot(epoch_avrglosses, label='Epoch AvrgLoss')
        plt.title(self.filename_format)  # 使用模型保存路径名称作为标题
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        results_path = os.path.join(self.save_images_dir, f'{self.filename_format}_Loss.png')
        plt.savefig(results_path)
        print(f'结果图已保存到: {results_path}')
        plt.show()

    def _save_model(self):
        """
        保存模型参数
        """
        model_save_path = os.path.join(self.save_models_dir, f'{self.filename_format}.pth')
        torch.save(self.model.state_dict(), model_save_path)
        print('模型已保存到: {}'.format(model_save_path))
