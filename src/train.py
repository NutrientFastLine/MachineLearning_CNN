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
        # 获取参数
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_class = optimizer_class

        # 数据预处理
        self.testloader, self.trainloader = self._load_data()

        # 定义损失函数，以及初始化优化器
        self.criterion = nn.CrossEntropyLoss()  # 这里使用常用的交叉熵损失函数
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)

        # 定义训练的轮数
        self.epochs = 10

        # 创建模型保存以及可视化图形保存文件夹
        self.save_models_dir = '../saved_models'
        self._create_save_models_dir()
        self.save_images_dir = '../images'
        self._create_save_images_dir()

        # 获取模型保存文件名
        self.model_name = type(self.model).__name__
        self.filename_format = "{}_lr{}_batch{}_{}".format(self.model_name,
                                                           self.learning_rate,
                                                           self.batch_size,
                                                           self.optimizer_class.__name__)

    def _load_data(self):
        """
        加载训练数据

        返回:
        DataLoader: 训练数据加载器
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='../dataset', train=True, download=True,
                                                     transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='../dataset', train=False, download=True,
                                                    transform=transform)

        return DataLoader(testset, batch_size=self.batch_size, shuffle=True), \
            DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

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
        total_train_step = 0  # 记录训练的次数
        epoch_train_avrglosses = []  # 记录每一轮次平均的损失，用于绘制损失随轮次变化的曲线
        epoch_eval_avrglosses = []
        epoch_eval_avrgaccuracies = []

        for epoch in range(self.epochs):
            print("-------第 {} 轮训练开始-------".format(epoch + 1))
            epoch_train_loss = 0.0  # 每轮中的总损失清零
            self.model.train()  # 用于将模型切换到训练模式，一般主要用于Dropout层和BatchNorm层
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()  # 清空梯度
                outputs = self.model(images)  # 前向传播
                loss = self.criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 优化器更新参数

                epoch_train_loss += loss.item()
                total_train_step += 1
                if total_train_step % 100 == 0:
                    print("训练次数：{}, 当前Loss: {:.4f}".format(total_train_step, loss.item()))

            print("第 {} 轮训练结束, 平均Loss: {:.4f}".format(epoch + 1, epoch_train_loss / len(self.trainloader)))

            # 训练测试步骤开始
            self.model.eval()
            epoch_test_loss = 0
            epoch_accuracy = 0
            with torch.no_grad():
                for images, labels in self.testloader:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    accuracy = (outputs.argmax(1) == labels).sum().item() / labels.size(0)
                    epoch_test_loss += loss.item()
                    epoch_accuracy += accuracy

            epoch_train_avrglosses.append(epoch_train_loss / len(self.trainloader))
            epoch_eval_avrglosses.append(epoch_test_loss / len(self.testloader))
            epoch_eval_avrgaccuracies.append(epoch_accuracy / len(self.testloader))

        self._plot_results(epoch_train_avrglosses, epoch_eval_avrglosses, epoch_eval_avrgaccuracies)  # 绘制损失曲线
        self._save_model()  # 保存当前模型训练参数
        print('Finished Training')

    def _plot_results(self, epoch_train_avrglosses, epoch_eval_avrglosses, epoch_eval_avrgaccuracies):
        """
        可视化损失和准确率随批次变化的曲线，并保存图片

        参数:
        epoch_train_avrglosses (list): 每一轮次训练的损失
        epoch_eval_avrglosses (list): 每一轮次测试的损失
        epoch_eval_avrgaccuracies (list): 每一轮次测试的准确率
        """
        fig, ax1 = plt.subplots(figsize=(12, 5))

        epochs = range(1, len(epoch_train_avrglosses) + 1)

        ax1.plot(epochs, epoch_train_avrglosses, label='Train Loss', color='tab:blue')
        ax1.plot(epochs, epoch_eval_avrglosses, label='Eval Loss', color='tab:orange')
        ax1.set_xlabel('Epoch Number')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(epochs, epoch_eval_avrgaccuracies, label='Eval Accuracy', color='tab:green')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='upper right')

        plt.title(self.filename_format)  # 使用模型文件名作为标题

        fig.tight_layout()
        results_path = os.path.join(self.save_images_dir, f'{self.filename_format}_Train.png')
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
