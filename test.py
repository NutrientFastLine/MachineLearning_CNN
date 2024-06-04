# -*- coding: UTF-8 -*- #
"""
@filename: test.py
@author: ‘非瑜’
@time: 2024-06-03
"""
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ModelTester:
    def __init__(self, model, model_save_path):
        """
        初始化模型测试类

        参数:
        model (torch.nn.Module): 要测试的模型
        model_save_path (str): 已训练模型参数的保存路径
        """
        self.model = model
        self.model_save_path = model_save_path

        # 数据预处理
        self.testset, self.testloader = self._load_data()

        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数，这里使用常用的交叉熵损失函数
        self._load_model()  # 加载训练好的模型参数
        self.filename_format = os.path.splitext(os.path.basename(model_save_path))[0]  # 获取去除后缀名的文件名

        self.save_images_dir = './images'  # 可视化图像保存路径
        self._create_save_images_dir()

    def _load_data(self):
        """
        加载测试数据

        返回:
        DataLoader: 测试数据加载器
        """
        if type(self.model).__name__ == 'SimpleCNN':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            print(type(self.model).__name__)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        testset = torchvision.datasets.FashionMNIST(root='./dataset', train=False, download=True,
                                                    transform=transform)
        return testset, DataLoader(testset, batch_size=64, shuffle=False)

    def _load_model(self):
        """
        加载训练好的模型参数
        """
        self.model.load_state_dict(torch.load(self.model_save_path))

    def _create_save_images_dir(self):
        """
        创建保存图片的文件夹
        """
        if not os.path.exists(self.save_images_dir):
            os.makedirs(self.save_images_dir)

    def test(self):
        """
        测试模型并输出结果
        """
        total_test_loss = 0  # 测试集整体损失，用于计算整体的平均损失
        total_accuracy = 0  # 测试集整体正确分类的数量，除以测试集样本数可以得到整体正确率
        total_samples = 0  # 记录测试集测试的样本数
        all_labels = []  # 存储所有标签值，用于构建混淆矩阵
        all_preds = []  # 存储所有预测值，用于构建混淆矩阵
        batch_losses = []  # 记录每一批次的损失，用于绘制损失随批次变化的曲线
        batch_accuracies = []  # 记录每一批次的准确率，用于绘制准确率随批次变化的曲线

        self.model.eval()  # 用于将模型切换到评估（推理）模式，一般主要用于Dropout层和BatchNorm层
        print("-------测试开始-------")
        with torch.no_grad():
            for images, labels in self.testloader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_loss = loss.item()
                # 计算在每一行中最大值的索引，找到每个样本的预测类别,之后将预测值与标签值对比，用sum求出所有准确数量。
                batch_accuracy = (outputs.argmax(1) == labels).sum().item() / labels.size(0)
                batch_losses.append(batch_loss)
                batch_accuracies.append(batch_accuracy)

                all_labels.extend(labels.cpu().numpy())  # 将标签值从tensor格式转换为numpy数组
                all_preds.extend(outputs.argmax(1).cpu().numpy())

                total_test_loss += batch_loss
                total_accuracy += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

                if total_samples % 640 == 0:  # 每次测试640个样本后打印一次
                    print("测试样本数：{}, 当前Loss: {:.4f}".format(total_samples, batch_loss))
                    print("测试样本数：{}, 当前accuracy: {:.4f}".format(total_samples, batch_accuracy))
        # 计算整体平均损失和准确率
        avg_loss = total_test_loss / len(self.testloader)
        avg_accuracy = total_accuracy / total_samples
        print("整体测试集上的Loss: {:.4f}".format(avg_loss))
        print("整体测试集上的正确率: {:.4f}".format(avg_accuracy))

        self._plot_results(batch_losses, batch_accuracies, avg_loss, avg_accuracy)
        self._plot_confusion_matrix(all_labels, all_preds)

    def _plot_results(self, batch_losses, batch_accuracies, avg_loss, avg_accuracy):
        """
        可视化损失和准确率随批次变化的曲线，并保存图片

        参数:
        batch_losses (list): 每一批次的损失
        batch_accuracies (list): 每一批次的准确率
        avg_loss (float): 平均损失
        avg_accuracy (float): 平均准确率
        """
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(batch_losses, label='Batch Loss')
        plt.axhline(y=avg_loss, color='r', linestyle='-', label=f'Avg Loss: {avg_loss:.4f}')  # 显示均值直线
        plt.title(self.filename_format)
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(batch_accuracies, label='Batch Accuracy')
        plt.axhline(y=avg_accuracy, color='g', linestyle='-', label=f'Avg Accuracy: {avg_accuracy:.4f}')
        plt.title(self.filename_format)
        plt.xlabel('Batch Number')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        results_path = os.path.join(self.save_images_dir, f'{self.filename_format}_LossAccuracy.png')
        plt.savefig(results_path)
        print(f'结果图已保存到: {results_path}')
        plt.show()

    def _plot_confusion_matrix(self, all_labels, all_preds):
        """
        绘制混淆矩阵

        参数:
        all_labels (list): 所有真实标签
        all_preds (list): 所有预测标签
        """
        conf_mat = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=self.testset.classes,
                    yticklabels=self.testset.classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(self.filename_format)

        confusion_matrix_path = os.path.join(self.save_images_dir, f'{self.filename_format}_ConfusionMatrix.png')
        plt.savefig(confusion_matrix_path)
        print(f'混淆矩阵已保存到: {confusion_matrix_path}')
        plt.show()
