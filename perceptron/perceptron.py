'''
数据集: Mnist
训练集数量： 60000
测试集数量： 10000
------------------------------
运行结果: 
正确率： 86.8%
运行时长： 99s
'''

import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import TensorDataset, DataLoader

def loadData(path):
    '''
    加载数据并转换为PyTorch张量
    :param path: 数据集路径
    :param device: 计算设备 (cpu/cuda)
    :return: 数据集和标签 (PyTorch张量)
    '''
    data = pd.read_csv(path, header=None)
    labels = data.iloc[:, 0].values
    features = data.iloc[:, 1:].values / 255

    # to GPU
    features = torch.tensor(features , dtype=torch.float32).cuda()
    # 感知机是一个二元分类器，标签需要转换为-1和1
    labels = torch.tensor(np.where(labels >= 5, 1, -1), dtype=torch.float32).cuda()

    return features, labels

def perceptron(trainData, trainLabel, iter = 30, lr = 0.0001):
    '''
    感知机训练 (GPU加速版)
    :param trainData: 训练集 (PyTorch张量)
    :param trainLabel: 训练标签 (PyTorch张量)
    :param iter: 迭代次数
    :param lr: 学习率
    :return: 权重和偏置
    '''
    n, m = trainData.shape
    
    # 初始化参数 (自动放在与数据相同的设备上)
    w = torch.zeros(m, dtype=torch.float32, device=trainData.device)
    b = torch.zeros(1, dtype=torch.float32, device=trainData.device)
    
    # 使用DataLoader提高数据加载效率
    dataset = TensorDataset(trainData, trainLabel)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    for _ in range(iter):
        for x, y in dataloader:
            # print(x.shape)
            # 向量化计算
            predictions = torch.matmul(x, w) + b
            misclassified = (y * predictions) <= 0
            
            if torch.any(misclassified):
                # 只更新错误分类的样本
                update = lr * y[misclassified]
                w += torch.matmul(x[misclassified].T, update).squeeze()
                b += torch.sum(update)
    
    return w, b

def model_test(testData, testLabel, w, b):
    '''
    测试准确率 (GPU加速版)
    :param testData: 测试集 (PyTorch张量)
    :param testLabel: 测试标签 (PyTorch张量)
    :param w: 权重
    :param b: 偏置
    :return: 准确率
    '''
    with torch.no_grad():
        predictions = torch.matmul(testData, w) + b
        correct = (testLabel * predictions) > 0
        accuRate = correct.float().mean().item()
    
    return accuRate

if __name__== '__main__':
    # 当前时间
    start = time.time()

    # 读取数据
    trainData, trainLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_train.csv")
    testData, testLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_test.csv")

    # 训练获得权重
    w, b = perceptron(trainData, trainLabel, iter = 50)

    # 预测
    accuRate = model_test(testData, testLabel, w, b)

    # 结束时间
    end = time.time()

    print("正确率：", accuRate)
    print("运行时长：", end-start)