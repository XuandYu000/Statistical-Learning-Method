'''
数据集: Mnist
训练集数量： 60000
测试集数量： 10000
------------------------------
运行结果: (topK=25)
欧几里得距离
正确率： 96.1%
运行时长：71.3s

曼哈顿距离
正确率: 95.3%
运行时长: 151.3s
'''

import numpy as np
import time

import pandas as pd
import torch


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
    features = torch.tensor(features, dtype=torch.float32).cuda()
    labels = torch.tensor(labels, dtype=torch.int32).cuda()

    return features, labels


def calcDist(trainData, x):
    '''
    计算测试样本与所有训练样本的距离
    :param trainData: 训练集数据
    :param x: 测试样本
    :return: 距离
    '''
    # 计算欧式距离
    # return torch.sqrt(torch.sum((trainData - x) ** 2, dim=1))
    # 计算曼哈顿距离
    return torch.sum(torch.abs(trainData - x), dim=1)

def getClosest(trainData, trainLabel, x, topK):
    '''
    获取最近的K个邻居
    :param trainData: 训练集数据
    :param trainLabel: 训练集标签
    :param x: 测试样本
    :param topK: K值
    :return: 预测标签
    '''
    # 计算距离
    distances = calcDist(trainData, x)

    # 获取最近的K个邻居
    _, indices = torch.topk(distances, topK, largest=False)
    label = trainLabel[indices]

    # 统计邻居中出现次数最多的标签
    count = torch.bincount(label)
    return torch.argmax(count)


def model_test(trainData, trainLabel, testData, testLabel, topK):
    '''
    KNN模型测试
    :param trainData: 训练集数据
    :param trainLabel: 训练集标签
    :param testData: 测试集数据
    :param testLabel: 测试集标签
    :param param: K值
    :return: 正确率
    '''
    with torch.no_grad():
        errorCount = 0
        for i in range(len(testData)):
            print('test %d:%d' % (i, len(testData)))
            x = testData[i]
            y = getClosest(trainData, trainLabel, x, topK)
            if y != testLabel[i]:
                errorCount += 1

        accuRate = 1 - errorCount / len(testData)

    return accuRate


if __name__ == "__main__":
    start = time.time()

    # 读取数据
    trainData, trainLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_train.csv")
    testData, testLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_test.csv")

    # 计算测试集正确率
    accuRate = model_test(trainData, trainLabel, testData, testLabel, 25)

    # 结束时间
    end = time.time()

    print("正确率：", accuRate)
    print("运行时长：", end - start)