'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
层数：40
------------------------------
运行结果：
    正确率：97.7%
    运行时长：98.5s
'''
import os
import time
from pyexpat import features

import pandas as pd
import torch


def loadData(path, num = None, device="cpu"):
    """
    加载数据集
    :param path: 数据集路径
    :param num: 读取数据集的数量
    :param device: 设备类型
    :return: 数据集和标签
    """
    if num is None:
        data = torch.from_numpy(pd.read_csv(path, header=None).values)
    else:
        data = torch.from_numpy(pd.read_csv(path, header=None).values[:num])

    # 标签设置为二分类-1，1
    label = data[:, 0].to(device, dtype=torch.int8)
    label = torch.where(label == 0, -1, 1).to(device)

    # 二值化
    features = data[:, 1:].to(device, dtype=torch.float32)
    features = torch.where(features > 128, 1.0, 0.0).to(device)

    return features, label

class DecisionTree:
    def __init__(self, trainData, trainLabel, D, device="cpu"):
        """
        决策树
        :param trainData: 训练数据
        :param trainLabel: 训练标签
        :param D: 权重
        :param device: 设备类型
        """
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.D = D
        self.device = device

    def fit(self):
        """
        决策树训练函数
        :return: None
        """
        m, n = self.trainData.shape
        # 单层树字典，存放当前决策树的参数
        singleTree = {}
        # 初始化错误率为一个极大的数
        singleTree['error'] = 1e10

        # 遍历每个特征，寻找合适的特征划分
        for feature in range(n):
            # 遍历每个特征的取值，特征已二值化为0， 1，选取-0.5, 0.5, 1.5作为划分点
            for value in [-0.5, 0.5, 1.5]:
                # 特征划分两种情况都需要遍历：
                # 1. 特征值小于value，标记为-1
                # 2. 特征值小于value，标记为1
                for rule in [-1, 1]:
                    # 计算当前特征划分的错误率
                    Gx = torch.where(trainData[:, feature] < value, 1.0 * rule, -1.0 * rule)
                    error = torch.sum((Gx != self.trainLabel) * self.D)
                    # 如果当前错误率小于最小错误率，则更新单层树的参数
                    if error < singleTree['error']:
                        singleTree['error'] = error
                        singleTree['feature'] = feature
                        singleTree['value'] = value
                        singleTree['rule'] = rule

        self.singleTree = singleTree

    def predict(self, testData):
        """
        决策树预测函数
        :param testData: 测试数据
        :return: 预测结果
        """
        # 获取单层树的参数
        feature = self.singleTree['feature']
        value = self.singleTree['value']
        rule = self.singleTree['rule']

        # 遍历每个测试样本，进行预测
        predict = torch.where(testData[:, feature] < value, 1.0 * rule, -1.0 * rule)

        return predict

    def setattr(self, attr, value):
        """
        设置单层树的属性
        :param attr: 属性名称
        :param value: 属性值
        :return: None
        """
        self.singleTree[attr] = value

def AdaBoostTree(trainData, trainLabel, numTrees=40, device="cpu"):
    """
    AdaBoost算法 算法8.1
    :param trainData: 训练数据
    :param trainLabel: 训练标签
    :param testData: 测试数据
    :param testLabel: 测试标签
    :param numTrees: 树的数量
    :return: 正确率和运行时间
    """
    finalPredict = torch.zeros(trainData.shape[0], device=device)
    AdaBoostTree = []
    m, n = trainData.shape

    # (1) 初始化权重
    D = torch.ones(m, device=device) / m

    # (2) 循环创建提升树
    for i in range(numTrees):
        # (2.1) 创建决策树
        tree = DecisionTree(trainData, trainLabel, D, device)
        tree.fit()
        # (2.2) 计算错误率
        error = tree.singleTree['error']
        # (2.3) 计算alpha值
        alpha = 0.5 * torch.log((1 - error) / error)
        tree.setattr('alpha', alpha)
        # (2.4) 更新权重
        Gx = tree.predict(trainData)
        D = D * torch.exp(-1 * alpha * trainLabel * Gx)
        D /= torch.sum(D)
        # (2.5) 保存树
        AdaBoostTree.append(tree)

        # -----辅助代码-----
        finalPredict += alpha * Gx
        # 当前预测与实际标签的误差
        error = torch.sum(trainLabel != torch.sign(finalPredict)) / m
        # 如果误差为0，则停止训练
        if error == 0:
            return AdaBoostTree

        print(f"iter: {i + 1} : {numTrees}, single error: {tree.singleTree['error']}, final error: {error.item()}, feature: {tree.singleTree['feature']}" )

    # (3) 返回提升树
    return AdaBoostTree

def test_model(tree, testData, testLabel, device="cpu"):
    """
    测试函数
    :param tree: 提升树
    :param testData: 测试数据
    :param testLabel: 测试标签
    :return: 正确率
    """
    finalPredict = torch.zeros(testData.shape[0], device=device)
    for i in range(len(tree)):
        # 预测结果
        Gx = tree[i].predict(testData)
        # alpha值
        alpha = tree[i].singleTree['alpha']
        finalPredict += alpha * Gx

    # 计算正确率
    accuRate = torch.sum(testLabel == torch.sign(finalPredict)) / testLabel.shape[0]
    return accuRate.item()

if __name__ == "__main__":
    start = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 读取数据
        trainData, trainLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_train.csv", None, device)
        testData, testLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_test.csv", None, device)

        # 提升树创建
        print("start training")
        tree = AdaBoostTree(trainData, trainLabel, 40, device)

        # 测试
        print("start testing")
        accuRate = test_model(tree, testData, testLabel, device)
        print(f"accuRate: {accuRate}")
        end = time.time()
        print(f"run time: {end - start:.2f}s")