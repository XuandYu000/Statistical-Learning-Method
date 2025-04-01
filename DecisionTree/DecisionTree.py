# '''
# 数据集：Mnist
# 训练集数量：60000
# 测试集数量：10000
# ------------------------------
# 运行结果：ID3(MaxDepth=10)
#     正确率： 83.6%
#     运行时长： 810.4s
# ------------------------------
# 运行结果：ID3(MaxDepth=10)
#     正确率： 85.8%
#     运行时长： 2817.4s
# '''

import numpy as np
import time
import pandas as pd
import torch
import os

def loadData(path, device='cuda'):
    """
    读取数据
    :param path: 数据集路径
    :param device: 设备
    :return: 数据和标签
    """
    data = torch.from_numpy(pd.read_csv(path, header=None).values)
    labels = data[:, 0].to(device, dtype=torch.int32)
    # 二值化
    features = (data[:, 1:].to(device, dtype=torch.int32) > 128).int()
    return features, labels


def calcBestFeature(features, labels):
    """
    计算信息增益
    :param features: 特征
    :param labels: 标签
    :return: 最佳特征和信息增益
    """
    numFeatures = features.shape[1]
    numLabels = len(torch.unique(labels))
    EpsilonGet = -1e10
    Ag = -1

    # 计算当前数据集的熵
    def calcEntropy(labels):
        p = torch.bincount(labels) / len(labels)
        return -torch.sum(p * torch.log2(p + 1e-10))

    baseEntropy = calcEntropy(labels)

    for i in range(numFeatures):
        # 计算每个特征的信息增益
        featureValues = torch.unique(features[:, i])
        EpsilonGet_i = 0

        for value in featureValues:
            subLabels = labels[features[:, i] == value]
            EpsilonGet_i += len(subLabels) / len(labels) * calcEntropy(subLabels)

        infoGain = baseEntropy - EpsilonGet_i
        if infoGain > EpsilonGet:
            EpsilonGet = infoGain
            Ag = i

    return Ag, EpsilonGet


def getSubDataArray(features, labels, Ag, param):
    """
    获取子数据集
    :param features: 特征
    :param labels: 标签
    :param Ag: 最佳特征
    :param param: 特征值
    :return: 子数据集
    """
    subFeatures = features[features[:, Ag] == param]
    subLabels = labels[features[:, Ag] == param]

    # 删除特征Ag
    subFeatures = torch.cat((subFeatures[:, :Ag], subFeatures[:, Ag + 1:]), dim=1)

    return subFeatures, subLabels


def createTree(*dataSet, maxDepth):
    """
    创建决策树
    :param dataSet: 数据集
    :param maxDepth: 最大深度
    :return: 决策树
    """
    Epsilon = 0.1
    # 计算每个特征的信息增益
    features, labels = dataSet[0]
    features = features.int()
    labels = labels.int()

    numFeatures = features.shape[1]
    uniqueLabels = torch.unique(labels).int()

    # 如果达到最大深度，返回出现次数最多的类
    if maxDepth <= 0:
        return torch.mode(labels).values.int().item()

    # 训练集D中所有实例均属于同一类，不需要再分类
    if len(uniqueLabels) == 1:
        return uniqueLabels[0].int().item()

    # 如果特征数为0，返回出现次数最多的类
    if numFeatures == 0:
        return torch.mode(labels).values.int().item()

    # 否则，计算A中特征值的信息增益，选最大的特征Ag
    Ag, EpsilonGet = calcBestFeature(features, labels)

    # 如果信息增益小于阈值，返回出现次数最多的类
    if EpsilonGet < Epsilon:
        return torch.mode(labels).values.int().item()

    # 创建子树
    treeDict = {Ag: {}}

    subFeatures, subLabels = getSubDataArray(features, labels, Ag, 0)
    subFeatures.int()
    subLabels.int()
    treeDict[Ag][0] = createTree(getSubDataArray(features, labels, Ag, 0), maxDepth=maxDepth - 1)

    subFeatures, subLabels = getSubDataArray(features, labels, Ag, 1)
    subFeatures.int()
    subLabels.int()
    treeDict[Ag][1] = createTree(getSubDataArray(features, labels, Ag, 1), maxDepth=maxDepth - 1)

    # print(treeDict)

    return treeDict


# def modelTest(tree, testData, testLabel):
#     """
#     测试决策树
#     :param tree: 决策树
#     :param testData: 测试数据
#     :param testLabel: 测试标签
#     :return: 准确率
#     """
#     testData = testData.int()
#     testLabel = testLabel.int()
#
#     numTest = testData.shape[0]
#     correct = 0
#
#     for i in range(numTest):
#         # 遍历测试数据，进行预测
#         node = tree
#         while isinstance(node, dict):
#             feature = list(node.keys())[0]
#             value = testData[i, feature].int().item()
#             node = node[feature][value]
#
#         print(f"Predicted: {node}, Actual: {testLabel[i].item()}")
#         if node == testLabel[i].int().item():
#             correct += 1
#
#     return correct / numTest

def predict(testDataList, tree):
    '''
    预测标签
    :param testDataList:样本
    :param tree: 决策树
    :return: 预测结果
    '''
    # treeDict = copy.deepcopy(tree)

    #死循环，直到找到一个有效地分类
    while True:
        #因为有时候当前字典只有一个节点
        #例如{73: {0: {74:6}}}看起来节点很多，但是对于字典的最顶层来说，只有73一个key，其余都是value
        #若还是采用for来读取的话不太合适，所以使用下行这种方式读取key和value
        (key, value), = tree.items()
        #如果当前的value是字典，说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':
            #获取目前所在节点的feature值，需要在样本中删除该feature
            #因为在创建树的过程中，feature的索引值永远是对于当时剩余的feature来设置的
            #所以需要不断地删除已经用掉的特征，保证索引相对位置的一致性

            dataVal = testDataList[key].item()
            testDataList = torch.cat((testDataList[:key], testDataList[key + 1:]))

            #将tree更新为其子节点的字典
            tree = value[dataVal]
            #如果当前节点的子节点的值是int，就直接返回该int值
            #例如{403: {0: 7, 1: {297:7}}，dataVal=0
            #此时上一行tree = value[dataVal]，将tree定位到了7，而7不再是一个字典了，
            #这里就可以直接返回7了，如果tree = value[1]，那就是一个新的子节点，需要继续遍历下去
            if type(tree).__name__ == 'int':
                #返回该节点值，也就是分类值
                return tree
        else:
            #如果当前value不是字典，那就返回分类值
            return value

def modelTest(tree, testDataList, testLabelList):
    '''
    测试准确率
    :param testDataList:待测试数据集
    :param testLabelList: 待测试标签集
    :param tree: 训练集生成的树
    :return: 准确率
    '''
    #错误次数计数
    errorCnt = 0
    #遍历测试集中每一个测试样本
    for i in range(len(testDataList)):
        #判断预测与标签中结果是否一致
        if testLabelList[i].item() != predict(testDataList[i], tree):
            errorCnt += 1
    #返回准确率
    return 1 - errorCnt / len(testDataList)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只让程序看到 GPU 1
    with torch.no_grad():
        torch.cuda.empty_cache()
        start = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 读取数据
        trainData, trainLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_train.csv", device)
        testData, testLabel = loadData("/root/Desktop/course/datasets/Mnist/mnist_test.csv", device)

        # 创建决策树
        print("Creating Decision Tree...")
        tree = createTree((trainData, trainLabel), maxDepth=100)
        print("Decision Tree created.")

        # 测试决策树
        print("Testing Decision Tree...")
        accRate = modelTest(tree, testData, testLabel)
        print("Decision Tree tested.")

        end = time.time()
        print(f"Accuracy: {accRate:.4f}")
        print(f"Time: {end - start:.4f} seconds")
