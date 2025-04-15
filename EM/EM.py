'''
数据集：伪造数据集（两个高斯分布混合）
数据集长度：1000
------------------------------
运行结果：
----------------------------
the Parameters set is:
alpha0:0.5, mu0:0., sigmod0:1.0, alpha1:0.5, mu1:3.0, sigmod1:1.0
----------------------------
the Parameters predict is:
alpha0:0.5029, mu0:-0.0185, sigmod0:1.0050, alpha1:0.4971, mu1:3.0245, sigmod1:0.9614
----------------------------
'''

import time
import torch
import numpy as np

def loadData(K, dataLen, mu, sigma, alpha):
    """
    生成数据
    :param K: 高斯模型个数
    :param dataLen: 数据长度
    :param mu: 均值
    :param sigma: 方差
    :param alpha: 权重系数
    :return: 生成的数据
    """
    # 生成数据
    data = []
    for i in range(K):
        # 生成数据
        data.extend(torch.normal(mu[i], sigma[i], size=(int(dataLen * alpha[i]),)))


    return torch.tensor(data, device=mu.device)

def EM(dataSet, K, alpha, mu, sigma, iter=500):
    """
    EM算法
    :param dataSet: 数据集
    :param K: 高斯模型个数
    :param alpha: 权重系数
    :param mu: 均值
    :param sigma: 方差
    :param iter: 迭代次数
    :return: 估计的参数
    """
    # 计算数据长度
    dataLen = len(dataSet)

    # 迭代次数
    for i in range(iter):
        # E步，计算每个数据点属于每个高斯模型的概率
        prob = torch.zeros((dataLen, K), device=mu.device)
        for j in range(K):
            prob[:, j] = alpha[j] * torch.exp(-0.5 * ((dataSet - mu[j]) / sigma[j]) ** 2) / (sigma[j] * np.sqrt(2 * np.pi))

        # 计算响应度
        probSum = prob.sum(dim=1, keepdim=True)
        prob /= probSum

        # M步，更新参数
        # 更新alpha
        alpha = prob.mean(dim=0)

        # 更新mu
        mu = (prob * dataSet.unsqueeze(1)).sum(dim=0) / prob.sum(dim=0)

        # 更新sigma
        sigma = torch.sqrt((prob * (dataSet.unsqueeze(1) - mu) ** 2).sum(dim=0) / prob.sum(dim=0))

    return alpha, mu, sigma


if __name__ == "__main__":
    begin = time.time()
    K = 2
    dataLen = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置两个高斯模型进行混合，这里是初始化两个模型各自的参数
    # 见“9.3 EM算法在高斯混合模型学习中的应用”
    # alpha是“9.3.1 高斯混合模型” 定义9.2中的系数α
    # mu0是均值μ
    # sigmod是方差σ
    # 在设置上两个alpha的和必须为1，其他没有什么具体要求，符合高斯定义就可以
    alpha = torch.tensor([0.5, 0.5], device=device)
    mu = torch.tensor([0.0, 3.0], device=device)
    sigma = torch.tensor([1.0, 1.0], device=device)

    # 生成数据
    dataSet = loadData(K, dataLen, mu, sigma, alpha)

    # 打印设置参数
    print('--' * 20)
    print("alpha: ", alpha)
    print("mu: ", mu)
    print("sigma: ", sigma)

    # EM算法，参数估计
    alphaE, muE, sigmaE = EM(dataSet, K, alpha, mu, sigma)

    # 打印估计参数
    print('--' * 20)
    print("alphaE: ", alphaE)
    print("muE: ", muE)
    print("sigmaE: ", sigmaE)
    print('--' * 20)
    print("Time: ", time.time() - begin)