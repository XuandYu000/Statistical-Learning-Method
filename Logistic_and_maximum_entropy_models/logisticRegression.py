'''
数据集: MNIST
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：86.6%
    运行时长：20.4s
'''
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MNISTDataset(Dataset):
    """自定义MNIST数据集加载器"""
    def __init__(self, file_path, device):
        data = torch.from_numpy(pd.read_csv(file_path, header=None).values)

        # Mnsit有0-9是个标记，由于是二分类任务，所以将标记0的作为1，其余为0
        # 以lable=5为分界线，定义二分类问题，>=5为1，<5为0
        self.labels = torch.where(data[:, 0] < 5, 0, 1).to(device)
        # features
        self.features = data[:, 1:].float().to(device) / 255.0
        # 将bias加到features中，wx+b => wx
        self.features = torch.cat((self.features, torch.ones(self.features.size(0), 1).to(device)), dim=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LogisticRegression:
    """逻辑回归模型"""
    def __init__(self, device='cuda'):
        self.device = device

    def fit(self, train_loader, feature_num = 784, epochs=10, lr=0.01):
        feature_num = feature_num + 1

        w = torch.zeros(feature_num, 1).to(self.device)
        for _ in range(epochs):
            for features, labels in tqdm(train_loader, desc='Training'):
                # 随机梯度上升部分
                # 在“6.1.3 模型参数估计”一章中给出了似然函数，我们需要极大化似然函数
                # 但是似然函数由于有求和项，并不能直接对w求导得出最优w，所以针对似然函数求和
                # 部分中每一项进行单独地求导w，得到针对该样本的梯度，并进行梯度上升（因为是
                # 要求似然函数的极大值，所以是梯度上升，如果是极小值就梯度下降。梯度上升是
                # 加号，下降是减号）
                # 求和式中每一项单独对w求导结果为：xi * yi - (exp(w * xi) * xi) / (1 + exp(w * xi))
                features, labels = features.to(self.device), labels.to(self.device)
                wx = torch.matmul(features, w)
                y = labels
                x = features
                # 梯度上升
                grad = x.t() @ (y.unsqueeze(1) - torch.sigmoid(wx))
                w += lr * grad

        self.w = w


    def evaluate(self, test_loader):
        """
        测试模型
        :param test_loader: 测试数据加载器
        :return: 准确率
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc='Testing'):
                features, labels = features.to(self.device), labels.to(self.device)
                logits = torch.matmul(features, self.w)
                predictions = torch.sigmoid(logits).round()
                correct += (predictions == labels.view(-1, 1)).sum().item()
                total += labels.size(0)

        return correct / total



if __name__ == '__main__':
    start_time = time.time()

    # 参数设置
    batch_size = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取数据
    print('Loading data...')
    train_dataset = MNISTDataset("/root/Desktop/course/datasets/Mnist/mnist_train.csv", device)
    test_dataset = MNISTDataset("/root/Desktop/course/datasets/Mnist/mnist_test.csv", device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    print('Training model...')
    model = LogisticRegression(device)
    model.fit(train_loader, epochs=20)

    # 测试模型
    print('Testing model...')
    accuRate = model.evaluate(test_loader)

    end_time = time.time()
    print(f'Accuracy: {accuRate:.4f}')
    print(f'Time: {end_time - start_time:.4f}s')
