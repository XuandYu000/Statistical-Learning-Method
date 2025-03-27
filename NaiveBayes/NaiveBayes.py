'''
数据集: MNIST
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：84.3%
    运行时长：19.5s
'''

import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MNISTDataset(Dataset):
    """自定义MNIST数据集加载器"""
    def __init__(self, file_path, device='cuda'):
        data = torch.from_numpy(pd.read_csv(file_path, header=None).values)
        self.labels = data[:, 0].to(device, dtype=torch.long)
        # 二值化处理并转移到GPU
        self.features = (data[:, 1:].to(device, dtype=torch.float32) > 128).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NaiveBayesClassifier:
    """优化后的朴素贝叶斯分类器"""
    def __init__(self, n_features=784, n_classes=10, device='cuda'):
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = device
        self.Py = None
        self.Px_y = None

    def fit(self, train_loader):
        """计算先验概率和条件概率"""
        # 初始化计数
        label_counts = torch.zeros(self.n_classes, device=self.device)
        feature_counts = torch.zeros((self.n_classes, self.n_features, 2),
                                   device=self.device)

        # 统计计数
        for features, labels in tqdm(train_loader, desc="Training"):
            for c in range(self.n_classes):
                mask = (labels == c)
                if mask.any():
                    # 统计每个特征出现0/1的次数
                    class_features = features[mask]
                    feature_counts[c, :, 0] += (class_features == 0).sum(dim=0)
                    feature_counts[c, :, 1] += (class_features == 1).sum(dim=0)
                    label_counts[c] += mask.sum()

        # 计算先验概率（拉普拉斯平滑）
        self.Py = torch.log((label_counts + 1) / (len(train_loader.dataset) + self.n_classes))

        # 计算条件概率（拉普拉斯平滑）
        self.Px_y = torch.zeros_like(feature_counts)
        for c in range(self.n_classes):
            total = label_counts[c] + 2  # 平滑分母
            self.Px_y[c, :, 0] = torch.log((feature_counts[c, :, 0] + 1) / total)
            self.Px_y[c, :, 1] = torch.log((feature_counts[c, :, 1] + 1) / total)

    def predict_batch(self, features):
        """批量预测"""
        batch_size = features.shape[0]
        # 扩展维度以便广播
        log_probs = torch.zeros((batch_size, self.n_classes), device=self.device)

        # 向量化计算每个类别的对数概率
        for c in range(self.n_classes):
            # 收集所有特征的条件概率
            probs = self.Px_y[c, torch.arange(self.n_features), features.long()].sum(dim=1)
            log_probs[:, c] = probs + self.Py[c]

        return log_probs.argmax(dim=1)

    def evaluate(self, test_loader):
        """评估模型准确率"""
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Testing"):
                preds = self.predict_batch(features)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

if __name__ == '__main__':
    start_time = time.time()

    # 参数设置
    batch_size = 1024  # 批处理大小
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    print("Loading data...")
    train_dataset = MNISTDataset("/root/Desktop/course/datasets/Mnist/mnist_train.csv", device)
    test_dataset = MNISTDataset("/root/Desktop/course/datasets/Mnist/mnist_test.csv", device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 训练模型
    print("Training model...")
    model = NaiveBayesClassifier(device=device)
    model.fit(train_loader)

    # 测试模型
    print("Testing model...")
    accuracy = model.evaluate(test_loader)

    # 输出结果
    end_time = time.time()
    print(f"\n正确率: {accuracy:.4f}")
    print(f"运行时长: {end_time - start_time:.2f}秒")