'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：87.2%
    运行时长：181.2s

从0到1实现最大熵模型我还不会，先从网上找一个实现的代码，太菜了。
'''

import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MaxEntropyModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MaxEntropyModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        return self.log_softmax(self.linear(x))

class MNISTDataset(Dataset):
    def __init__(self, file_path, device):
        data = pd.read_csv(file_path, header=None).values
        self.labels = torch.where(torch.from_numpy(data[:, 0]) < 5, 0, 1).to(device)
        self.features = torch.from_numpy(data[:, 1:]).float().to(device) / 255.0
        # 添加偏置项
        self.features = torch.cat((self.features, torch.ones(len(self.features), 1).to(device)), dim=1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_maxent(model, train_loader, test_loader, device, epochs=10, lr=0.01):
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        def closure():
            optimizer.zero_grad()
            loss = 0.0
            for features, labels in train_loader:
                outputs = model(features)
                loss += criterion(outputs, labels)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
            
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")
    
    return best_acc

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == '__main__':
    start_time = time.time()
    
    # 参数设置
    batch_size = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 784 + 1  # 28x28像素 + 偏置
    num_classes = 10
    epochs = 20
    
    # 读取数据
    print('Loading data...')
    train_dataset = MNISTDataset("/root/Desktop/course/datasets/Mnist/mnist_train.csv", device)
    test_dataset = MNISTDataset("/root/Desktop/course/datasets/Mnist/mnist_test.csv", device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = MaxEntropyModel(input_dim, num_classes).to(device)
    
    # 训练模型
    print('Training...')
    best_acc = train_maxent(model, train_loader, test_loader, device, epochs=epochs)
    
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    print(f"Total time: {time.time()-start_time:.2f} seconds")