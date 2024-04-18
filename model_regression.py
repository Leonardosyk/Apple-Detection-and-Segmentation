import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 使用一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # 使用最大池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 使用自适应平均池化层
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 定义全连接层
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        # 输出层调整为单个值，适合单变量回归任务
        self.fc2 = nn.Linear(128, 1)

        # 定义dropout层
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 应用卷积和池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # 应用自适应平均池化层
        x = self.adapt_pool(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 应用全连接层和dropout层
        x = self.dropout(F.relu(self.fc1(x)))
        # 移除激活函数
        x = self.fc2(x)

        return x

# 创建模型实例，用于回归任务
model = SimpleCNN()
