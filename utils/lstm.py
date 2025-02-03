import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 自定义数据集类
class YieldDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 每个Band的数据是一个长度为32的序列
        features = torch.tensor([row[f'Band_{i}'] for i in range(1, 8)], dtype=torch.float32)
        features = features.permute(1, 0)  # 变换形状为 (32, 7) 时间步为第一维度
        unit = row['单位']
        yield_val = row['产量']
        # 统一产量单位到吨
        if unit == '吨':
            yield_val /= 100000
        yield_val = torch.tensor(yield_val, dtype=torch.float32)
        return features, yield_val

# 定义模型
class YieldPredictionModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2):
        super(YieldPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM 层
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 获取最后一个时间步的输出

        # MLP 层
        x = self.relu(self.fc1(lstm_out))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        return output.squeeze()

# 数据预处理
data_path = 'feature-1.xlsx'
df = pd.read_excel(data_path)

# 检查数据框是否有NaN值
print(df.isna().sum())
df = df.dropna()

# 对每个Band的时间序列数据进行标准化
# 计算每个Band的均值和标准差
scalers = {}
for band in range(1, 8):
    band_values = np.concatenate(df[f'Band_{band}'].apply(eval).values)
    scalers[f'Band_{band}'] = StandardScaler().fit(band_values.reshape(-1, 1))

# 应用标准化
for band in range(1, 8):
    df[f'Band_{band}'] = df[f'Band_{band}'].apply(
        lambda x: scalers[f'Band_{band}'].transform(np.array(eval(x)).reshape(-1, 1)).flatten())

# 将数据按照年份划分
train_data = df[df['Year'] < 2019]
val_data = df[(df['Year'] >= 2019) & (df['Year'] <= 2021)]

train_dataset = YieldDataset(train_data)
val_dataset = YieldDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 初始化模型、损失函数和优化器
model = YieldPredictionModel(input_dim=7)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 调整为更低的学习率

# 权重初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0.01)

model.apply(init_weights)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for features, target in train_loader:
        output = model(features)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()

        # 添加梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    # 每10轮进行一次验证
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                output = model(features)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    for features, target in val_loader:
        output = model(features)
        print(f'Predicted: {output}, Actual: {target}')
