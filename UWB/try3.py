# 安装必要的库（如果没有的话）
# pip install torch numpy matplotlib scikit-learn
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

def load_cir_data(json_path):
    """从JSON文件加载CIR数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取CIR数据
    cir_data = np.array(data['CIR_DATA'])
    
    print(f"加载了 {len(cir_data)} 个CIR样本")
    print(f"每个样本有 {cir_data.shape[1]} 个采样点")
    
    return cir_data


def preprocess_cir_data(cir_data):
    """预处理CIR数据"""
    processed = np.zeros_like(cir_data, dtype=np.float32)
    
    for i in range(len(cir_data)):
        sample = cir_data[i].astype(np.float32)
        # 归一化到[-1, 1]
        max_val = np.max(np.abs(sample))
        if max_val > 0:
            processed[i] = sample / max_val
    
    return processed

def create_simulated_cir_data(n_samples=1000, n_points=32):
    """创建模拟CIR数据（如果真实数据不可用）"""
    np.random.seed(42)
    
    # 创建两类数据
    data = []
    labels = []
    
    for i in range(n_samples):
        if i % 2 == 0:
            # 第一类：早期脉冲
            signal = np.zeros(n_points)
            peak_pos = np.random.randint(5, 10)
            signal[peak_pos] = np.random.rand() * 2 - 1
            labels.append(0)
        else:
            # 第二类：晚期脉冲
            signal = np.zeros(n_points)
            peak_pos = np.random.randint(20, 27)
            signal[peak_pos] = np.random.rand() * 2 - 1
            labels.append(1)
        
        # 添加噪声
        signal += np.random.randn(n_points) * 0.1
        data.append(signal)
    
    return np.array(data), np.array(labels)

class CIRDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.FloatTensor(data).unsqueeze(1)  # 添加通道维度
        self.labels = None if labels is None else torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

class SimpleCIRCNN(nn.Module):
    def __init__(self, input_length=32, num_classes=2):
        super(SimpleCIRCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        
        # 全连接层
        # 注意：这里根据input_length计算全连接层输入大小
        # 经过3次pooling：input_length -> input_length/2 -> input_length/4 -> input_length/8
        fc_input_size = 64 * (input_length // 8)
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout和批标准化
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        # 输入x形状: [batch_size, 1, 32]
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 32 -> 16
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 16 -> 8
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 8 -> 4
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def visualize_cir_samples(cir_data, labels=None, n_samples=6):
    """可视化CIR样本"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    indices = np.random.choice(len(cir_data), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        if i < len(axes):
            ax = axes[i]
            sample = cir_data[idx]
            ax.plot(sample, linewidth=2)
            
            title = f'样本 {idx}'
            if labels is not None and idx < len(labels):
                title += f' (类别: {labels[idx]})'
            
            ax.set_title(title)
            ax.set_xlabel('采样点')
            ax.set_ylabel('幅值')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, test_loader, device, epochs=20):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1:3d}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Acc: {test_acc:.2f}%')
    
    return train_losses, test_losses, test_accuracies

def evaluate_model(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def plot_training_history(train_losses, test_losses, test_accuracies):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(train_losses, label='training loss', linewidth=2)
    axes[0].plot(test_losses, label='test loss', linewidth=2)
    axes[0].set_xlabel('training cycle')
    axes[0].set_ylabel('loss')
    axes[0].set_title('training and test loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(test_accuracies, linewidth=2, color='green')
    axes[1].set_xlabel('training cycle')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('test Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, sample_input):
    """可视化卷积特征图"""
    model.eval()
    
    # 注册hook来获取中间特征
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())
    
    # 注册hook到卷积层
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv1d):
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # 前向传播
    with torch.no_grad():
        _ = model(sample_input)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 可视化特征图
    if len(features) > 0:
        fig, axes = plt.subplots(len(features), 1, figsize=(12, 3*len(features)))
        
        if len(features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i]
            # 取第一个样本的第一个通道
            feature_map = feature[0, 0, :].numpy()
            ax.plot(feature_map, linewidth=2)
            ax.set_title(f'卷积层 {i+1} 特征图')
            ax.set_xlabel('采样点')
            ax.set_ylabel('激活值')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    print("=== CIR数据CNN处理演示 ===")
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    json_path = r"C:\Users\FakerSmith\Documents\CSDocument\UWB\data1.json"  # 修改为你的文件路径
    
    try:
        print("尝试加载JSON数据...")
        cir_data = load_cir_data(json_path)
        
        # 创建简单标签（根据数据特征）
        # 这里使用简单的阈值分类，实际应用需要真实标签
        labels = np.zeros(len(cir_data), dtype=np.int64)
        for i in range(len(cir_data)):
            sample = cir_data[i]
            if np.max(np.abs(sample[:len(sample)//2])) > np.max(np.abs(sample[len(sample)//2:])):
                labels[i] = 0
            else:
                labels[i] = 1
        
    except FileNotFoundError:
        print(f"文件 {json_path} 未找到，使用模拟数据...")
        cir_data, labels = create_simulated_cir_data()
    except Exception as e:
        print(f"加载数据时出错: {e}，使用模拟数据...")
        cir_data, labels = create_simulated_cir_data()
    
    print(f"数据形状: {cir_data.shape}")
    print(f"标签分布: {np.bincount(labels)}")
    
    # 2. 预处理数据
    print("\n预处理数据...")
    processed_data = preprocess_cir_data(cir_data)
    
    # 3. 可视化样本
    print("\n可视化CIR样本...")
    visualize_cir_samples(processed_data[:100], labels[:100])
    
    # 4. 分割数据集
    print("\n分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, labels, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {len(X_train)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")
    
    # 5. 创建数据加载器
    train_dataset = CIRDataset(X_train, y_train)
    test_dataset = CIRDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6. 创建模型
    print("\n创建CNN模型...")
    input_length = cir_data.shape[1]  # 获取实际数据长度
    model = SimpleCIRCNN(input_length=input_length).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. 训练模型
    print("\n开始训练...")
    train_losses, test_losses, test_accuracies = train_model(
        model, train_loader, test_loader, device, epochs=15
    )
    
    # 8. 可视化训练过程
    print("\n可视化训练结果...")
    plot_training_history(train_losses, test_losses, test_accuracies)
    
    # 9. 最终评估
    criterion = nn.CrossEntropyLoss()
    final_loss, final_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"\n最终结果 - 测试损失: {final_loss:.4f}, 测试准确率: {final_acc:.2f}%")
    
    # 10. 保存模型
    try:
        model_path = "cir_cnn_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'input_length': input_length
        }, model_path)
        print(f"\n模型已保存到: {model_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # 11. 进行预测
    print("\n进行预测...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, tuple):
                batch_data = batch_data[0]
            
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    predictions = np.array(predictions)
    print(f"预测结果: {predictions[:10]}...")
    print(f"真实标签: {y_test[:10]}")
    
    # 12. 可视化特征图
    print("\n可视化卷积特征...")
    if len(test_dataset) > 0:
        sample_input = test_dataset[0][0].unsqueeze(0).to(device)
        visualize_feature_maps(model, sample_input)
    
    print("\n=== 处理完成 ===")

def run_simple_example():
    """最简化的运行示例"""
    print("=== 最简化CNN演示 ===")
    
    # 1. 创建数据
    X, y = create_simulated_cir_data(n_samples=200, n_points=32)
    X = preprocess_cir_data(X)
    
    # 2. 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 4. 创建简单模型
    class TinyCIRCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv1d(8, 16, 3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(16 * 8, 2)  # 16通道 * 8个时间点
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
            x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x
    
    # 5. 训练
    device = torch.device('cpu')
    model = TinyCIRCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("开始训练...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean().item()
        
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={accuracy*100:.1f}%")
    
    print("演示完成！")

# 运行示例
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n尝试运行简化版本...")
        run_simple_example()