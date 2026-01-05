"""
U-Net多目标检测 - 简洁版
处理文件: 20251229_stop_rear_middle_b22_5_h_90_dyn_id_0x723_RX2.json
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
import os
import sys

print("=" * 60)
print("U-Net多目标检测系统")
print("=" * 60)

# ===================== 1. 加载数据 =====================
file_path = r"C:\Users\FakerSmith\Documents\CSDocument\UWB\more target\20251229_stop_rear_middle_b22_5_h_90_dyn_id_0x723_RX2.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"✅ 成功加载文件: {os.path.basename(file_path)}")
except Exception as e:
    print(f"❌ 加载文件失败: {e}")
    print(f"请检查文件路径: {file_path}")
    input("按Enter键退出...")
    sys.exit(1)

# 提取CIR数据
if 'CIR_DATA' in data:
    cir_data = np.array(data['CIR_DATA'], dtype=np.float32)
    print(f"📊 数据维度: {cir_data.shape}")
    print(f"📈 样本数量: {len(cir_data)}")
    print(f"📏 每个样本长度: {len(cir_data[0])}")
else:
    print("❌ 文件中没有找到CIR_DATA字段")
    sys.exit(1)

# ===================== 2. 数据预处理 =====================
print("\n🔧 数据预处理中...")

def preprocess_signal(signal):
    """预处理单个CIR信号"""
    # 1. 去除基线
    baseline = np.mean(signal[-10:])
    signal = signal - baseline
    
    # 2. 归一化
    signal_min, signal_max = signal.min(), signal.max()
    if signal_max - signal_min > 0:
        signal = (signal - signal_min) / (signal_max - signal_min)
    
    # 3. 转换为2D图像 (32x32)
    signal = signal[:1024]  # 取前1024个点
    signal_2d = signal.reshape(32, 32)
    
    return signal_2d

# 处理所有样本
processed_data = []
for i in range(len(cir_data)):
    processed_2d = preprocess_signal(cir_data[i])
    processed_data.append(processed_2d)

processed_data = np.array(processed_data)
print(f"📊 处理后数据维度: {processed_data.shape}")

# ===================== 3. U-Net模型 =====================
class SimpleUNet(nn.Module):
    """简化的U-Net模型"""
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # 编码器
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # 输出层
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        """卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)
        
        e4 = self.enc4(p3)
        
        # 解码器
        d3 = self.up3(e4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # 输出
        output = torch.sigmoid(self.final(d1))
        return output

# ===================== 4. 训练准备 =====================
print("\n🤖 准备训练...")

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")

# 准备训练数据
train_data = processed_data[:800]  # 前800个样本用于训练
test_data = processed_data[800:]   # 剩下的用于测试

# 转换为PyTorch张量
train_tensor = torch.FloatTensor(train_data).unsqueeze(1)  # [800, 1, 32, 32]
test_tensor = torch.FloatTensor(test_data).unsqueeze(1)    # [n, 1, 32, 32]

print(f"📚 训练集大小: {len(train_tensor)}")
print(f"📚 测试集大小: {len(test_tensor)}")

# ===================== 5. 训练函数 =====================
def train_model(model, train_data, epochs=20, lr=0.001):
    """训练模型"""
    model = model.to(device)
    criterion = nn.MSELoss()  # 使用MSE损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    print(f"\n🎯 开始训练，共 {epochs} 轮...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # 打乱数据
        indices = torch.randperm(len(train_data))
        
        for i in range(0, len(train_data), 32):  # 批次大小32
            batch_indices = indices[i:i+32]
            batch = train_data[batch_indices].to(device)
            
            # 前向传播
            output = model(batch)
            
            # 损失计算（自编码器：重建损失）
            loss = criterion(output, batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(train_data) // 32)
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  轮次 {epoch+1:3d}/{epochs} | 损失: {avg_loss:.6f}")
    
    return losses

# ===================== 6. 主训练流程 =====================
print("\n" + "=" * 60)
print("开始训练U-Net模型...")
print("=" * 60)

# 创建模型
model = SimpleUNet()
print(f"📐 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练模型
train_data_tensor = train_tensor.to(device)
losses = train_model(model, train_data_tensor, epochs=30, lr=0.001)

# ===================== 7. 可视化结果 =====================
print("\n📊 可视化结果...")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('训练损失曲线')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.grid(True, alpha=0.3)

# 测试模型
model.eval()
with torch.no_grad():
    # 选择一个测试样本
    test_sample = test_tensor[0:1].to(device)
    output = model(test_sample)
    
    # 转换为numpy用于显示
    input_img = test_sample[0, 0].cpu().numpy()
    output_img = output[0, 0].cpu().numpy()
    
    # 绘制输入和输出
    plt.subplot(2, 2, 2)
    plt.imshow(input_img, cmap='viridis')
    plt.title('输入信号')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(output_img, cmap='viridis')
    plt.title('重建信号')
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    diff = np.abs(input_img - output_img)
    plt.imshow(diff, cmap='hot')
    plt.title('差异图')
    plt.colorbar()

plt.tight_layout()
plt.show()

# ===================== 8. 目标检测 =====================
print("\n🎯 目标检测中...")

def detect_targets(output_map, threshold=0.5):
    """从输出图中检测目标"""
    # 二值化
    binary_map = (output_map > threshold).astype(np.uint8)
    
    # 寻找连通区域
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(binary_map)
    
    targets = []
    for i in range(1, num_features + 1):
        # 获取该区域的位置
        positions = np.argwhere(labeled_array == i)
        if len(positions) > 0:
            # 计算中心位置
            center_y, center_x = positions.mean(axis=0)
            # 计算区域大小
            size = len(positions)
            # 计算置信度
            confidence = output_map[positions[:, 0], positions[:, 1]].mean()
            
            targets.append({
                'id': i,
                'center': (float(center_x), float(center_y)),
                'size': size,
                'confidence': float(confidence)
            })
    
    return targets

# 对测试样本进行目标检测
with torch.no_grad():
    test_batch = test_tensor[:5].to(device)
    outputs = model(test_batch)
    
    print(f"\n检测到 {len(test_batch)} 个样本的目标:")
    for i in range(len(test_batch)):
        output_np = outputs[i, 0].cpu().numpy()
        targets = detect_targets(output_np, threshold=0.3)
        
        print(f"\n样本 {i+1}:")
        print(f"  └─ 检测到 {len(targets)} 个潜在目标")
        for j, target in enumerate(targets):
            print(f"    目标 {j+1}: 位置 ({target['center'][0]:.1f}, {target['center'][1]:.1f}), "
                  f"大小 {target['size']}, 置信度 {target['confidence']:.3f}")

# ===================== 9. 保存模型 =====================
print("\n💾 保存模型中...")
try:
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_shape': (1, 32, 32),
        'losses': losses
    }, 'unet_target_detector.pth')
    print("✅ 模型已保存为: unet_target_detector.pth")
except Exception as e:
    print(f"⚠️  保存模型时出错: {e}")

# ===================== 10. 分析报告 =====================
print("\n" + "=" * 60)
print("分析报告")
print("=" * 60)

# 统计信息
print(f"📊 数据统计:")
print(f"  原始数据点数: {len(cir_data[0])}")
print(f"  处理后的图像大小: 32×32")
print(f"  总样本数: {len(cir_data)}")
print(f"  训练样本数: {len(train_tensor)}")
print(f"  测试样本数: {len(test_tensor)}")

# 模型信息
print(f"\n🤖 模型信息:")
print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"  训练轮次: {len(losses)}")
print(f"  最终损失: {losses[-1]:.6f}")
print(f"  损失下降: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

# 性能评估
if len(test_tensor) > 0:
    with torch.no_grad():
        test_outputs = model(test_tensor.to(device))
        mse = F.mse_loss(test_outputs, test_tensor.to(device)).item()
        print(f"\n📈 性能评估:")
        print(f"  测试集MSE: {mse:.6f}")
        print(f"  测试集PSNR: {10 * np.log10(1.0 / mse):.2f} dB")

print("\n" + "=" * 60)
print("🎉 处理完成！")
print("=" * 60)
print("\n📋 生成的文件:")
print("  ✓ 训练损失图")
print("  ✓ 输入/输出对比图")
print("  ✓ 目标检测结果")
print("  ✓ 模型文件: unet_target_detector.pth")

print("\n🔧 使用方法:")
print("  1. 修改阈值参数可以调整检测灵敏度")
print("  2. 增加训练轮次可以提高重建质量")
print("  3. 调整图像大小可以适应不同长度的数据")

input("\n按Enter键退出程序...")