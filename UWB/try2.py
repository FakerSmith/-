import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RadarDataProcessor:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.cir_data = np.array(self.data["CIR_DATA"])
        self.config = self.data["RADAR_CONFIGURATION"]
        
    def get_cir_matrix(self):
        """返回CIR数据的numpy数组"""
        return self.cir_data
    
    def get_config_summary(self):
        """获取配置摘要"""
        summary = {
            'channel_freq': self.config["UWB_CHANNEL_FREQUENCY"],
            'burst_interval': self.config["BURST_INTERVAL"],
            'tx_power': self.config["TX_POWER_NOMINAL"],
            'cir_taps': self.config["CIR_TAPS"],
            'cir_offset': self.config["CIR_OFFSET"]
        }
        return summary
    
    def plot_cir_heatmap(self):
        """绘制CIR数据的热图"""
        plt.figure(figsize=(12, 6))
        plt.imshow(self.cir_data.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Sample Index')
        plt.ylabel('Tap Index')
        plt.title('CIR Data Heatmap')
        plt.show()
    
    def analyze_statistics(self):
        """分析统计数据"""
        df = pd.DataFrame(self.cir_data)
        stats = {
            'mean': df.mean().values,
            'std': df.std().values,
            'min': df.min().values,
            'max': df.max().values
        }
        return stats

# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = RadarDataProcessor('20251229_stop_rear_middle_b22_5_h_90_stat+adult_id_0x723_RX2.json')
    
    # 获取配置信息
    config = processor.get_config_summary()
    print("雷达配置:", config)
    
    # 获取CIR数据矩阵
    cir_matrix = processor.get_cir_matrix()
    print(f"CIR数据形状: {cir_matrix.shape}")
    
    # 绘制热图
    processor.plot_cir_heatmap()
    
    # 分析统计数据
    stats = processor.analyze_statistics()
    print("均值:", stats['mean'][:10])  # 显示前10个tap的均值