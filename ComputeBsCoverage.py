# 假设一个以基站为中心的小区半径为200m，
# 路径衰落至小区边界处路径衰落值的100倍（或者说是比边界处pathloss多20dB）的范围
# 设为基站信号覆盖区域，超出该区域时基站的影响可以忽略不计，
# 那么基站信号覆盖区域的半径是多少？


import numpy as np
import matplotlib.pyplot as plt

# 定义距离范围，从 0.1 到 1000，取 1000 个点
distances = np.logspace(-1, 3, 1000)

# 计算路径损耗（dB）
path_loss_dB = 38.46 + 20 * np.log10(distances)

# 将 dB 转换为衰落倍数
fading_factors = 10 ** (path_loss_dB / 10)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制图像
plt.plot(distances, 1 / fading_factors, label='Fading Factor')

# 设置坐标轴标签和标题
plt.xlabel('Distance (d)')
plt.ylabel('Fading Factor')
plt.title('Fading Factor vs Distance')

# 设置 x 轴为对数刻度
plt.xscale('log')
plt.yscale('log')

# 显示网格
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# 显示图例
plt.legend()

# 显示图形
plt.show()