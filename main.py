import numpy as np

import LayoutsGenerator as LG
from torch_geometric.data import DataLoader


# 初始化参数
BS_num_per_Layout = 16  # 取值应满足可开方
Layouts_num = 16  # 取值应满足可开方
BS_num = BS_num_per_Layout * Layouts_num  # total BS/cell num, 相当于生成一张大图
avg_UE_num = 6  # average UE_num per cell
PathLoss_exponent = 4  # 路径衰落系数（常取3~5）
Region_size = 120 * (Layouts_num ** 0.5)  # Layout生成区域边长,注意与小区数匹配
Nt_num = 4  # MISO信道发射天线数

# 生成拓扑结构
topology = LG.generate_topology(
        N=BS_num,
        K=avg_UE_num,
        pl_exponent=PathLoss_exponent,
        region_size=Region_size,
        Nt=Nt_num
    )

# 提取子图并转化为pyg异构图结构 (已包含数据标准化)
subgraph_list = LG.cell_convert_to_pyg(topology)

# 加载图数据为批次
train_loader = DataLoader(subgraph_list, batch_size=BS_num_per_Layout, shuffle=False, num_workers=0)
# 禁止重排序是为了更好模拟各layout数量（按layout进行规范化的）
#
print("end")
