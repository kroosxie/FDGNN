import numpy as np
import networkx as nx
import scipy.io as sio
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def save_as_mat(result, filename):
    """ 保存拓扑数据到MAT文件 """
    mat_data = {
        'bs_positions': result['positions_bs'],
        'ue_positions': result['positions_users'],
        'service_loss': result['service_losses'],
        'interference_loss': result['interference_losses'],
        'pl_exponent': np.array([result['pl_exponent']]),
        'cell_size': np.array(result['cell_size']),
        'region_size': np.array([result['region_size']])
    }
    sio.savemat(filename, mat_data)


def convert_to_pyg(result):
    """ 转换为PyG图数据 """
    G = result['graph']

    # 节点特征矩阵
    node_features = []
    node_mapping = {}
    for idx, node in enumerate(G.nodes(data=True)):
        pos = node[1]['pos']
        node_type = 0 if node[1]['type'] == 'BS' else 1
        node_features.append([pos[0], pos[1], node_type])
        node_mapping[node[0]] = idx

    # 边信息
    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        src = node_mapping[u]
        dst = node_mapping[v]
        edge_index.append([src, dst])
        edge_attr.append([
            data['loss'],
            0 if data['type'] == 'service' else 1,
            data['distance']
        ])

    # 转换为Tensor
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor([result['pl_exponent']], dtype=torch.float)
    )


def analyze_topology(result):
    """ 拓扑分析报告 """
    print(f"路径损耗指数: {result['pl_exponent']}")
    print(f"服务信道数量: {len(result['service_losses'])}")
    print(f"干扰信道数量: {len(result['interference_losses'])}")
    print("\n服务信道损耗统计:")
    print(f"均值: {np.mean(result['service_losses']):.2e}")
    print(f"最大值: {np.max(result['service_losses']):.2e}")
    print(f"最小值: {np.min(result['service_losses']):.2e}")
    print("\n干扰信道损耗统计:")
    print(f"均值: {np.mean(result['interference_losses']):.2e}")
    print(f"最大值: {np.max(result['interference_losses']):.2e}")
    print(f"最小值: {np.min(result['interference_losses']):.2e}")


def plot_topology(result):
    """ 可视化拓扑结构 """
    G = result['graph']
    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 绘制小区边界
    h, w = result['cell_size']
    dx, dy = h/2, w/2
    for (x, y) in result['positions_bs']:
        rect = Rectangle((x - dx, y - dy), 2 * dx, 2 * dy,
                         fill=False, edgecolor='pink', linestyle='--', alpha=0.5)
        ax.add_patch(rect)

    # 绘制覆盖范围
    radius = result['interference_radius']
    for (x, y) in result['positions_bs']:
        circle = Circle((x, y), radius,
                        fill=False, edgecolor='lime', alpha=0.3)
        ax.add_patch(circle)

    # 绘制网络元素
    bs_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'BS']
    ue_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'UE']

    nx.draw_networkx_nodes(G, pos, nodelist=bs_nodes, node_size=300,
                           node_color='red', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=ue_nodes, node_size=80,
                           node_color='blue', alpha=0.6)

    # 绘制带透明度的边
    service_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr['type'] == 'service']
    interference_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr['type'] == 'interference']

    nx.draw_networkx_edges(G, pos, edgelist=service_edges, edge_color='green',
                           width=1.5, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=interference_edges, edge_color='gray',
                           style='dotted', alpha=0.4)

    plt.xlim(-0.1, result['region_size'] + 0.1)
    plt.ylim(-0.1, result['region_size'] + 0.1)
    plt.gca().set_aspect('equal')
    plt.title(f"Cellular Network Topology (PL Exponent={result['pl_exponent']})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def calculate_path_loss(distance, pl_exponent):
    """ 计算路径损耗 """
    PL = (distance + 1) ** (-pl_exponent)
    Nt = 4
    # 生成小尺度瑞利衰落
    h_small = (np.random.randn(Nt) + 1j * np.random.randn(Nt)) / np.sqrt(2)

    # 转换为线性衰减因子
    linear_attenuation = 10 ** (-PL / 20)  # 电压幅度的衰减因子

    # 总信道响应
    h_total = linear_attenuation * h_small

    return h_total  # xjc: +1 是防止太大


def generate_topology(N, K, pl_exponent=3.0, interference_radius=0.35, region_size=1.0):
    """
    生成带路径损耗的通信系统拓扑
    参数：
        pl_exponent: 路径损耗指数（默认3.0）
        interference_radius: 干扰半径（覆盖范围）
        region_size: 区域尺寸
    返回：
        包含拓扑数据和损耗信息的字典
    """
    # 生成基站网格布局
    rows = int(np.ceil(np.sqrt(N)))
    cols = rows
    while rows * cols < N:
        cols += 1

    dx = region_size / (cols * 2)
    dy = region_size / (rows * 2)

    # 生成基站坐标
    positions_bs = np.array([
                                (dx * (2 * i + 1), dy * (2 * j + 1))
                                for i in range(cols) for j in range(rows)
                            ][:N])

    # 生成全域随机用户
    positions_users = np.random.uniform(0, region_size, (N * K, 2))

    # 计算用户归属关系
    grid_x = (positions_users[:, 0] // (2 * dx)).astype(int)
    grid_y = (positions_users[:, 1] // (2 * dy)).astype(int)
    grid_x = np.clip(grid_x, 0, cols - 1)
    grid_y = np.clip(grid_y, 0, rows - 1)
    associations = (grid_x * rows + grid_y).clip(max=N - 1)

    # 计算所有基站与用户的距离矩阵
    dist_matrix = np.linalg.norm(
        positions_bs[:, np.newaxis, :] - positions_users[np.newaxis, :, :],
        axis=2
    )

    # 构建网络图
    G = nx.DiGraph()

    # 添加节点
    for bs_idx, pos in enumerate(positions_bs):
        G.add_node(f"BS{bs_idx}", pos=pos, type='BS')
    for user_idx, pos in enumerate(positions_users):
        G.add_node(f"UE{user_idx}", pos=pos, type='UE')

    # 添加服务边及损耗
    service_losses = []
    for ue_idx, bs_idx in enumerate(associations):
        distance = np.linalg.norm(positions_bs[bs_idx] - positions_users[ue_idx])
        loss = calculate_path_loss(distance, pl_exponent)
        G.add_edge(f"BS{bs_idx}", f"UE{ue_idx}",
                   type='service', loss=loss, distance=distance)
        service_losses.append(loss)

    # 添加干扰边及损耗
    interference_edges = []
    interference_losses = []
    for ue_idx in range(len(positions_users)):
        serving_bs = associations[ue_idx]
        valid_bs = np.where(
            (dist_matrix[:, ue_idx] <= interference_radius) &
            (np.arange(len(positions_bs)) != serving_bs)
        )[0]
        for bs_idx in valid_bs:
            distance = dist_matrix[bs_idx, ue_idx]
            loss = calculate_path_loss(distance, pl_exponent)
            G.add_edge(f"BS{bs_idx}", f"UE{ue_idx}",
                       type='interference', loss=loss, distance=distance)
            interference_edges.append((bs_idx, ue_idx))
            interference_losses.append(loss)

    return {
        'positions_bs': positions_bs,
        'positions_users': positions_users,
        'graph': G,
        'service_losses': np.array(service_losses),
        'interference_losses': np.array(interference_losses),
        'pl_exponent': pl_exponent,
        'interference_radius': interference_radius,
        'cell_size': (2 * dx, 2 * dy),
        'region_size': region_size
    }


# 示例用法
if __name__ == "__main__":
    # 生成拓扑
    topology = generate_topology(
        N=4,
        K=6,
        pl_exponent=2.2,
        interference_radius=0.4,
        region_size=1.2
    )

    analyze_topology(topology)
    plot_topology(topology)

    # 保存为MAT文件
    save_as_mat(topology, "cellular_topology.mat")

    # 转换为PyG数据
    pyg_data = convert_to_pyg(topology)

    # 验证数据
    print("\nPyG图数据信息:")
    print(f"节点数量: {pyg_data.num_nodes}")
    print(f"边数量: {pyg_data.num_edges}")
    print(f"节点特征维度: {pyg_data.x.shape}")
    print(f"边特征维度: {pyg_data.edge_attr.shape}")
    print("\n节点特征示例:")
    print(pyg_data.x[:3])  # 显示前3个节点
    print("\n边特征示例:")
    print(pyg_data.edge_attr[:3])  # 显示前3条边

    # 可视化Data：从 Data 对象重建 NetworkX 图（适用于动态生成的 Data）
    edge_index = pyg_data.edge_index.numpy().T  # 转换为边列表
    G_from_pyg = nx.Graph()
    G_from_pyg.add_edges_from(edge_index)

    fig, ax = plt.subplots(figsize=(6, 4))  # 创建 Figure 和 Axes
    nx.draw(G_from_pyg, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')  # 指定 ax 参数
    plt.title("Visualization from PyG Data")
    plt.show()