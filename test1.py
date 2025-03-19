import numpy as np
import networkx as nx
import scipy.io as sio
from torch_geometric.data import HeteroData
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch_geometric.transforms as T


def generate_MISO_channel(distance, pl_exponent, Nt=4):
    """ 生成MISO信道向量（含实部虚部分解） """
    # 计算路径损耗
    path_loss = (distance + 2) ** (-pl_exponent)
    # 生成小尺度瑞利衰落
    h_small = (np.random.randn(Nt) + 1j * np.random.randn(Nt)) / np.sqrt(2)
    # 组合信道响应
    h_total = path_loss * h_small
    # 分离实部虚部：前Nt个为实值，后Nt个为虚部值
    return np.stack([h_total.real, h_total.imag]).flatten()


def plot_pyg_topology(pyg_data):
    """ 可视化PyG数据的拓扑连接关系（力导向布局） """
    G = nx.Graph()

    # 添加所有节点（BS和UE）
    num_bs = pyg_data['BS'].x.shape[0]
    num_ue = pyg_data['UE'].x.shape[0]

    # 创建节点
    G.add_nodes_from([(f"BS{i}", {"type": "BS"}) for i in range(num_bs)])
    G.add_nodes_from([(f"UE{i}", {"type": "UE"}) for i in range(num_ue)])

    # 添加服务边
    service_edges = pyg_data['BS', 'service', 'UE'].edge_index.t().tolist()
    for src, dst in service_edges:
        G.add_edge(f"BS{src}", f"UE{dst}", edge_type='service')

    # 添加干扰边（如果有）
    if ('BS', 'interf', 'UE') in pyg_data.edge_types:
        interf_edges = pyg_data['BS', 'interf', 'UE'].edge_index.t().tolist()
        for src, dst in interf_edges:
            G.add_edge(f"BS{src}", f"UE{dst}", edge_type='interf')

    # 生成布局
    pos = nx.spring_layout(G, seed=42)

    # 绘制图形
    plt.figure(figsize=(12, 8))

    # 绘制节点
    bs_nodes = [n for n in G.nodes if 'BS' in n]
    ue_nodes = [n for n in G.nodes if 'UE' in n]

    nx.draw_networkx_nodes(G, pos, nodelist=bs_nodes, node_size=300,
                           node_color='red', edgecolors='black', label='BS')
    nx.draw_networkx_nodes(G, pos, nodelist=ue_nodes, node_size=80,
                           node_color='blue', alpha=0.6, label='UE')

    # 绘制边
    service_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'service']
    interf_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'interf']

    nx.draw_networkx_edges(G, pos, edgelist=service_edges, edge_color='green',
                           width=1.5, alpha=0.6, label='Service Links')
    nx.draw_networkx_edges(G, pos, edgelist=interf_edges, edge_color='gray',
                           style='dotted', alpha=0.4, label='Interference Links')

    plt.legend()
    plt.title("Network Topology Connection Patterns")
    plt.axis('off')
    plt.show()


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
    service_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr['edge_type'] == 'service']
    interference_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr['edge_type'] == 'interf']

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


def analyze_topology(result):
    """ 拓扑分析报告 """
    print(f"天线数量（Nt）: {result['Nt']}")
    print(f"服务链路数量: {len(result['service_channels'])}")
    print(f"干扰链路数量: {len(result['interference_channels'])}")
    print(f"注：一个链路定义为Nt个负信道的集合，包含2*Nt个值，前Nt个为实部值，后Nt个为虚部值")

    # 信道功率统计
    service_power = [np.mean(np.abs(c) ** 2) for c in result['service_channels']]
    interf_power = [np.mean(np.abs(c) ** 2) for c in result['interference_channels']]

    print("\n服务链路功率统计:")
    print(f"均值: {np.mean(service_power):.2e}")
    print(f"最大值: {np.max(service_power):.2e}")
    print(f"最小值: {np.min(service_power):.2e}")

    print("\n干扰链路功率统计:")
    print(f"均值: {np.mean(interf_power):.2e}")
    print(f"最大值: {np.max(interf_power):.2e}")
    print(f"最小值: {np.min(interf_power):.2e}")


def save_as_mat(result, filename):
    """ 保存拓扑数据到 MAT文件（更新版） """
    mat_data = {
        'bs_positions': result['positions_bs'],
        'ue_positions': result['positions_users'],
        'service_channels': result['service_channels'],
        'interference_channels': result['interference_channels'],
        'pl_exponent': np.array([result['pl_exponent']]),
        'cell_size': np.array(result['cell_size']),
        'region_size': np.array([result['region_size']]),
        'Nt': np.array([result['Nt']])
    }
    sio.savemat(filename, mat_data)


def global_convert_to_pyg(result):
    """ 转换为PyG异构图数据（支持MISO信道） """
    data = HeteroData()
    G = result['graph']
    Nt = result['Nt']

    # 分离BS和UE节点，并创建映射字典
    bs_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'BS']
    ue_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'UE']

    bs_mapping = {node: idx for idx, node in enumerate(bs_nodes)}
    ue_mapping = {node: idx for idx, node in enumerate(ue_nodes)}

    # 添加节点特征 (初始设为全1向量)
    data['BS'].x = torch.ones(len(bs_nodes), Nt, dtype=torch.float)  # 形状 [num_BS, Nt]
    data['UE'].x = torch.ones(len(ue_nodes), Nt, dtype=torch.float)  # 形状 [num_UE, Nt]

    # 处理边信息
    service_edges, service_attrs = [], []
    interf_edges, interf_attrs = [], []

    for u, v, attr in G.edges(data=True):
        if attr['edge_type'] == 'service':
            # 转换节点名为索引
            src_idx = bs_mapping[u]
            dst_idx = ue_mapping[v]
            service_edges.append((src_idx, dst_idx))
            service_attrs.append(attr['channel'])
        elif attr['edge_type'] == 'interf':
            src_idx = bs_mapping[u]
            dst_idx = ue_mapping[v]
            interf_edges.append((src_idx, dst_idx))
            interf_attrs.append(attr['channel'])

    # 全局信道统计量计算 --------------------------------------------------------
    all_service = np.array(service_attrs, dtype=np.float32) if service_attrs else np.empty((0, 2 * Nt))
    all_interf = np.array(interf_attrs, dtype=np.float32) if interf_attrs else np.empty((0, 2 * Nt))
    all_channels = np.concatenate([all_service, all_interf], axis=0)

    # 计算均值和标准差
    mean = np.mean(all_channels, axis=0, keepdims=True)
    std = np.std(all_channels, axis=0, keepdims=True) + 1e-8

    # 添加边到异构图
    if len(service_edges) > 0:
        service_edge_index = torch.tensor(service_edges).t().contiguous()  # 形状 [2, Num_service_edges]
        # 标准化数据作为特征并保留原始数据
        service_normalized = (all_service - mean) / std
        data['BS', 'service', 'UE'].edge_index = service_edge_index
        data['BS', 'service', 'UE'].edge_attr = torch.tensor(service_normalized, dtype=torch.float)
        data['BS', 'service', 'UE'].original_channel = torch.tensor(all_service, dtype=torch.float)

    if len(interf_edges) > 0:
        interf_edge_index = torch.tensor(interf_edges).t().contiguous()
        # 标准化数据作为特征并保留原始数据
        interf_normalized = (all_interf - mean) / std
        data['BS', 'interf', 'UE'].edge_index = interf_edge_index
        data['BS', 'interf', 'UE'].edge_attr = torch.tensor(interf_normalized, dtype=torch.float)
        data['BS', 'interf', 'UE'].original_channel = torch.tensor(all_interf, dtype=torch.float)

    return data


def cell_convert_to_pyg(topology):
    """ 将拓扑数据转换为基站级别的子图列表 （含特征规范化）"""
    G = topology['graph']
    subgraphs = []

    # 全局信道统计量计算（用于数据规范化） --------------------------------------------------------
    # 合并所有服务信道和干扰信道数据
    all_service = topology['service_channels']  # shape [num_service, 2*Nt]
    all_interf = topology['interference_channels']  # shape [num_interf, 2*Nt]
    all_channels = np.concatenate([all_service, all_interf], axis=0)

    # 计算全局均值和标准差（每个特征维度单独计算）
    mean = np.mean(all_channels, axis=0, keepdims=True)  # shape [1, 2*Nt]
    std = np.std(all_channels, axis=0, keepdims=True) + 1e-8  # 防止除零

    # 获取所有基站节点
    bs_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'BS']

    for bs in bs_nodes:
        # 创建异构图数据
        hetero_data = HeteroData()

        # 获取当前基站的服务用户和干扰用户
        service_ues = [v for _, v, d in G.out_edges(bs, data=True) if d['edge_type'] == 'service']
        interf_ues = [v for _, v, d in G.out_edges(bs, data=True) if d['edge_type'] == 'interf']

        # 收集节点特征（真实信道数据）
        served_feats = [G[bs][ue]['channel'] for ue in service_ues]
        interf_feats = [G[bs][ue]['channel'] for ue in interf_ues]

        # 规范化处理 ----------------------------------------------------------
        # 服务用户节点特征处理
        if len(served_feats) > 0:
            served_feats_np = np.array(served_feats)
            served_normalized = (served_feats_np - mean) / std
            hetero_data['served'].x = torch.tensor(served_normalized, dtype=torch.float)
            hetero_data['served'].original_channel = torch.tensor(served_feats_np)
        else:
            # 处理空节点情况，保持特征维度一致
            hetero_data['served'].x = torch.empty((0, all_channels.shape[1]), dtype=torch.float)
            hetero_data['served'].original_channel = torch.empty((0, all_channels.shape[1]))

        # 干扰用户节点特征处理
        if len(interf_feats) > 0:
            interf_feats_np = np.array(interf_feats)
            interf_normalized = (interf_feats_np - mean) / std
            hetero_data['interfered'].x = torch.tensor(interf_normalized, dtype=torch.float)
            hetero_data['interfered'].original_channel = torch.tensor(interf_feats_np)
        else:
            hetero_data['interfered'].x = torch.empty((0, all_channels.shape[1]), dtype=torch.float)
            hetero_data['interfered'].original_channel = torch.empty((0, all_channels.shape[1]))

        # 构建全连接边 （不同类节点间）
        if len(service_ues) > 0 and len(interf_ues) > 0:
            src = torch.arange(len(service_ues)).repeat_interleave(len(interf_ues))
            dst = torch.arange(len(interf_ues)).repeat(len(service_ues))
            hetero_data['served', 'conn', 'interfered'].edge_index = torch.stack([src, dst], dim=0)
            hetero_data['interfered', 'conn', 'served'].edge_index = torch.stack([dst, src], dim=0)  # 增加反向边，变为无向图（其实是一个二分图）
        else:
            print(f"基站 {bs} 缺少有效连接关系，跳过该子图")
            # raise RuntimeError(f"基站 {bs} 缺少有效连接关系，错误")
            continue  # 跳过该子图

        # hetero_data = T.ToUndirected()(hetero_data)  # 直接转为无向图，与增加反向边是一样的
        subgraphs.append(hetero_data)

    return subgraphs


def generate_topology(N, K, pl_exponent=3.0, region_size=1.0, Nt=4, max_attempts=100):
    """
    生成MISO系统拓扑
    新增参数：
        Nt: 基站天线数量
    """
    # 基站布局计算保持不变
    rows = int(np.ceil(np.sqrt(N)))
    cols = rows
    while rows * cols < N:
        cols += 1

    dx = region_size / (cols * 2)
    dy = region_size / (rows * 2)
    interference_radius = dx / 1.5 * 10 ** (2 / pl_exponent)

    for attempt in range(max_attempts):
        # 生成基站和用户坐标
        positions_bs = np.array([(dx * (2 * i + 1), dy * (2 * j + 1)) for i in range(cols) for j in range(rows)][:N])
        positions_users = np.random.uniform(0, region_size, (N * K, 2))  # 生成全域随机用户

        # 生成用户坐标（高斯分布）
        # positions_users = np.zeros((N * K, 2))
        # sigma = 0.1 * region_size  # 高斯分布标准差
        # for bs_idx in range(N):
        #     # 每个基站生成K个用户
        #     start_idx = bs_idx * K
        #     end_idx = start_idx + K
        #     # 以基站位置为中心生成高斯分布坐标
        #     positions_users[start_idx:end_idx] = np.random.normal(
        #         loc=positions_bs[bs_idx],
        #         scale=sigma,
        #         size=(K, 2)
        #     )
        # # 限制用户坐标在区域范围内
        # positions_users = np.clip(positions_users, 0, region_size)

        # 用户归属计算保持不变
        grid_x = (positions_users[:, 0] // (2 * dx)).astype(int).clip(0, cols - 1)
        grid_y = (positions_users[:, 1] // (2 * dy)).astype(int).clip(0, rows - 1)
        associations = (grid_x * rows + grid_y).clip(max=N - 1)

        # 距离矩阵计算
        dist_matrix = np.linalg.norm(positions_bs[:, np.newaxis, :] - positions_users[np.newaxis, :, :], axis=2)

        # 条件1: 检查每个基站都有服务用户
        service_counts = np.bincount(associations, minlength=N)
        if np.any(service_counts == 0):
            continue  # 重新生成

        # 构建网络图
        G = nx.DiGraph()
        [G.add_node(f"BS{bs_idx}", pos=pos, type='BS') for bs_idx, pos in enumerate(positions_bs)]
        [G.add_node(f"UE{ue_idx}", pos=pos, type='UE') for ue_idx, pos in enumerate(positions_users)]

        # 服务信道生成
        service_channels = []
        for ue_idx, bs_idx in enumerate(associations):
            distance = np.linalg.norm(positions_bs[bs_idx] - positions_users[ue_idx])
            channel = generate_MISO_channel(distance, pl_exponent, Nt)
            G.add_edge(f"BS{bs_idx}", f"UE{ue_idx}",
                       edge_type='service',
                       distance=distance,
                       channel=channel)
            service_channels.append(channel)

        # 干扰信道生成
        interference_channels = []
        interf_counts = np.zeros(N, dtype=int)
        for ue_idx in range(len(positions_users)):
            serving_bs = associations[ue_idx]
            for bs_idx in \
            np.where((dist_matrix[:, ue_idx] <= interference_radius) & (np.arange(len(positions_bs)) != serving_bs))[0]:
                distance = dist_matrix[bs_idx, ue_idx]
                channel = generate_MISO_channel(distance, pl_exponent, Nt)
                G.add_edge(f"BS{bs_idx}", f"UE{ue_idx}",
                           edge_type='interf',
                           distance=distance,
                           channel=channel)
                interference_channels.append(channel)
                interf_counts[bs_idx] += 1  # 记录干扰次数

        # 条件检查2：每个基站至少一个干扰用户
        if np.any(interf_counts == 0):
            continue  # 重新生成

        return {
            'positions_bs': positions_bs,
            'positions_users': positions_users,
            'graph': G,
            'service_channels': np.array(service_channels),
            'interference_channels': np.array(interference_channels),
            'pl_exponent': pl_exponent,
            'Nt': Nt,
            'interference_radius': interference_radius,
            'cell_size': (2 * dx, 2 * dy),
            'region_size': region_size
        }
    # 超过最大尝试次数仍未满足条件
    raise RuntimeError(
        f"无法在{max_attempts}次尝试内生成有效拓扑。建议：\n"
        f"1. 增大K值（当前K={K}）\n"
        f"2. 扩大区域尺寸（当前region_size={region_size}）\n"
        f"3. 减小路径损耗指数（当前pl_exponent={pl_exponent}）"
    )


if __name__ == "__main__":
    # 生成拓扑
    topology = generate_topology(
        N=16,
        K=6,
        pl_exponent=4,
        region_size=120,
        Nt=4
    )

    analyze_topology(topology)
    plot_topology(topology)

    # 保存数据
    save_as_mat(topology, "miso_topology.mat")

    # 转换为PyG数据
    pyg_data = global_convert_to_pyg(topology)

    # 转换为子图列表
    cell_graphs = cell_convert_to_pyg(topology)
    print(f"\n生成基站子图数量: {len(cell_graphs)}")
    for i, g in enumerate(cell_graphs):
        print(f"子图{i}: 服务用户数={g['served'].x.shape[0] if 'served' in g.node_types else 0}, "
              f"干扰用户数={g['interfered'].x.shape[0] if 'interfered' in g.node_types else 0}")

    # 可视化PyG数据 (将PyG数据转换回networkx，再次检查拓扑关系)
    plot_pyg_topology(pyg_data)

    # 验证PyG全局图数据维度
    print("\n验证PyG数据维度（全局）")

    # 验证节点特征维度
    print("BS节点特征维度:", pyg_data['BS'].x.shape)  # 应输出 torch.Size([16, 4])
    print("UE节点特征维度:", pyg_data['UE'].x.shape)  # 应输出 torch.Size([96, 4])

    # 验证边数据
    service_edge_index = pyg_data['BS', 'service', 'UE'].edge_index
    service_edge_attr = pyg_data['BS', 'service', 'UE'].edge_attr
    interf_edge_index = pyg_data['BS', 'interf', 'UE'].edge_index
    interf_edge_attr = pyg_data['BS', 'interf', 'UE'].edge_attr

    print("\n服务边索引维度:", service_edge_index.shape)  #  torch.Size([2, num_Hs])
    print("服务边属性维度:", service_edge_attr.shape)  #  torch.Size([num_Hs, num_Hs_real + num_Hs_imag])
    print("\n干扰边索引维度:", interf_edge_index.shape)  #  torch.Size([2, num_Hi])
    print("干扰边属性维度:", interf_edge_attr.shape)  #  torch.Size([num_Hi, num_Hi_real + num_Hi_imag])

    # 验证信道特征数值
    print("\n服务边属性示例（前3条）:")
    print(service_edge_attr[:3])
    print("干扰边属性示例（前3条）:")
    print(interf_edge_attr[:3])

    # 验证边连接有效性
    print("\n服务边源节点范围:", service_edge_index[0].min().item(), "-",
          service_edge_index[0].max().item())  # 应全在 0-num_BS 之间
    print("服务边目标节点范围:", service_edge_index[1].min().item(), "-",
          service_edge_index[1].max().item())  # 应全在 0-num_UE 之间
    print("干扰边源节点范围:", interf_edge_index[0].min().item(), "-", interf_edge_index[0].max().item())  # 应全在 0-Num_BS 之间
    print("干扰边目标节点范围:", interf_edge_index[1].min().item(), "-",
          interf_edge_index[1].max().item())  # 应全在 0-num_UE*4 之间
