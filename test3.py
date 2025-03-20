import torch
import torch_scatter
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, LeakyReLU, BatchNorm1d as BN, Softmax
from torch_geometric.loader import DataLoader
import LayoutsGenerator as LG
from torch_scatter import scatter
import numpy as np


def compute_Sum_SINR_rate(output_list, data):
    # 获取设备
    device = data['UE'].x.device

    # 获取 Nt
    Nt = data['BS', 'service', 'UE'].edge_attr.shape[1] // 2

    # 将每个基站的输出转换为复数波束成形矩阵
    beamforming_matrices = []
    for output in output_list:
        real_part = output[:, :Nt]
        imag_part = output[:, Nt:]
        w_complex = torch.complex(real_part, imag_part)
        beamforming_matrices.append(w_complex)

    # 获取用户到服务基站的关联关系
    service_edge_index = data['BS', 'service', 'UE'].edge_index
    num_users = data['UE'].x.shape[0]
    associations = torch.zeros(num_users, dtype=torch.long, device=device)
    for i in range(service_edge_index.shape[1]):
        bs_idx = service_edge_index[0, i]
        ue_idx = service_edge_index[1, i]
        associations[ue_idx] = bs_idx

    # 为每个基站构建其服务的用户索引列表
    num_bs = data['BS'].x.shape[0]
    bs_user_indices = [[] for _ in range(num_bs)]
    for ue_idx, bs_idx in enumerate(associations):
        bs_user_indices[bs_idx.item()].append(ue_idx)

    # 获取服务信道（每个用户到其服务基站的信道）
    service_channels = data['UE', 'service', 'BS'].original_channel
    sc_real = service_channels[:, :Nt]
    sc_imag = service_channels[:, Nt:]
    service_channels_complex = torch.complex(sc_real, sc_imag)

    # 噪声功率
    sigma2 = torch.tensor(1e-10, dtype=torch.float32, device=device)

    sum_rate = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    for ue_idx in range(num_users):
        # 当前用户的服务基站
        serving_bs = associations[ue_idx].item()
        # 在服务基站的波束成形矩阵中找到该用户的索引
        try:
            user_pos = bs_user_indices[serving_bs].index(ue_idx)
        except ValueError:
            raise ValueError(f"User {ue_idx} not found in BS {serving_bs}'s user list")

        # 服务基站的波束成形矩阵
        w_serving = beamforming_matrices[serving_bs]
        # 当前用户的波束向量
        w_u = w_serving[user_pos]
        # 当前用户到服务基站的信道
        h_serving = service_channels_complex[ue_idx]

        # 计算信号功率
        signal = torch.dot(h_serving.conj(), w_u)
        S = torch.abs(signal) ** 2

        # 同小区干扰（来自同一基站的其他用户）
        intra_interference = torch.tensor(0.0, dtype=torch.float32, device=device)
        for idx, w_other in enumerate(w_serving):
            if idx != user_pos:
                interf = torch.dot(h_serving.conj(), w_other)
                intra_interference += torch.abs(interf) ** 2

        # 跨小区干扰（来自其他基站）
        inter_interference = torch.tensor(0.0, dtype=torch.float32, device=device)
        ue_node = f"UE{ue_idx}"
        # 遍历所有干扰基站的入边
        interf_edge_index = data['BS', 'interf', 'UE'].edge_index
        interf_edge_attr = data['BS', 'interf', 'UE'].original_channel
        for i in range(interf_edge_index.shape[1]):
            bs_idx = interf_edge_index[0, i].item()
            target_ue_idx = interf_edge_index[1, i].item()
            if target_ue_idx == ue_idx:
                # 干扰信道
                h_interf = interf_edge_attr[i]
                h_real = h_interf[:Nt]
                h_imag = h_interf[Nt:]
                h_interf_complex = torch.complex(h_real, h_imag)
                # 干扰基站的波束成形矩阵
                w_i = beamforming_matrices[bs_idx]
                # 累加该基站所有用户的干扰
                for w_other in w_i:
                    interf = torch.dot(h_interf_complex.conj(), w_other)
                    inter_interference += torch.abs(interf) ** 2

        # 总干扰加噪声
        total_interference = intra_interference + inter_interference + sigma2

        # 计算SINR和速率
        if total_interference == 0:
            sinr = torch.tensor(0.0, dtype=torch.float32, device=device)
        else:
            sinr = S / total_interference
        rate = torch.log2(1 + sinr)
        sum_rate = sum_rate + rate
        loss = torch.neg(sum_rate)
    return loss


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), LeakyReLU())  # , BN(channels[i]))  # ReLU改为了LeakyReLU, 加入了BN
        for i in range(1, len(channels))
    ])


class PowerConstraintLayer(nn.Module):
    def __init__(self, Nt_num, P_max_per_antenna_norm):
        super().__init__()
        self.Nt_num = Nt_num
        self.P_max_per_antenna_norm = P_max_per_antenna_norm

    def forward(self, Beamforming_Matrix):
        num_antennas = self.Nt_num
        real_part = Beamforming_Matrix[:, :num_antennas]
        imag_part = Beamforming_Matrix[:, num_antennas:]
        complex_part = real_part + 1j * imag_part

        norms = torch.norm(complex_part, dim=0)  # 计算每列的范数
        mask = norms > self.P_max_per_antenna_norm
        scale_factors = torch.where(mask,
                                    self.P_max_per_antenna_norm / norms,
                                    torch.ones_like(norms))

        scaled_real = real_part * scale_factors
        scaled_imag = imag_part * scale_factors

        return torch.cat([scaled_real, scaled_imag], dim=1)


class HeteroGConv(MessagePassing):  # 边聚合
    """ 参数共享的异构消息传递层 """
    def __init__(self, mlp_m, mlp_u):
        super().__init__(aggr='sum')  # sum 保留 over-the-air 的可能
        # super().__init__(aggr='max')
        self.msg_mlp = mlp_m
        self.update_mlp = mlp_u

    def forward(self, x_dict, edge_index, edge_attr):
        return self.propagate(edge_index, x=x_dict, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)  # x_j为源节点
        agg = self.msg_mlp(tmp)
        return agg

    # def aggregate(self, x_j, edge_index):
    #     row, col = edge_index
    #     aggr_out = scatter(x_j, col, dim=-2, reduce='max')
    #     return aggr_out

    def update(self, aggr_out, x):
        """ 节点更新 """
        dst = x[1]
        tmp = torch.cat([dst, aggr_out], dim=1)
        update = self.update_mlp(tmp)
        return update


class MPNN(nn.Module):
    """ 参数共享 """
    def __init__(self):
        super().__init__()
        self.mlp_m = MLP([2 * 2 * Nt_num, 32, 32])  # 注意和FDNN不一样，因为有边特征的concat
        self.mlp_u = MLP([32 + 2 * Nt_num, 32, 2 * Nt_num])  # 参数共享,参数待调整
        self.hconv_edge = HeteroGConv(self.mlp_m, self.mlp_u)
        self.hconv = HeteroConv({
            ('UE', 'service', 'BS'): self.hconv_edge,  # 反向消息传递
            ('BS', 'service', 'UE'): self.hconv_edge,  # 方向：BS(src)-->UE(dst)
            ('UE', 'interf', 'BS'): self.hconv_edge,  # 双向消息传递
            ('BS', 'interf', 'UE'): self.hconv_edge
        })
        # 自己设计的输出层，等于将以原点为中心的正方形的可行域化为单位圆
        self.bf_output = Seq(Lin(2 * Nt_num, 2 * Nt_num), Tanh(),  # 角度部分，实部虚部均归一化为[-1, 1]
                             PowerConstraintLayer(Nt_num, P_max_per_antenna_norm=1))  # 对各天线的最大发射功率约束为单位1

    def forward(self, data: HeteroData):
        # 初始特征
        x0_dict, edge_attr_dict, edge_index_dict = data.x_dict, data.edge_attr_dict, data.edge_index_dict
        # 消息传递
        x1_dict = self.hconv(x0_dict, edge_index_dict, edge_attr_dict)
        x2_dict = self.hconv(x1_dict, edge_index_dict, edge_attr_dict)  # 边上后续可加注意力
        out_dict = self.hconv(x2_dict, edge_index_dict, edge_attr_dict)
        out_bfv = out_dict['UE']  # 在served_UE节点上输出对应beamforming_vector

        edge_index = edge_index_dict[('BS', 'service', 'UE')]
        out_bfm_list = [[] for _ in range(BS_num_per_Layout)]
        # 遍历 edge_index 中的每一对边
        for i in range(edge_index.shape[1]):
            # 获取源节点索引
            source_node = edge_index[0, i].item()
            # 获取目标节点索引
            target_node = edge_index[1, i].item()
            # 将目标节点的特征添加到对应源节点的列表中
            out_bfm_list[source_node].append(out_bfv[target_node])

        # 将每个源节点对应的目标节点特征列表转换为张量
        out_bfm_list = [torch.stack(item) if item else torch.empty(0, out_bfv.shape[1]) for item in out_bfm_list]
        Beamforming_Matrix_list = [self.bf_output(out_bfm_per_BS) for out_bfm_per_BS in out_bfm_list]  # 对各天线的最大发射功率进行约束
        return Beamforming_Matrix_list


def train():
    model.train()
    total_loss = 0
    for data, topology in zip(train_loader, global_topology_list):
        data = data.to(device)
        optimizer.zero_grad()
        out_list = model(data)
        # loss = compute_Sum_SINR_rate(out_list, topology)
        loss = compute_Sum_SINR_rate(out_list, data)
        loss.backward()
        total_loss += loss.item() * data.num_graphs  # 为什么要乘？因为在sr_rate中有mean，所以可以直接乘
        optimizer.step()
    return total_loss / Layouts_num


# 使用示例
if __name__ == "__main__":
    # 初始化参数
    BS_num_per_Layout = 16  # 取值应满足可开方
    Layouts_num = 16  # 取值应满足可开方
    BS_num = BS_num_per_Layout * Layouts_num  # total BS/cell num, 相当于生成一张大图
    avg_UE_num = 6  # average UE_num per cell
    PathLoss_exponent = 4  # 路径衰落系数（常取3~5）
    Region_size = 120 * (Layouts_num ** 0.5)  # Layout生成区域边长,注意与小区数匹配
    Nt_num = 4  # MISO信道发射天线数

    Batchsize_per_BS = 4  # 取值应可被Layouts_num整除
    graph_embedding_size = 8
    N0 = 1e-10
    num_epochs = 10
    # P_max = 6

    # 生成拓扑结构
    # 后续对比时，subgraph_list也可以这样生成，从每一个中图中拿
    # 也可以不改动，增加美观结果的概率
    global_graph_list = []
    global_topology_list = []
    for layout_idx in range(Layouts_num):
        topology_train = LG.generate_topology(
            N=BS_num_per_Layout,
            K=avg_UE_num,
            pl_exponent=PathLoss_exponent,
            region_size=Region_size,
            Nt=Nt_num
        )
        global_graph = LG.global_convert_to_pyg(topology_train)
        global_graph_list.append(global_graph)
        global_topology_list.append(topology_train)

    train_loader = DataLoader(global_graph_list, batch_size=1, shuffle=False,  # 暂设为1和False
                              num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MPNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 学习率调整

    for epoch in range(num_epochs):
        loss = train()
        print(f'epoch: {epoch}  train_loss: {loss}')
        scheduler.step()  # 动态调整优化器学习率
