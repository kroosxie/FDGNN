import torch
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, LeakyReLU, BatchNorm1d as BN, Softmax
from torch_geometric.loader import DataLoader
import LayoutsGenerator as LG
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import savemat
import utils

def compute_Sum_SINR_rate_d(output_list, data):
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
    # service_channels = data['BS', 'service', 'UE'].original_channel  # 按bs_idx顺序排列，有误，应按ue_idx排列
    service_channels = data['UE', 'service', 'BS'].original_channel  # 按bs_idx顺序排列，有误，应按ue_idx排列
    sc_real = service_channels[:, :Nt]
    sc_imag = service_channels[:, Nt:]
    service_channels_complex = torch.complex(sc_real, sc_imag)

    # 噪声功率
    sigma2 = torch.tensor(N0, dtype=torch.float32, device=device)

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
        # 遍历所有干扰基站的入边 （效率可能有点低）
        interf_edge_index = data['BS', 'interf', 'UE'].edge_index
        interf_edge_attr = data['BS', 'interf', 'UE'].original_channel
        for i in range(interf_edge_index.shape[1]):  # 遍历所有干扰边
            bs_idx = interf_edge_index[0, i].item()
            target_ue_idx = interf_edge_index[1, i].item()
            if target_ue_idx == ue_idx:
                # 干扰信道
                h_interf = interf_edge_attr[i]  # 第i条边，非直接取所有的干扰边，正确
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


def compute_Sum_SINR_rate_t(output_list, topology):
    # 将每个基站的输出转换为复数波束成形矩阵
    Nt = topology['Nt']
    beamforming_matrices = []
    for output in output_list:
        real_part = output[:, :Nt]
        imag_part = output[:, Nt:]
        w_complex = torch.complex(real_part, imag_part)
        beamforming_matrices.append(w_complex)

    # 获取用户到服务基站的关联关系
    graph = topology['graph']
    num_users = len(topology['positions_users'])
    associations = torch.tensor(topology['associations'], dtype=torch.long, device=device)

    # 为每个基站构建其服务的用户索引列表
    num_bs = len(topology['positions_bs'])
    bs_user_indices = [[] for _ in range(num_bs)]
    for ue_idx, bs_idx in enumerate(associations):
        bs_user_indices[bs_idx.item()].append(ue_idx)

    # 获取服务信道（每个用户到其服务基站的信道）
    service_channels = torch.tensor(topology['service_channels'], dtype=torch.float32, device=device)  # (num_users, 2 * Nt)， 按用户idx顺序排列
    sc_real = service_channels[:, :Nt]
    sc_imag = service_channels[:, Nt:]
    service_channels_complex = torch.complex(sc_real, sc_imag)

    # 噪声功率
    sigma2 = torch.tensor(N0, dtype=torch.float32, device=device)

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
        for pred in graph.predecessors(ue_node):
            edge_data = graph.get_edge_data(pred, ue_node)
            if edge_data.get('edge_type') == 'interf':
                interf_bs = int(pred[2:])  # 干扰基站索引
                # 干扰信道
                h_interf = torch.tensor(edge_data['channel'], dtype=torch.float32, device=device)
                h_real = h_interf[:Nt]
                h_imag = h_interf[Nt:]
                h_interf_complex = torch.complex(h_real, h_imag)
                # 干扰基站的波束成形矩阵
                w_i = beamforming_matrices[interf_bs]
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

    return sum_rate


def compute_Sum_SLNR_rate(output, data):
    slnr_value = compute_SLNR(output, data)  # 能一一对应吗？
    slnr_rate = torch.log2(1 + slnr_value)
    sum_slnr_rate = torch.sum(slnr_rate)
    loss = torch.neg(sum_slnr_rate)
    return loss


def compute_SLNR(output, data) -> torch.Tensor:
    # Nt = direct_h.size(1) // 2  # 获取天线数Nt
    Nt = Nt_num
    direct_h = data['served'].original_channel
    interf_h = data['interfered'].original_channel

    # 将实虚部分转换为复数tensor
    def to_complex(t: torch.Tensor):
        real = t[..., :Nt]
        imag = t[..., Nt:]
        return real + 1j * imag

    output_c = to_complex(output)
    direct_h_c = to_complex(direct_h)
    interf_h_c = to_complex(interf_h)

    results = []
    num_served_users = output_c.size(0)

    for served_idx in range(num_served_users):
        # 计算分子：|h^H u|²
        numerator = torch.abs(torch.sum(torch.conj(direct_h_c[served_idx]) * output_c[served_idx])) ** 2

        # Noise
        denominator = N0

        # 组内干扰
        mask = torch.arange(direct_h_c.size(0)) != served_idx
        for inter_interf_user in direct_h_c[mask]:
            term_inter = torch.abs(torch.sum(torch.conj(inter_interf_user) * output_c[served_idx])) ** 2
            denominator += term_inter

        # 组间干扰 Σ|interf_h^H u|²
        for interf_user in interf_h_c:
            term_intra = torch.abs(torch.sum(torch.conj(interf_user) * output_c[served_idx])) ** 2  # 共轭转置
            denominator += term_intra

        # 计算(numerator / denominator)
        slnr = numerator / denominator
        results.append(slnr)

    return torch.stack(results).unsqueeze(1)  # 转为[服务用户数, 1]维度


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), LeakyReLU())#, BN(channels[i]))  # ReLU改为了LeakyReLU, 加入了BN
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
        super().__init__(aggr='sum')
        # super().__init__(aggr='max')
        self.msg_mlp = mlp_m
        self.update_mlp = mlp_u

    def forward(self, x_dict, edge_index):
        return self.propagate(edge_index, x=x_dict)

    def message(self, x_i, x_j):  # x_j是各边的源节点（src）
        """ 消息生成 """
        msg = self.msg_mlp(x_j)
        return msg

    def update(self, aggr_out, x):
        """ 节点更新 """
        dst = x[1]
        tmp = torch.cat([dst, aggr_out], dim=1)
        update = self.update_mlp(tmp)
        return update


class HeteroEConv(MessagePassing):  # 边聚合, CGNN
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


class FDGNN(nn.Module):
    """ 参数共享的联邦GNN """
    def __init__(self):
        super().__init__()
        self.mlp_m = MLP([2 * Nt_num, 32, 64])
        self.mlp_u = MLP([64 + 2 * Nt_num, 32, 2 * Nt_num])  # 参数共享,参数待调整,需要增大，增加学习能力
        self.hconv_edge = HeteroGConv(self.mlp_m, self.mlp_u)
        self.hconv = HeteroConv({
            ('served', 'conn', 'interfered'): self.hconv_edge,
            ('interfered', 'conn', 'served'): self.hconv_edge
        })
        # self.h2o = Seq(Lin(2 * Nt_num, 2 * Nt_num, bias=True), Tanh())
        # self.h2o = Seq(Lin(2 * Nt_num, 2 * Nt_num, bias=True), PowerNormalization(P_max))

        # 输出层拆分为角度和功率两部分
        # self.angle_output = Seq(Lin(2 * Nt_num, Nt_num), Tanh())  # 角度范围 [-1, 1]，后续映射到 [-pi, pi]
        # self.power_output = Seq(Lin(2 * Nt_num, Nt_num), ReLU())  # 功率非负
        # self.power_output = Seq(Lin(2 * Nt_num, Nt_num), Sigmoid())  # 对每天线进行功率约束,错误
        # self.power_output = Seq(
        #     Lin(2 * Nt_num, Nt_num),
        #     Sigmoid(),
        #     PowerConstraintLayer(Nt_num, P_max_per_antenna=1)
        # )

        self.bf_output = Seq(Lin(2 * Nt_num, 2 * Nt_num), Tanh(),  # 角度部分，实部虚部均归一化为[-1, 1]
            PowerConstraintLayer(Nt_num, P_max_per_antenna_norm=1))  # 对各天线的最大发射功率约束为单位1
        # 自己设计的输出层，等于将以原点为中心的正方形的可行域化为单位圆


        # self.h2o = MLP([graph_embedding_size, 16])
        # self.h2o = Seq(*[self.h2o, Seq(Lin(16, 1, bias=True), Tanh())])
        # 原为bias=True & Sigmoid(0,1) 改为 bias=False & tanh(-1,1)

    def forward(self, data: HeteroData):
        # 初始特征
        # x0_dict = {
        #     'served': data['served'].x,
        #     'interfered': data['interfered'].x
        # }
        x0_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # 消息传递
        x1_dict = self.hconv(x0_dict, edge_index_dict)
        x2_dict = self.hconv(x1_dict, edge_index_dict)  # 边上后续可加注意力
        out_dict = self.hconv(x2_dict, edge_index_dict)

        out_bfm = out_dict['served']  # 在served_UE节点上输出对应beamforming_vector
        Beamforming_Matrix = self.bf_output(out_bfm)  # 对各天线的最大发射功率进行约束

        # 输出角度和功率
        # angles = self.angle_output(out_bfm) * np.pi  # 映射到 [-pi, pi]
        # powers = self.power_output(out_bfm)
        # # 组合成复数波束成形向量
        # real_part = powers * torch.cos(angles)
        # imag_part = powers * torch.sin(angles)
        # Beamforming_Matrix = torch.cat((real_part, imag_part), dim=1)

        # p_view = torch.norm(Beamforming_Matrix)
        # print("\nBS's T_power_norm: ", p_view)

        return Beamforming_Matrix


class CGNN(nn.Module):
    """ 参数共享 """
    def __init__(self):
        super().__init__()
        self.mlp_m = MLP([2 * 2 * Nt_num, 32, 32])  # 注意和FDNN不一样，因为有边特征的concat
        self.mlp_u = MLP([32 + 2 * Nt_num, 32, 2 * Nt_num])  # 参数共享,参数待调整
        self.hconv_edge = HeteroEConv(self.mlp_m, self.mlp_u)
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


def Notrain_FDGNN():
    total_loss = 0.0
    total_SINR_rate = 0.0
    num_batches = len(train_subgraph_list_FDGNN) // batch_size_FDGNN + (1 if len(train_subgraph_list_FDGNN) % batch_size_FDGNN != 0 else 0)
    for batch_idx in range(num_batches):
        output_list_layout = []
        batch_start = batch_idx * batch_size_FDGNN
        batch_end = min(batch_start + batch_size_FDGNN, len(train_subgraph_list_FDGNN))
        batch_data = train_subgraph_list_FDGNN[batch_start:batch_end]
        batch_loss = 0.0
        optimizer_FDGNN.zero_grad()
        for data in batch_data:
            data = data.to(device)
            output = model_FDGNN(data)
            output_list_layout.append(output)
            loss = compute_Sum_SLNR_rate(output, data)
            batch_loss += loss
        total_loss += batch_loss
        # compute SINR rate
        layout_data = global_graph_list[batch_idx].to(device)
        layout_sum_rate = compute_Sum_SINR_rate_d(output_list_layout, layout_data)
        total_SINR_rate += layout_sum_rate
    return total_loss / len(train_subgraph_list_FDGNN) * BS_num_per_Layout, total_SINR_rate / num_batches


def train_FDGNN():
    total_loss = 0.0
    total_SINR_rate = 0.0
    num_batches = len(train_subgraph_list_FDGNN) // batch_size_FDGNN + (1 if len(train_subgraph_list_FDGNN) % batch_size_FDGNN != 0 else 0)
    for batch_idx in range(num_batches):
        output_list_layout = []
        batch_start = batch_idx * batch_size_FDGNN
        batch_end = min(batch_start + batch_size_FDGNN, len(train_subgraph_list_FDGNN))
        batch_data = train_subgraph_list_FDGNN[batch_start:batch_end]
        batch_loss = 0.0
        optimizer_FDGNN.zero_grad()
        for data in batch_data:
            data = data.to(device)
            output = model_FDGNN(data)
            output_list_layout.append(output)
            loss = compute_Sum_SLNR_rate(output, data)
            # loss.backward()
            batch_loss += loss
        batch_loss.backward()  # batch's sum_loss
        # avg_batch_loss = batch_loss / len(batch_data)
        # avg_batch_loss.backward()  # batch's sum_loss
        # 这里可能还是有问题，因为相当于是对全局进行后向传递，而不是对各基站的local SLNR 进行backward
        optimizer_FDGNN.step()
        total_loss += batch_loss
        # compute SINR rate
        layout_data = global_graph_list[batch_idx].to(device)
        layout_sum_rate = compute_Sum_SINR_rate_d(output_list_layout, layout_data)
        total_SINR_rate += layout_sum_rate
        # print(f'Batch {batch_idx + 1}/{num_batches}, Layout\'s SLNR rate: {batch_loss.item()}')
        # print(f'Batch {batch_idx + 1}/{num_batches}, Layout\'s SINR rate: {layout_sum_rate.item()}')
    # batch_sum_rate = compute_Sum_SINR_rate_t(output_list_train, topology_train_FDGNN)
    # avg_sum_rate = batch_sum_rate / Layouts_num
    return total_loss / len(train_subgraph_list_FDGNN) * BS_num_per_Layout, total_SINR_rate / num_batches


def test_FDGNN():
    model_FDGNN.eval()
    output_list = []
    with torch.no_grad():
        layout_sum_loss = 0
        for data in subgraph_list_test:
            data = data.to(device)
            output = model_FDGNN(data)
            output_list.append(output)
            loss = compute_Sum_SLNR_rate(output, data)
            layout_sum_loss += loss
        print(f'Test FDGNN Layout‘s sum loss (SLNR)：{layout_sum_loss}')
        # layout_sum_rate = compute_Sum_SINR_rate_t(output_list, topology_test)
        # layout_sum_rate_d = compute_Sum_SINR_rate_d(output_list, global_graph_test)
        # print(f'Test FDGNN Layout‘s sum rate based topology：{layout_sum_rate}')
        # print(f'Test FDGNN Layout‘s sum rate based pyg_data：{-layout_sum_rate_d}')

        # layout_sum_rate = compute_Sum_SINR_rate_t(output_list, topology_train_FDGNN) / Layouts_num  # 训练集数据
        # print(f'Test FDGNN Layout‘s sum rate based topology：{layout_sum_rate}')

    return output_list


def train_CGNN():
    model_CGNN.train()
    total_loss = 0
    # for data, topology in zip(train_loader_CGNN, global_topology_list):
    for data in train_loader_CGNN:
        data = data.to(device)
        optimizer_CGNN.zero_grad()
        out_list = model_CGNN(data)
        # loss = compute_Sum_SINR_rate(out_list, topology)
        loss = compute_Sum_SINR_rate_d(out_list, data)
        loss.backward()
        total_loss += loss.item() * data.num_graphs  # 为什么要乘？因为在sr_rate中有mean，所以可以直接乘
        optimizer_CGNN.step()
    return total_loss / Layouts_num


def test_CGNN():
    model_CGNN.eval()
    with torch.no_grad():
        out_list = model_CGNN(global_graph_test)
        # loss = compute_Sum_SINR_rate_d(out_list, global_graph_test)
        # print(f'Test CGNN Layout‘s sum rate：{-loss}')
    return out_list


# 初始化参数
Scale_exponent = 1  # 区域扩大乘数
BS_num_per_Layout = 16 * Scale_exponent  # 取值应满足可开方
Layouts_num = 32  # 可调整
BS_num = BS_num_per_Layout * Layouts_num  # total BS/cell num, 相当于生成一张大图
avg_UE_num = 2  # average UE_num per cell
PathLoss_exponent = 4.5  # 路径衰落系数（常取3~5），与形成干扰的半径有关, 4的半径有点大，可设为5
Region_size = 120 * int(np.sqrt(Scale_exponent))  # Layout生成区域边长,注意与小区数匹配，4*4小区对应120
Nt_num = 8  # MISO信道发射天线数

Batchsize_per_BS = 1  # 取值应可被Layouts_num整除
graph_embedding_size = 8
# N0 = 1e-12
N0 = 1e-8
shuffle_FDGNN = False  # FDGNN训练子图list是否打乱顺序

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FDGNN Data Generation
# 法一，从超大图提取子图，这样训练有利于扩展性
# topology_train_FDGNN = LG.generate_topology(
#     N=BS_num,
#     K=avg_UE_num,
#     pl_exponent=PathLoss_exponent,
#     region_size=Region_size,
#     Nt=Nt_num
# )
# # 提取子图并转化为pyg异构图结构 (已包含数据标准化)
# '''数据标准化可能需要改进，具体见OneNote'''
# subgraph_list = LG.cell_convert_to_pyg(topology_train_FDGNN)


# 生成拓扑结构
# 后续对比时，subgraph_list也可以这样生成，从每一个中图中拿
# 也可以不改动，增加美观结果的概率
global_topology_list = []
global_graph_list = []
all_subgraph_list = []
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
    all_subgraph_list.extend(LG.cell_convert_to_pyg(topology_train))  # 用于FDGNN训练

train_subgraph_list_FDGNN = all_subgraph_list
if(shuffle_FDGNN == True):
    random.shuffle(train_subgraph_list_FDGNN)  # 可选择打乱FDGNN的训练子图顺序

batch_size_FDGNN = BS_num_per_Layout * Batchsize_per_BS  # FDGNN

train_loader_CGNN = DataLoader(global_graph_list, batch_size=1, shuffle=False, num_workers=0)  # 暂设为1和False


# 加载图数据为批次
'''数据集导入见OneNote'''

model_FDGNN = FDGNN().to(device)
optimizer_FDGNN = torch.optim.Adam(model_FDGNN.parameters(), lr=0.001)
scheduler_FDGNN = torch.optim.lr_scheduler.StepLR(optimizer_FDGNN, step_size=40, gamma=0.9)  # 每step_size个epoch调整一次

model_CGNN = CGNN().to(device)
optimizer_CGNN = torch.optim.Adam(model_CGNN.parameters(), lr=0.001)
scheduler_CGNN = torch.optim.lr_scheduler.StepLR(optimizer_CGNN, step_size=40, gamma=0.9)  # 学习率调整

# 前向传播
num_epochs_FDGNN = 20
num_epochs_CGNN = 20
train_log_FDGNN = []
train_log_CGNN = []

# Train FDGNN
orig_loss_FDGNN, orig_SINR_rate_FDGNN = Notrain_FDGNN()
print(f'FDGNN Oringinal Average Layout\'s Loss (SLNR): {orig_loss_FDGNN}, Average Layout\'s Loss (SINR): {orig_SINR_rate_FDGNN}')
# 以subgraph的list为样本进行训练
for epoch in range(num_epochs_FDGNN):
    model_FDGNN.train()
    epoch_loss_FDGNN, epoch_SINR_rate_FDGNN = train_FDGNN()
    scheduler_FDGNN.step()  # 更新学习率,可暂时不
    print(f'FDGNN Epoch {epoch + 1}/{num_epochs_FDGNN}, Average Layout\'s Loss (SLNR): {epoch_loss_FDGNN}, '
          f'Average Layout\'s Loss (SINR): {epoch_SINR_rate_FDGNN}')
    train_log_FDGNN.append(-epoch_loss_FDGNN.item())
print('FDGNN Training finished.')

train_log_array_FDGNN = np.array(train_log_FDGNN)  # 将列表转换为NumPy数组
data_to_save_FDGNN = {'train_loss': train_log_array_FDGNN}  # 创建字典
savemat('train_loss_FDGNN.mat', data_to_save_FDGNN)  # 保存为.mat文件

torch.save(model_FDGNN, 'model_FDGNN.pth')
print('model_FDGNN saved.')

for epoch in range(num_epochs_CGNN):
    model_CGNN.train()
    epoch_loss_CGNN = train_CGNN()
    print(f'CGNN epoch: {epoch+1}/{num_epochs_CGNN}, Average Layout\'s Loss (SINR): {epoch_loss_CGNN}')
    train_log_CGNN.append(epoch_loss_CGNN)
    scheduler_CGNN.step()  # 动态调整优化器学习率
print('CGNN Training finished.')

train_log_array_CGNN = np.array(train_log_CGNN)  # 将列表转换为NumPy数组
data_to_save_CGNN = {'train_loss': train_log_array_CGNN}  # 创建字典
savemat('train_loss_CGNN.mat', data_to_save_CGNN)  # 保存为.mat文件

torch.save(model_CGNN, 'model_CGNN.pth')
print('model_CGNN saved.')

# test data generator
BS_num_per_Layout_test = 16
topology_test = LG.generate_topology(
    N=BS_num_per_Layout_test, K=avg_UE_num,
    pl_exponent=PathLoss_exponent,
    region_size=Region_size,
    Nt=Nt_num
)
subgraph_list_test = LG.cell_convert_to_pyg(topology_test)
global_graph_test = LG.global_convert_to_pyg(topology_test).to(device)

# test
out_FDGNN = test_FDGNN()
out_CGNN = test_CGNN()
out_ZF = utils.compute_ZF_beamforming(global_graph_test, Nt_num)

rate_FDGNN = compute_Sum_SINR_rate_d(out_FDGNN, global_graph_test)
rate_CGNN = compute_Sum_SINR_rate_d(out_CGNN, global_graph_test)
rate_ZF = compute_Sum_SINR_rate_d(out_ZF, global_graph_test)

# print(f'Test FDGNN Layout‘s sum rate based topology：{layout_sum_rate}')
print(f'FDGNN Layout_test‘s sum rate based pyg_data：{rate_FDGNN}')
print(f'CGNN Layout_test‘s sum rate based pyg_data：{rate_CGNN}')
print(f'ZF Layout_test‘s sum rate based pyg_data：{rate_ZF}')

print('Testing finished.')

# plot train loss
plt.plot(train_log_FDGNN, label='FDGNN')
plt.plot(train_log_CGNN, label='CGNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.show()