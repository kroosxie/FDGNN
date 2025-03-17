import torch
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh, LeakyReLU, BatchNorm1d as BN, Softmax
from torch_geometric.loader import DataLoader
import LayoutsGenerator as LG
import numpy as np


def compute_Sum_SLNR_rate(output: torch.Tensor, direct_h: torch.Tensor, interf_h: torch.Tensor) -> torch.Tensor:
    slnr_value = compute_SLNR(output, direct_h, interf_h)  # 能一一对应吗？
    slnr_rate = torch.log2(1 + slnr_value)
    sum_slnr_rate = torch.sum(slnr_rate)
    loss = torch.neg(sum_slnr_rate)
    return loss


def compute_SLNR(output: torch.Tensor, direct_h: torch.Tensor, interf_h: torch.Tensor) -> torch.Tensor:
    # Nt = direct_h.size(1) // 2  # 获取天线数Nt
    Nt = Nt_num

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

        # 计算分母：Σ|interf_h^H u|² + N0（N0设为1e-8）
        denominator = N0
        for interf_user in interf_h_c:
            term = torch.abs(torch.sum(torch.conj(interf_user) * output_c[served_idx])) ** 2
            denominator += term

        # 计算(numerator / denominator)
        slnr = numerator / denominator
        results.append(slnr)

    return torch.stack(results).unsqueeze(1)  # 转为[服务用户数, 1]维度


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), LeakyReLU())#, BN(channels[i]))  # ReLU改为了LeakyReLU, 加入了BN
        for i in range(1, len(channels))
    ])


# class PowerNormalization(nn.Module):
#     def __init__(self, P_max=1.0, eps=1e-8):
#         super().__init__()
#         self.P_max = P_max
#         self.eps = eps
#
#     def forward(self, W_raw):
#         # 输入形状: (K, 2*Nt)
#         real = W_raw[..., :W_raw.shape[-1] // 2]  # 实部
#         imag = W_raw[..., W_raw.shape[-1] // 2:]  # 虚部
#         # 计算总功率 (batch_size, 1, 1)
#         power = torch.sum(real ** 2 + imag ** 2, dim=(-2, -1), keepdim=True)
#         # 计算缩放因子
#         scaling = torch.sqrt(self.P_max / (power + self.eps))
#         # 缩放并合并
#         real_norm = real * scaling
#         imag_norm = imag * scaling
#         return torch.cat([real_norm, imag_norm], dim=-1)


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

    def message(self, x_i, x_j):
        """ 消息生成 """
        msg = self.msg_mlp(x_j)
        return msg

    def update(self, aggr_out, x):
        """ 节点更新 """
        src = x[1]
        tmp = torch.cat([src, aggr_out], dim=1)
        update = self.update_mlp(tmp)
        return update


class FDGNN(nn.Module):
    """ 参数共享的联邦GNN """
    def __init__(self):
        super().__init__()
        self.mlp_m = MLP([2 * Nt_num, 32, 32])
        self.mlp_u = MLP([32 + 2 * Nt_num, 32, 2 * Nt_num])  # 参数共享,参数待调整
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
        # 自己设计的，等于将以原点为中心的正方形的可行域化为单位圆


        # self.h2o = MLP([graph_embedding_size, 16])
        # self.h2o = Seq(*[self.h2o, Seq(Lin(16, 1, bias=True), Tanh())])
        # 原为bias=True & Sigmoid(0,1) 改为 bias=False & tanh(-1,1)

    def forward(self, data: HeteroData):
        # 初始特征
        x0_dict = {
            'served': data['served'].x,
            'interfered': data['interfered'].x
        }
        edge_index_dict = data.edge_index_dict

        # 消息传递
        x1_dict = self.hconv(x0_dict, edge_index_dict)
        x2_dict = self.hconv(x1_dict, edge_index_dict)  # 边上后续可加注意力
        out_dict = self.hconv(x2_dict, edge_index_dict)

        out_bfm = out_dict['served']  # 在served_UE节点上输出beamforming_vector
        # Beamforming_Matrix = self.h2o(out_bfm)  # 注意功率约束
        Beamforming_Matrix = self.bf_output(out_bfm)

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
    N0 = 1e-12
    P_max = 6

    # 生成拓扑结构
    topology = LG.generate_topology(
        N=BS_num,
        K=avg_UE_num,
        pl_exponent=PathLoss_exponent,
        region_size=Region_size,
        Nt=Nt_num
    )

    # 提取子图并转化为pyg异构图结构 (已包含数据标准化)
    '''数据标准化可能需要改进，具体见OneNote'''
    subgraph_list = LG.cell_convert_to_pyg(topology)

    # 加载图数据为批次
    '''数据集导入见OneNote'''
    # train_loader = DataLoader(subgraph_list, batch_size=BS_num_per_Layout * Batchsize_per_BS, shuffle=False,
    #                           num_workers=0)
    # 禁止重排序是为了更好模拟各layout数量（按layout进行规范化的）
    # 合成一张大图，但各子图互不相连
    # 不能用DataLoader，子图形状不定，后面计算本地损失函数时没法拆分，而且索引可能有误

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FDGNN().to(device)
    # model = torch.compile(model)  # torch 2.0.1可以进行编译优化,but Windows not yet supported for torch.compile
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # 前向传播
    batch_size = BS_num_per_Layout * Batchsize_per_BS
    num_epochs = 80

    # for data in train_loader:
    # for data in subgraph_list:
    #     data = data.to(device)
    #     output = model(data)
    #     # print(f"输出维度: {output.shape}")
    #     direct_h = data['served'].original_channel
    #     interf_h = data['interfered'].original_channel
    #     loss = compute_Sum_SLNR_rate(output, direct_h, interf_h)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(subgraph_list) // batch_size + (1 if len(subgraph_list) % batch_size != 0 else 0)
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(subgraph_list))
            batch_data = subgraph_list[batch_start:batch_end]
            batch_loss = 0.0
            optimizer.zero_grad()

            for data in batch_data:
                data = data.to(device)
                output = model(data)
                direct_h = data['served'].original_channel
                interf_h = data['interfered'].original_channel
                loss = compute_Sum_SLNR_rate(output, direct_h, interf_h)
                # loss.backward()
                batch_loss += loss
            batch_loss.backward()  # batch's sum_loss

            optimizer.step()

            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}, Loss: {batch_loss.item()}')
            epoch_loss += batch_loss

            # 更新学习率
        scheduler.step()

        # 打印每个 epoch 的平均损失
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(subgraph_list)}')

    print('Training finished.')


