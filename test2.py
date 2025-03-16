import torch
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Tanh
from torch_geometric.loader import DataLoader
import LayoutsGenerator as LG


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())#, BN(channels[i])
        for i in range(1, len(channels))
    ])


class HeteroGConv(MessagePassing):  # 边聚合
    """ 参数共享的异构消息传递层 """
    def __init__(self, mlp_m, mlp_u):
        super().__init__(aggr='sum')
        # super().__init__(aggr='max')
        self.msg_mlp = mlp_m  # 这样写其实会使参数在内存冗余
        self.update_mlp = mlp_u

    def forward(self, x_dict, edge_index):
        return self.propagate(edge_index, x=x_dict)

    def message(self, x_i, x_j):
        """ 消息生成 """
        msg = self.msg_mlp(x_j)
        return msg

    def update(self, aggr_out):
        """ 节点更新 """
        update = self.update_mlp(aggr_out)
        return update


class FDGNN(nn.Module):
    """ 参数共享的联邦GNN """
    def __init__(self):
        super().__init__()
        self.mlp_m = MLP([2 * Nt_num, 32, 2 * Nt_num])
        self.mlp_u = MLP([2 * Nt_num, 16, 2 * Nt_num])  # 参数共享,参数待调整
        self.hconv_edge = HeteroGConv(self.mlp_m, self.mlp_u)
        self.hconv = HeteroConv({
            ('served', 'conn', 'interfered'): self.hconv_edge,
            ('interfered', 'conn', 'served'): self.hconv_edge
        })
        # self.hconv = HeteroConv({
        #     ('served', 'conn', 'interfered'): HeteroGConv(self.mlp_m, self.mlp_u),
        #     ('interfered', 'conn', 'served'): HeteroGConv(self.mlp_m, self.mlp_u)
        # }, aggr='sum')
        self.h2o = Seq(Lin(2 * Nt_num, 2 * Nt_num, bias=True), Tanh())

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
        print(edge_index_dict)


        # 消息传递
        x1_dict = self.hconv(x0_dict, edge_index_dict)
        x2_dict = self.hconv(x1_dict, edge_index_dict)  # 边上后续可加注意力
        out_dict = self.hconv(x2_dict, edge_index_dict)

        out_bfv_dict = out_dict['served']  # 在served_UE节点上输出beamforming_vector
        Beamforming_Matrix = self.h2o(out_bfv_dict)  # 注意功率约束

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

    Batchsize_per_BS = 8  # 取值应可被Layouts_num整除
    graph_embedding_size = 8

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
    train_loader = DataLoader(subgraph_list, batch_size=BS_num_per_Layout * Batchsize_per_BS, shuffle=False,
                              num_workers=4)
    # 禁止重排序是为了更好模拟各layout数量（按layout进行规范化的）
    # 合成一张大图，但各子图互不相连
    # 不能用DataLoader，子图形状不定，后面计算本地损失函数时没法拆分，而且索引可能有误

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FDGNN().to(device)
    # model = torch.compile(model)  # torch 2.0.1可以进行编译优化,but Windows not yet supported for torch.compile

    print("end")

    # 前向传播
    for data in train_loader:
        data = data.to(device)
        output = model(data)
        print(f"输出维度: {output.shape}")  # 应为 torch.Size([1, 1])