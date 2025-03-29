import torch

def compute_ZF_beamforming(data, Nt):
    """
    from doubao
    计算 ZF 波束成形矩阵
    :param data: PyG 异构图数据
    :param Nt: 基站天线数量
    :return: ZF 波束成形矩阵列表
    """
    service_edge_index = data['BS', 'service', 'UE'].edge_index
    service_channels = data['BS', 'service', 'UE'].original_channel  # 按ue_idx排列
    num_bs = data['BS'].x.shape[0]

    ZF_beamforming_matrices = []
    for bs_idx in range(num_bs):
        # 获取当前基站服务的用户索引
        user_indices = service_edge_index[1, service_edge_index[0] == bs_idx]
        if len(user_indices) == 0:
            ZF_beamforming_matrices.append(torch.empty(0, 2 * Nt))
            continue

        # 获取当前基站到服务用户的信道矩阵
        # channel_matrix = service_channels[user_indices]
        channel_matrix = service_channels[service_edge_index[0] == bs_idx]
        channel_matrix = channel_matrix.view(-1, Nt, 2)  # 分离实部和虚部
        channel_matrix = torch.complex(channel_matrix[..., 0], channel_matrix[..., 1])  # 转换为复数张量

        # 计算信道矩阵的伪逆
        H_H = torch.conj(channel_matrix).T  # 信道矩阵的共轭转置
        H_pinv = torch.pinverse(H_H @ channel_matrix) @ H_H  # 伪逆

        # 归一化波束成形向量
        norms = torch.norm(H_pinv, dim=0, keepdim=True)
        H_pinv_normalized = H_pinv / norms

        # 将复数波束成形矩阵转换为实虚部拼接的形式
        real_part = H_pinv_normalized.real
        imag_part = H_pinv_normalized.imag
        ZF_beamforming_matrix = torch.cat([real_part.T, imag_part.T], dim=1)

        ZF_beamforming_matrices.append(ZF_beamforming_matrix)

    return ZF_beamforming_matrices


