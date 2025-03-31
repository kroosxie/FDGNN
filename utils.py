import numpy as np
import torch

def compute_ZF_beamforming_global(data, Nt):
    """
    计算 ZF 波束成形矩阵（修正版）
    :param data: 包含信道信息的异构图数据
    :param Nt: 基站天线数量
    :return: ZF 波束成形矩阵列表
    """
    service_edge_index = data['BS', 'service', 'UE'].edge_index
    service_channels = data['BS', 'service', 'UE'].original_channel
    num_bs = data['BS'].x.shape[0]

    ZF_beamforming_matrices = []
    for bs_idx in range(num_bs):
        # 获取当前基站服务的用户索引
        user_mask = service_edge_index[0] == bs_idx
        user_indices = service_edge_index[1, user_mask]
        if len(user_indices) == 0:
            ZF_beamforming_matrices.append(torch.empty(0, 2 * Nt))
            continue

        # 构建复数信道矩阵 [K, Nt]
        H_bs = service_channels[user_mask]
        H_real = H_bs[:, :Nt]
        H_imag = H_bs[:, Nt:]
        H = torch.complex(H_real, H_imag)  # [K, Nt]

        # 计算右伪逆：H^H (H H^H)^{-1}
        H_H = H.conj().T  # [Nt, K]
        H_HH = H @ H_H  # [K, K]
        H_pinv = H_H @ torch.linalg.pinv(H_HH)  # [Nt, K]

        # 总功率归一化
        norm = torch.norm(H_pinv)
        # norm = torch.norm(H_pinv, dim=0, keepdim=True)  # 天线发射功率约束，注意这里的dim=0应该是对每一列（各天线功率)进行归一化
        H_pinv_normalized = H_pinv / norm

        # 转换为实虚部拼接格式 [K, 2*Nt]
        real_part = H_pinv_normalized.real.T  # [K, Nt]
        imag_part = H_pinv_normalized.imag.T  # [K, Nt]
        ZF_matrix = torch.cat([real_part, imag_part], dim=1)

        ZF_beamforming_matrices.append(ZF_matrix)

    return ZF_beamforming_matrices

def compute_ZF_beamforming_local(data, Nt):
    """
    计算 ZF 波束成形矩阵（修正版）
    :param data: 包含信道信息的异构图数据
    :param Nt: 基站天线数量
    :return: ZF 波束成形矩阵列表
    """
    service_channels = data['served'].original_channel

    # 构建复数信道矩阵 [K, Nt]
    H_bs = service_channels
    H_real = H_bs[:, :Nt]
    H_imag = H_bs[:, Nt:]
    H = torch.complex(H_real, H_imag)  # [K, Nt]

    # 计算右伪逆：H^H (H H^H)^{-1}
    H_H = H.conj().T  # [Nt, K]
    H_HH = H @ H_H  # [K, K]
    H_pinv = H_H @ torch.linalg.pinv(H_HH)  # [Nt, K]

    # 总功率归一化
    norm = torch.norm(H_pinv)
    # norm = torch.norm(H_pinv, dim=0, keepdim=True)  # 天线发射功率约束，注意这里的dim=0应该是对每一列（各天线功率)进行归一化
    H_pinv_normalized = H_pinv / norm

    # 转换为实虚部拼接格式 [K, 2*Nt]
    real_part = H_pinv_normalized.real.T  # [K, Nt]
    imag_part = H_pinv_normalized.imag.T  # [K, Nt]
    ZF_matrix = torch.cat([real_part, imag_part], dim=1)

    return ZF_matrix


