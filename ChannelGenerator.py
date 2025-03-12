import numpy as np


def generate_miso_channel(Nt, distance, reference_distance=100.0, path_loss_exponent=3.0, carrier_freq=2e9):
    """
    生成含路径损耗的瑞利MISO信道
    :param Nt: 基站天线数
    :param distance: 用户距离（米）
    :param reference_distance: 参考距离（默认100米）
    :param path_loss_exponent: 路径损耗指数（默认3.0，城市环境）
    :param carrier_freq: 载波频率（默认2GHz）
    :return: 信道向量 h ∈ C^Nt
    """
    # 生成小尺度瑞利衰落
    h_small = (np.random.randn(Nt) + 1j * np.random.randn(Nt)) / np.sqrt(2)

    # 计算路径损耗（dB）
    wavelength = 3e8 / carrier_freq  # 波长计算
    PL0 = 20 * np.log10(4 * np.pi * reference_distance / wavelength)  # 参考距离处的自由空间损耗
    PL = PL0 + 10 * path_loss_exponent * np.log10(distance / reference_distance)

    # 转换为线性衰减因子
    linear_attenuation = 10 ** (-PL / 20)  # 电压幅度的衰减因子

    # 总信道响应
    h_total = linear_attenuation * h_small
    return h_total


# 示例使用
if __name__ == "__main__":
    Nt = 4  # 基站天线数
    distance = 500  # 用户距离基站500米

    # 生成信道
    h = generate_miso_channel(Nt, distance)

    print("信道向量（含路径损耗）:\n", h)
    print("\n信道幅度:\n", np.abs(h))
    print("\n信道功率（|h|²）:\n", np.round(np.abs(h) ** 2, 4))
    print("\n总平均功率:", np.round(np.mean(np.abs(h) ** 2), 4))