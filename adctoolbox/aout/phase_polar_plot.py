"""
Phase-domain Error Polar Plot
相位域误差极坐标图（用于图(c) Phase-domain error）

功能：
- 将误差投影到极坐标系统
- 显示误差在360度相位范围内的分布
- 使用极坐标图展示

作者：ADC Toolbox
日期：2025-11-21
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def phase_domain_polar_plot(data, fin=None, output_path=None):
    """
    生成相位域误差极坐标图

    参数:
        data: ADC输出数据 (1D numpy数组)
        fin: 归一化输入频率 f_in/f_sample (None则自动检测)
        output_path: 输出图片路径 (None则不保存)

    返回:
        phase_deg: 相位数组（度数）
        error: 误差数组
    """

    # 确保data是1D数组
    data = np.asarray(data).flatten()
    N = len(data)

    # 自动检测频率
    if fin is None or fin == 0:
        try:
            from findFin import findFin
            fin = findFin(data, 1)
        except ImportError:
            spec = np.abs(np.fft.fft(data))
            spec[0] = 0
            bin_max = np.argmax(spec[:N//2])
            fin = bin_max / N
            print(f"Warning: findFin not found, using FFT: fin = {fin:.6f}")

    # 执行Thompson分解获取误差和相位
    try:
        from tomDecomp import tomDecomp
        signal, error, indep, dep, phi = tomDecomp(data, fin, order=1, disp=0)
    except ImportError:
        print("Warning: tomDecomp not found, using simple sine fit")
        t = np.arange(N)
        SI = np.cos(t * fin * 2 * np.pi)
        SQ = np.sin(t * fin * 2 * np.pi)
        WI = np.mean(SI * data) * 2
        WQ = np.mean(SQ * data) * 2
        DC = np.mean(data)
        signal = DC + SI * WI + SQ * WQ
        error = data - signal
        phi = -np.arctan2(WQ, WI)

    # 计算每个样本的相位（度数）
    t = np.arange(N)
    phase_rad = (phi + 2 * np.pi * fin * t) % (2 * np.pi)
    phase_deg = np.rad2deg(phase_rad)

    # 可视化
    if output_path is not None or True:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')

        # 绘制散点图
        ax.scatter(phase_rad, error, c='blue', s=1, alpha=0.3, marker='.')

        # 设置标题和标签
        ax.set_title('(c) Phase-domain error\nSpectrum Phase',
                     fontsize=14, fontweight='bold', pad=20)

        # 设置径向范围
        error_max = np.max(np.abs(error))
        ax.set_ylim([-error_max * 1.1, error_max * 1.1])

        # 设置角度标签
        ax.set_theta_zero_location('N')  # 0度在顶部
        ax.set_theta_direction(1)  # 顺时针

        # 添加径向网格标签
        ax.set_rlabel_position(0)

        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Phase-domain polar plot saved to: {output_path}")

        plt.close()

    return phase_deg, error


def test_phase_polar_plot():
    """测试相位域极坐标图"""

    print("=" * 70)
    print("Phase-domain Polar Plot - 测试")
    print("=" * 70)

    # 生成测试数据
    N = 4096
    fs = 1e6
    fin = 28320.3125
    re_fin = fin / fs

    t = np.arange(N) / fs

    # 理想信号 + 相位相关失真 + 噪声
    signal_ideal = np.sin(2 * np.pi * fin * t) * 1000 + 2048

    # 添加3次谐波失真（相位相关）
    phase = (2 * np.pi * fin * t) % (2 * np.pi)
    distortion = 50 * np.sin(3 * phase)

    # 添加随机噪声
    noise = 10 * np.random.randn(N)

    # 合成ADC输出
    adc_output = signal_ideal + distortion + noise

    print(f"\n测试参数:")
    print(f"  采样点数: {N}")
    print(f"  归一化频率: {re_fin:.10f}")
    print(f"  失真幅度: 50 LSB")
    print(f"  噪声RMS: 10 LSB")

    # 执行分析
    output_path = os.path.join(os.path.dirname(__file__), "..", "output_data", "test_phase_polar.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    phase_deg, error = phase_domain_polar_plot(adc_output, fin=re_fin, output_path=output_path)

    print(f"\n结果:")
    print(f"  误差RMS: {np.sqrt(np.mean(error**2)):.2f} LSB")
    print(f"  误差范围: [{np.min(error):.2f}, {np.max(error):.2f}]")

    print(f"\n[OK] Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_phase_polar_plot()
