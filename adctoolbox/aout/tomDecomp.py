"""
tomDecomp.m的Python移植版本
Thompson Decomposition - Thompson误差分解算法

将ADC输出分解为:
- signal: 理想信号(DC +基波 + 指定阶数谐波)
- error: 总误差
- indep: 独立误差(随机噪声)
- dep: 依赖误差(相位相关误差)
- phi: 基波相位

原始MATLAB代码: matlab_reference/tomDecomp.m
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加findFin模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def tomDecomp(data, re_fin=None, order=10, disp=1):
    """
    Thompson Decomposition - Thompson误差分解

    参数:
        data: ADC输出数据, 1D numpy数组
        re_fin: 相对输入频率(归一化频率 f_in/f_sample), 如果为None则自动检测
        order: 谐波阶数,用于计算依赖误差(默认10,表示基波+前10次谐波视为依赖误差)
        disp: 是否显示结果图(0/1), 默认1

    返回:
        signal: 理想信号 (DC + 基波)
        error: 总误差 (data - signal)
        indep: 独立误差 (随机噪声部分)
        dep: 依赖误差 (谐波失真部分)
        phi: 基波相位 (弧度)

    原理:
        signal = DC + WI*cos(ωt) + WQ*sin(ωt)  # 仅基波
        signal_all = DC + Σ[WI_k*cos(kωt) + WQ_k*sin(kωt)]  # 基波+谐波
        error = data - signal
        indep = data - signal_all  # 去除基波和谐波后的残差
        dep = signal_all - signal  # 谐波成分
    """

    # 确保data是列向量
    data = np.asarray(data).flatten()
    N = len(data)

    # 如果没有提供归一化频率,则自动检测
    if re_fin is None or np.isnan(re_fin):
        try:
            from findFin import findFin
            re_fin = findFin(data)
        except ImportError:
            # 简单FFT频率检测
            spec = np.abs(np.fft.fft(data))
            spec[0] = 0
            bin_max = np.argmax(spec[:N//2])
            re_fin = bin_max / N
            print(f"Warning: findFin not found, using simple FFT detection: re_fin = {re_fin:.6f}")

    # 时间轴
    t = np.arange(N)

    # 计算基波的I/Q分量
    SI = np.cos(t * re_fin * 2 * np.pi)
    SQ = np.sin(t * re_fin * 2 * np.pi)

    # 估计基波权重和DC
    WI = np.mean(SI * data) * 2
    WQ = np.mean(SQ * data) * 2
    DC = np.mean(data)

    # 重建信号 (仅基波)
    signal = DC + SI * WI + SQ * WQ

    # 基波相位
    phi = -np.arctan2(WQ, WI)

    # 构建多阶谐波矩阵
    SI_matrix = np.zeros((N, order))
    SQ_matrix = np.zeros((N, order))

    for ii in range(order):
        SI_matrix[:, ii] = np.cos(t * re_fin * (ii + 1) * 2 * np.pi)
        SQ_matrix[:, ii] = np.sin(t * re_fin * (ii + 1) * 2 * np.pi)

    # 合并I/Q矩阵
    A = np.column_stack([SI_matrix, SQ_matrix])

    # 最小二乘求解谐波权重
    W, residuals, rank, s = np.linalg.lstsq(A, data, rcond=None)

    # 重建信号 (DC + 基波 + 谐波)
    signal_all = DC + A @ W

    # 误差分解
    error = data - signal
    indep = data - signal_all
    dep = signal - signal_all  # Fixed: was signal_all - signal

    # Visualization
    if disp:
        # Only create new figure if one doesn't exist
        if plt.get_fignums() == []:
            plt.figure(figsize=(12, 6))

        # 左侧Y轴: 信号
        ax1 = plt.gca()
        ax1.plot(data, 'kx', label='data', markersize=3, alpha=0.5)
        ax1.plot(signal, '-', color=[0.5, 0.5, 0.5], label='signal', linewidth=1.5)

        # 限制显示范围 (最多显示1.5个周期或100个点)
        xlim_max = min(max(int(1.5 / re_fin), 100), N)
        ax1.set_xlim([0, xlim_max])

        data_min, data_max = np.min(data), np.max(data)
        ax1.set_ylim([data_min * 1.1, data_max * 1.1])
        ax1.set_ylabel('Signal', color='k')
        ax1.tick_params(axis='y', labelcolor='k')

        # 右侧Y轴: 误差
        ax2 = ax1.twinx()
        ax2.plot(dep, 'r-', label='dependent err', linewidth=1.5)
        ax2.plot(indep, 'b-', label='independent err', linewidth=1)

        error_min, error_max = np.min(error), np.max(error)
        ax2.set_ylim([error_min * 1.1, error_max * 1.1])
        ax2.set_ylabel('Error', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.set_xlabel('Samples')
        ax1.set_title(f'Thompson Decomposition (freq={re_fin:.6f}, order={order})')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return signal, error, indep, dep, phi


if __name__ == "__main__":
    print("=" * 70)
    print("tomDecomp.py - Thompson Decomposition 测试")
    print("=" * 70)

    # 测试用例: 生成含有谐波失真和噪声的信号
    N = 4096
    fs = 1e6
    fin = 28320.3125  # 相干采样频率
    re_fin = fin / fs

    t = np.arange(N) / fs

    # 理想正弦波
    signal_ideal = np.sin(2 * np.pi * fin * t) * 1000 + 2048

    # 添加谐波失真 (3次谐波, 5次谐波)
    harmonic_3rd = 0.05 * np.sin(3 * 2 * np.pi * fin * t) * 1000
    harmonic_5th = 0.02 * np.sin(5 * 2 * np.pi * fin * t) * 1000

    # 添加随机噪声
    noise = 10 * np.random.randn(N)

    # 合成ADC输出
    adc_output = signal_ideal + harmonic_3rd + harmonic_5th + noise

    print(f"\n测试参数:")
    print(f"  采样点数: {N}")
    print(f"  采样频率: {fs/1e6:.2f} MHz")
    print(f"  输入频率: {fin/1e3:.2f} kHz")
    print(f"  归一化频率: {re_fin:.10f}")
    print(f"  3次谐波幅度: 5%")
    print(f"  5次谐波幅度: 2%")
    print(f"  噪声RMS: 10 LSB")

    # 执行Thompson分解
    print(f"\n执行Thompson分解...")
    signal, error, indep, dep, phi = tomDecomp(adc_output, re_fin=re_fin, order=10, disp=1)

    # 分析结果
    print(f"\n分解结果:")
    print(f"  信号RMS: {np.sqrt(np.mean(signal**2)):.2f}")
    print(f"  总误差RMS: {np.sqrt(np.mean(error**2)):.2f}")
    print(f"  依赖误差RMS: {np.sqrt(np.mean(dep**2)):.2f}")
    print(f"  独立误差RMS: {np.sqrt(np.mean(indep**2)):.2f}")
    print(f"  基波相位: {np.rad2deg(phi):.2f}°")

    # 理论验证
    theoretical_dep_rms = np.sqrt(np.mean((harmonic_3rd + harmonic_5th)**2))
    theoretical_indep_rms = np.sqrt(np.mean(noise**2))

    print(f"\n理论对比:")
    print(f"  理论依赖误差RMS: {theoretical_dep_rms:.2f}")
    print(f"  实际依赖误差RMS: {np.sqrt(np.mean(dep**2)):.2f}")
    print(f"  误差: {abs(theoretical_dep_rms - np.sqrt(np.mean(dep**2))):.2f}")
    print(f"")
    print(f"  理论独立误差RMS: {theoretical_indep_rms:.2f}")
    print(f"  实际独立误差RMS: {np.sqrt(np.mean(indep**2)):.2f}")
    print(f"  误差: {abs(theoretical_indep_rms - np.sqrt(np.mean(indep**2))):.2f}")

    print(f"\n✅ Thompson分解测试完成!")
    print("=" * 70)
