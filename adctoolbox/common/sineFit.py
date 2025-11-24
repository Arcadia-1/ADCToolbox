import numpy as np
import pandas as pd

def find_relative_freq(data):
    """
    查找给定信号的相对频率索引（k）。

    该函数通过正弦拟合来确定信号的主频率分量，并返回其相对频率。
    这个返回值'k'代表在给定的数据窗口内信号包含的完整周期数。

    重要提示：
    此函数假设采样频率被归一化为1。因此，返回的频率是相对于
    采样率的归一化值。

    参数：
    - data: 输入的信号数据数组。

    返回：
    - k: 信号的相对频率索引。
    """
    # 使用 sine_fit 函数拟合正弦波并获取频率
    _, relative_freq, _, _, _ = sine_fit(data)
    
    return relative_freq


def sine_fit(data, f0=None, tol=1e-12, rate=0.5):
    """
    将正弦波拟合到输入数据。

    Matches MATLAB sineFit.m exactly.

    参数：
    - data: 要拟合的数据。
    - f0: 估计的频率（可选）。如果没有提供，将从数据中估计。
    - tol: 停止准则的容差（默认值：1e-12，匹配MATLAB）。
    - rate: 频率调整步长（默认值：0.5）。

    返回：
    - data_fit: 拟合的正弦波。
    - freq: 估计的频率。
    - mag: 正弦波的幅度。
    - dc: 直流分量。
    - phi: 正弦波的相位。
    """
    
    # 确保数据是列向量
    data = np.mean(data, axis=1) if data.ndim > 1 else data
    N = len(data)
    
    # 如果没有提供频率 f0，则估计频率
    if f0 is None:
        spec = np.abs(np.fft.fft(data))
        spec[0] = 0  # 将直流分量设为0
        spec = spec[:N // 2]
        
        k0 = np.argmax(spec)
        r = 1 if spec[k0 + 1] > spec[k0 - 1] else -1
        f0 = (k0 + r * spec[k0 + r] / (spec[k0] + spec[k0 + r])) / N

    # 时间轴
    time = np.arange(N)
    
    # 初始参数（A，B，DC）
    theta = 2 * np.pi * f0 * time
    M = np.column_stack([np.cos(theta), np.sin(theta), np.ones(N)])
    x = np.linalg.lstsq(M, data, rcond=None)[0]
    
    A, B, dc = x[0], x[1], x[2]
    freq = f0
    delta_f = 0
    
    # 频率的迭代优化 (MATLAB lines 49-67)
    for _ in range(100):
        freq += delta_f
        theta = 2 * np.pi * freq * time
        # MATLAB line 53: 4th column is divided by N
        M = np.column_stack([np.cos(theta), np.sin(theta), np.ones(N),
                             (-A * 2 * np.pi * time * np.sin(theta) + B * 2 * np.pi * time * np.cos(theta)) / N])
        x = np.linalg.lstsq(M, data, rcond=None)[0]

        A, B, dc = x[0], x[1], x[2]
        # MATLAB line 58: delta_f = x(4)*rate/N
        delta_f = x[3] * rate / N

        # MATLAB line 59: relerr = rms(x(end)/N*M(:,end)) / sqrt(x(1)^2+x(2)^2)
        # M(:,end) is the 4th column of M
        mag_current = np.sqrt(A**2 + B**2)
        if mag_current > 0:
            # Calculate RMS of (x(4)/N * M(:,4))
            residual = x[3] / N * M[:, 3]
            relerr = np.sqrt(np.mean(residual**2)) / mag_current
        else:
            relerr = 0

        # 检查停止准则 (MATLAB line 63)
        if relerr < tol:
            break
    
    # 使用最终参数拟合数据
    data_fit = A * np.cos(2 * np.pi * freq * time) + B * np.sin(2 * np.pi * freq * time) + dc
    mag = np.sqrt(A**2 + B**2)
    phi = -np.arctan2(B, A)  # CRITICAL: Negative sign to match MATLAB (line 71)
    
    return data_fit, freq, mag, dc, phi


# 主函数，加载数据并寻找频率
def main():
    # 加载 CSV 文件
    df = pd.read_csv('sine_wave_with_harmonics.csv')  # 请确保文件路径正确
    data = df['Signal'].values
    
    # 调用 find_relative_freq 函数找出频率的索引
    relative_freq = find_relative_freq(data)
    
    # 获取信号的长度
    N = len(data)
    print(f"[信号长度 N 为: {N}]")
    
    # 计算实际的频率索引 k
    k = round(relative_freq * N)
    print(f"[信号频率 k 为: {k}]")
    
# 调用主函数
if __name__ == '__main__':
    main()
