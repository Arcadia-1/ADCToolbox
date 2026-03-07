> **[English Version (英文版)](README.md)** | 由AI翻译，如内容有冲突，以英文版 README 为准。

# ADCToolbox - MATLAB

一个全面的 MATLAB 工具箱，用于 ADC（模数转换器）测试、表征和调试。该工具箱提供了频谱分析、校准、线性度测试、信号处理和 ADC 性能评估等高级功能。

## 目录

- [安装](#安装)
- [快速入门](#快速入门)
- [功能目录](#功能目录)
  - [频谱分析](#频谱分析)
  - [信号拟合与频率分析](#信号拟合与频率分析)
  - [校准功能](#校准功能)
  - [线性度分析](#线性度分析)
  - [噪声传递函数分析](#噪声传递函数分析)
  - [快捷功能](#快捷功能)
  - [实用工具](#实用工具)
- [详细函数参考](#详细函数参考)
- [使用示例](#使用示例)
- [旧版函数](#旧版函数)
- [环境要求](#环境要求)
- [贡献指南](#贡献指南)

## 安装

### 方式一：安装工具箱包（推荐）

1. 进入 `toolbox/` 目录
2. 双击 `ADCToolbox_1v30.mltbx` 进行安装
3. 工具箱将自动添加到 MATLAB 路径中
4. 也可从 [MATLAB Add-Ons](https://www.mathworks.com/matlabcentral/fileexchange/181879-adctoolbox) 下载此工具箱

### 方式二：手动添加路径

```matlab
% 将工具箱添加到 MATLAB 路径
addpath(genpath('path/to/ADCToolbox/matlab/src'))
savepath  % 可选：保存路径以供后续使用
```

### 方式三：使用安装脚本

```matlab
% 运行安装脚本（从 matlab/ 目录执行）
run('setupLib.m')
```

## 快速入门

```matlab
% 加载 ADC 输出数据
load('adc_data.mat');  % 假设包含变量 'sig'

% 方式一：使用综合分析面板（推荐首次分析时使用）
rep = adcpanel(sig, 'fs', 100e6);  % 一站式分析，包含所有指标

% 方式二：调用单独函数进行特定分析，例如：
% 执行综合频谱分析
[enob, sndr, sfdr, snr, thd] = plotspec(sig, 'Fs', 100e6);

% 查找主频
freq = findfreq(sig, 100e6);

% 对数据进行正弦拟合
[fitout, freq, mag, dc, phi] = sinfit(sig);

% 使用直方图法计算 INL 和 DNL
[inl, dnl, code] = inlsin(sig);

% 分析相位谱
plotphase(sig);
```

## 功能目录

### 综合分析面板

整合多种测试方法的统一分析功能。

- **[`adcpanel`](#adcpanel)** - 综合 ADC 分析面板，自动处理数据格式

### 频谱分析

用于分析 ADC 输出数据频域特性的函数。

- **[`plotspec`](#plotspec)** - 综合频谱分析，计算 ENOB、SNDR、SFDR、SNR 和 THD
- **[`plotphase`](#plotphase)** - 相干相位谱分析，含极坐标显示

### 信号拟合与频率分析

用于提取信号参数和频率信息的函数。

- **[`sinfit`](#sinfit)** - 四参数迭代正弦拟合（幅度、相位、直流偏移、频率）
- **[`findfreq`](#findfreq)** - 使用正弦拟合查找主频
- **[`findbin`](#findbin)** - 查找给定信号频率的相干 FFT bin
- **[`tomdec`](#tomdec)** - Thompson 分解：将信号分解为正弦波、谐波和误差

### 校准功能

用于 ADC 比特权重校准和误差校正的函数。

- **[`wcalsin`](#wcalsin)** - 使用正弦波输入进行权重校准（支持单组或多组数据）
- **[`cdacwgt`](#cdacwgt)** - 计算多段电容 DAC 的比特权重
- **[`plotwgt`](#plotwgt)** - 可视化比特权重并标注 radix，计算最优缩放因子和有效分辨率
- **[`plotres`](#plotres)** - 绘制 ADC 比特矩阵的部分和残差散点图

### 线性度与误差分析

用于分析 ADC 线性度性能的函数。

- **[`inlsin`](#inlsin)** - 使用直方图法从正弦波数据计算 INL 和 DNL
- **[`errsin`](#errsin)** - 带直方图分箱的正弦拟合误差分析

### 噪声传递函数分析

用于分析噪声整形 ADC（Delta-Sigma 调制器）的函数。

- **[`ntfperf`](#ntfperf)** - 分析噪声传递函数性能和 SNR 改善

### 快捷功能

将多个步骤合并为一次调用的便捷封装函数。

- **[`plotressin`](#plotressin)** - 直接从比特矩阵绘制部分和残差（自动通过 `wcalsin` 校准）
- **[`errsinv`](#errsinv)** - `errsin` 的快捷方式，默认使用值模式分箱（`xaxis='value'`）

### 实用工具

用于信号处理和分析的辅助函数。

- **[`alias`](#alias)** - 计算采样后的混叠频率
- **[`ifilter`](#ifilter)** - 基于 FFT 的理想滤波器，保留指定频带
- **[`bitchk`](#bitchk)** - 通过分析比特段残差分布检查 ADC 溢出

## 详细函数参考

### adcpanel

**功能：** 综合 ADC 分析面板，自动检测输入数据类型并运行相应的分析流程。

**语法：**
```matlab
rep = adcpanel(dat)
rep = adcpanel(dat, 'Name', Value)
```

**主要特性：**
- 在单一面板中显示多项 ADC 分析结果的统一界面
- 自动数据类型检测（值波形 vs. 逐位数据）
- 三条分析流水线：
  - **流水线 A**：值波形 + 正弦波（完整表征）
  - **流水线 B**：值波形 + 其他信号（基础分析）
  - **流水线 C**：逐位数据（校准 + 完整表征）
- 生成有序的 tiledlayout 图形（正弦波分析为 3×4 网格）
- 返回包含所有指标和图形句柄的综合报告结构体

**分析流水线：**

**流水线 A - 值波形 + 正弦波：**
1. `plotspec` - 频谱分析（ENOB、SNDR、SFDR、SNR、THD）
2. `tomdec` - Thompson 分解，获取时域误差波形
3. 时域图 - 信号和误差波形（缩放至最大误差区域）
4. `errsin` - 正弦误差分析（相位和值两种模式）
5. `inlsin` - INL/DNL 计算
6. `perfosr` - 性能随 OSR 变化的扫描
7. `plotphase` - 谐波相位分析（FFT 和 LMS 两种模式）

**流水线 B - 值波形 + 其他信号：**
1. 时域波形图
2. `plotspec` - 基础频谱显示

**流水线 C - 逐位数据：**
1. `bitchk` - 溢出/下溢检测
2. `wcalsin` - 正弦波权重校准
3. `plotwgt` - 可视化校准权重，计算最优缩放因子和有效分辨率
4. 从权重比自动检测 `maxCode`（识别有效位与噪声位）
5. 若校准成功：对校准值运行流水线 A

**参数：**
- `'dataType'` - 数据解释方式：'auto'（默认）、'values' 或 'bits'
- `'signalType'` - 输入信号类型：'sinewave'（默认）或 'other'
- `'OSR'` - 过采样比（默认：1）
- `'fs'` - 采样频率，单位 Hz（默认：1）
- `'maxCode'` - 满量程范围（默认：自动检测；逐位数据使用权重比分析识别有效位）
- `'harmonic'` - 分析的谐波数（默认：5）
- `'window'` - 窗函数：'hann'（默认）、'rect' 或函数句柄
- `'fin'` - 归一化输入频率（默认：0，自动检测）
- `'disp'` - 启用图形显示（默认：true）
- `'verbose'` - 启用详细输出（默认：false）

**输出：**
- `rep` - 报告结构体，包含：
  - `.dataType` - 'values' 或 'bits'
  - `.signalType` - 'sinewave' 或 'other'
  - `.spectrum` - 频谱指标（ENOB、SNDR、SFDR、SNR、THD 等）
  - `.decomp` - Thompson 分解结果（sine、err、har、oth、freq）
  - `.errorPhase` - 相位分箱误差分析（emean、erms、anoi、pnoi）
  - `.errorValue` - 值分箱误差分析（emean、erms）
  - `.linearity` - INL/DNL 结果（inl、dnl、code）
  - `.osr` - OSR 扫描结果（osr、sndr、sfdr、enob）
  - `.phaseFFT` - FFT 模式相位分析
  - `.phaseLMS` - LMS 模式相位分析（含噪声圆）
  - `.bits` - 逐位分析（weights、offset、overflow、effres），仅逐位数据时有效
  - `.figures` - 所有生成图形和坐标轴的句柄

**示例：**
```matlab
% 值波形数据的基本用法
sig = sin(2*pi*0.123*(0:4095)') + 0.01*randn(4096,1);
rep = adcpanel(sig);

% 逐位数据分析（将 sig 量化为比特）
bits = dec2bin(round((sig/2.1+0.5)*2^12), 12) - '0';  % 转换为 12 位表示
rep = adcpanel(bits);

% 带特定参数的过采样数据
rep = adcpanel(sig, 'OSR', 4, 'fs', 100e6, 'harmonic', 7);

% 非正弦信号（仅时域 + 频谱）
noise_sig = randn(4096, 1);
rep = adcpanel(noise_sig, 'signalType', 'other');

% 从报告中获取特定结果
fprintf('ENOB: %.2f bits\n', rep.spectrum.enob);
fprintf('SNDR: %.2f dB\n', rep.spectrum.sndr);
fprintf('Max INL: %.3f LSB\n', max(abs(rep.linearity.inl)));
```

**注意事项：**
- INL/DNL 分析需要整数码值；非整数数据会自动取整
- 当 N < maxCode 时会发出警告，因为 INL/DNL 可能不可靠
- 逐位数据会生成单独的图形显示 bitchk 和 plotwgt 结果
- 时域显示会缩放至最大误差点附近约 3 个正弦周期

**另见：** plotspec, tomdec, errsin, inlsin, perfosr, plotphase, bitchk, wcalsin, plotwgt

---

### plotspec

**功能：** 综合功率谱分析和 ADC 性能指标计算。

**语法：**
```matlab
[enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(sig)
[enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(sig, Fs, maxCode, harmonic)
[enob, sndr, sfdr, snr, thd, sigpwr, noi, nsd, h] = plotspec(sig, 'Name', Value)
```

**主要特性：**
- 计算 ENOB（有效位数）
- 计算 SNDR（信号与噪声失真比）
- 测量 SFDR（无杂散动态范围）
- 计算 SNR（信噪比）和 THD（总谐波失真）
- 支持噪声整形 ADC 的过采样比（OSR）
- 多种平均模式：普通（功率平均）和相干（相位对齐）
- 灵活的窗函数：内置 Hanning/矩形窗或自定义窗函数
- 噪底估计模式：auto（多种方法的中值）、median、trimmed mean 或 exclude harmonics
- 可配置信号带宽和闪烁噪声消除

**参数：**
- `'OSR'` - 过采样比（默认：1）
- `'window'` - 窗函数：'hann'、'rect' 或函数句柄（默认：'hann'）
- `'averageMode'` - 'normal' 或 'coherent' 平均（默认：'normal'）
- `'NFMethod'` - 噪底估计：'auto'、'median'、'mean' 或 'exclude'（默认：'auto'）；数值：0=auto，1=median，2=mean，3=exclude
- `'sideBin'` - 信号峰值两侧的额外 bin 数（默认：'auto'）
- `'cutoff'` - 闪烁噪声消除的高通截止频率（默认：0）
- `'label'` - 启用图形标注（默认：true）
- `'disp'` - 启用绘图（默认：true）
- `'dispItem'` - 显示项目选择器（默认：'sfedutrlyhop'，所有项目）
  - 字符串中每个字符（不区分大小写）启用一个特定标注：
  - `'s'` - 信号功率文本和信号 bin 标记
  - `'f'` - 输入频率和采样频率（Fin/Fs）
  - `'e'` - 有效位数（ENOB）
  - `'d'` - 信号与噪声失真比（SNDR）
  - `'u'` - 无杂散动态范围（SFDR）
  - `'t'` - 总谐波失真（THD）
  - `'r'` - 信噪比（SNR）
  - `'l'` - 噪底电平
  - `'y'` - 噪声谱密度（NSD）及水平虚线
  - `'o'` - 过采样比（OSR）及垂直带宽线
  - `'h'` - 谐波标记
  - `'p'` - 最大杂散标记
  - 示例：`'sfe'` 仅显示信号、频率和 ENOB 标注

**示例：**
```matlab
% 32 倍过采样和相干平均的基本用法
[enob, sndr, sfdr] = plotspec(sig, 100e6, 2^16, 'OSR', 32, 'averageMode', 'coherent');

% 使用自定义窗函数的多次测量
sig_multi = randn(10, 1024);  % 10 次运行，每次 1024 个采样
[enob, sndr] = plotspec(sig_multi, 'window', @blackman);

% 自定义绘图标注 - 仅显示关键指标
[enob, sndr] = plotspec(sig, 'dispItem', 'fedu');  % 仅显示 Fin/Fs、ENOB、SNDR、SFDR
```

### plotphase

**功能：** 可视化相干相位谱，在极坐标系中显示谐波。

**语法：**
```matlab
h = plotphase(sig)
h = plotphase(sig, harmonic, maxSignal)
h = plotphase(sig, 'Name', Value)
```

**主要特性：**
- 两种分析模式：基于 FFT 和基于 LMS（默认）
- **LMS 模式**：使用最小二乘谐波拟合，显示噪声圆作为参考
- **FFT 模式**：传统相干平均加相位对齐
- 极坐标图显示幅度（半径）和相位（角度）
- 基波用红色表示，谐波用蓝色表示
- 支持过采样和闪烁噪声消除

**参数：**
- `'mode'` - 分析模式：'LMS'（默认）或 'FFT'
- `'OSR'` - 过采样比（默认：1）
- `'Fs'` - 采样频率（默认：1）
- `'cutoff'` - 高通截止频率（默认：0）

**示例：**
```matlab
% LMS 模式显示噪声圆
plotphase(sig, 7, 2^16, 'mode', 'LMS');

% FFT 模式加过采样
plotphase(sig, 10, 'mode', 'FFT', 'OSR', 64);
```

### sinfit

**功能：** 四参数迭代正弦拟合，提取幅度、相位、直流偏移和频率。

**语法：**
```matlab
[fitout, freq, mag, dc, phi] = sinfit(sig)
[fitout, freq, mag, dc, phi] = sinfit(sig, f0, tol, rate, fsearch, verbose, niter)
[fitout, freq, mag, dc, phi] = sinfit(sig, 'Name', Value, ...)
```

**主要特性：**
- 带频率梯度下降的迭代最小二乘优化
- 使用 FFT 抛物线插值的自动频率估计
- 可配置的收敛容差和学习率
- 可选的精细频率搜索控制（`fsearch`）
- 用于调试迭代过程的详细输出
- 支持单通道和多通道（平均）输入

**算法：**
1. 使用线性最小二乘进行初始三参数拟合（余弦、正弦、直流）
2. 通过计算频率梯度进行迭代频率精化（当 `fsearch=1` 时）
3. 当相对误差 < 容差（默认：1e-12）时收敛
4. 若达到最大迭代次数仍未满足容差，发出收敛警告

**参数（位置参数或名值对）：**
- `f0` - 初始频率估计（归一化，默认：0，自动检测）
- `tol` - 收敛容差（默认：1e-12）
- `rate` - 频率更新学习率（默认：0.5）
- `fsearch` - 强制精细频率搜索迭代（默认：0，当 f0=0 时自动启用）
- `verbose` - 启用迭代过程详细输出（默认：0）
- `niter` - 频率精化最大迭代次数（默认：100）

**示例：**
```matlab
% 自动频率估计（自动启用 fsearch）
[fitout, freq, mag, dc, phi] = sinfit(sig);

% 已知频率加自定义容差（位置参数）
[fitout, freq] = sinfit(sig, 0.123, 1e-10, 0.7);

% 已知频率的三参数拟合（不迭代）
[fitout, freq] = sinfit(sig, 0.123);

% 强制迭代并启用详细输出（名值对）
[fitout, freq] = sinfit(sig, 'f0', 0.123, 'fsearch', 1, 'verbose', 1);

% 自定义迭代上限和容差
[fitout, freq] = sinfit(sig, 'niter', 200, 'tol', 1e-15);
```

### findfreq

**功能：** 使用正弦拟合查找信号中的主频。

**语法：**
```matlab
freq = findfreq(sig, fs)
```

**主要特性：**
- 内部使用迭代正弦拟合算法（sinfit）
- 返回拟合频率，而非 FFT 峰值频率
- 支持绝对频率（带 fs）和归一化频率（fs=1）

**示例：**
```matlab
% 查找以 10 kHz 采样的 1 kHz 信号的频率
freq = findfreq(sig, 10000);  % 返回 ~1000 Hz

% 获取归一化频率
freq_norm = findfreq(sig);  % 返回 ~0.1
```

### findbin

**功能：** 查找确保相干采样（FFT 窗口内整数周期）的最近 FFT bin。

**语法：**
```matlab
b = findbin(fs, fin, n)
```

**主要特性：**
- 确保 gcd(bin, n) = 1 以实现相干采样
- 防止 FFT 窗口中的重复相位采样
- 从初始估计向上和向下搜索
- 两个 bin 等距时返回较大的 bin

**示例：**
```matlab
% 查找 1 kHz 信号的相干 bin
fs = 10000;  % 10 kHz 采样
n = 1024;    % 1024 点 FFT
fin = 1000;  % 1 kHz 信号
b = findbin(fs, fin, n);  % 返回 103

% 计算实际相干频率
fin_actual = b * fs / n;  % = 1006.8 Hz
```

### wcalsin

**功能：** 使用正弦波输入估计 ADC 逐位权重和直流偏移，用于校准。

**语法：**
```matlab
[weight, offset, postcal, ideal, err, freqcal] = wcalsin(bits)
[weight, offset, postcal, ideal, err, freqcal] = wcalsin(bits, 'Name', Value)
[weight, offset, postcal, ideal, err, freqcal] = wcalsin({bits1, bits2, ...}, 'Name', Value)
```

**主要特性：**
- 支持单组或多组数据联合校准
- 粗搜索和精细迭代的自动频率搜索
- 通过合并相关列处理秩亏比特矩阵
- 可指定阶数的谐波排除
- 自动极性校正
- SNR 检查，校准质量差（< 20 dB）时发出警告

**算法：**
1. 若频率未知：使用多种比特组合进行粗搜索，然后精细迭代搜索
2. 构建基波和谐波的正弦/余弦基函数
3. 求解两种最小二乘公式（基于余弦和基于正弦）
4. 选择残差较小的解
5. 使用梯度下降迭代精化频率（当 fsearch=1 时）
6. 通过识别和合并相关列处理秩亏

**参数：**
- `'freq'` - 归一化输入频率（默认：0，自动搜索）
- `'order'` - 拟合中排除的谐波数（默认：1）
- `'rate'` - 频率迭代的自适应学习率（默认：0.5）
- `'reltol'` - 相对误差容差（默认：1e-12）
- `'niter'` - 精细搜索最大迭代次数（默认：100）
- `'fsearch'` - 强制精细频率搜索（默认：0）
- `'verbose'` - 启用详细输出（默认：0）
- `'nomWeight'` - 用于秩亏处理的标称权重

**示例：**
```matlab
% 自动频率搜索的基本校准
[wgt, off] = wcalsin(bits);

% 已知频率并排除 3 次谐波
[wgt, off, cal, ideal, err, freq] = wcalsin(bits, 'freq', 0.123, 'order', 3);

% 多组数据联合校准
[wgt, off] = wcalsin({bits1, bits2}, 'freq', [0.1, 0.2], 'order', 5);
```

### cdacwgt

**功能：** 计算带桥接电容和寄生参数的多段电容 DAC 的比特权重。

**语法：**
```matlab
[weight, ctot] = cdacwgt(cd, cb, cp)
```

**主要特性：**
- 多段 CDAC 架构建模
- 考虑电容分压效应
- 处理段间桥接电容
- 包含寄生电容
- 从 LSB 到 MSB 处理，确保权重衰减准确

**电路模型：**
```
MSB 侧 <---||------------||---< LSB 侧
             cb   |    |   Cl（前序比特负载）
                 ---  ---
             cp  ---  ---  cd
                  |    |
                 gnd   Vbot
```

**参数：**
- `cd` - DAC 电容 [MSB ... LSB]
- `cb` - 桥接电容 [MSB ... LSB]（0 表示无桥接）
- `cp` - 寄生电容 [MSB ... LSB]

**示例：**
```matlab
% 简单二进制加权 4 位 DAC
cd = [8 4 2 1];
cb = [0 0 0 0];
cp = [0 0 0 0];
[weight, ctot] = cdacwgt(cd, cb, cp);
% 返回：weight = [0.5333 0.2667 0.1333 0.0667], ctot = 15

% 带 3+3 段的 6 位 CDAC
cd = [4 2 1  4 2 1];
cb = [0 4 0  8/7 0 0];  % 段间桥接
cp = [0 0 0  0 0 1];
[weight, ctot] = cdacwgt(cd, cb, cp);
% 返回：weight = [0.5000 0.2500 0.1250 0.0625 0.0312 0.0156]
```

### plotwgt

**功能：** 可视化绝对比特权重并标注 radix，识别 ADC 架构和检测校准误差。同时计算最优权重缩放因子和有效分辨率。

**语法：**
```matlab
radix = plotwgt(weights)
radix = plotwgt(weights, disp)
[radix, wgtsca] = plotwgt(weights)
[radix, wgtsca, effres] = plotwgt(weights)
```

**主要特性：**
- 在对数 Y 轴上绘制绝对比特权重
- 标注相邻比特间的 radix（缩放因子）
- 负权重以红色显示，表示符号错误
- MSB（最大权重）标记为红色，LSB（最小有效权重）标记为绿色
- 双 X 轴标签：升序数组索引（底部）和降序显著性顺序（顶部）
- 在图中显示有效分辨率
- 计算最优权重缩放因子（`wgtsca`），最小化取整误差
- 从有效比特权重估计有效分辨率（`effres`）
- 可选 `disp` 参数禁用绘图

**wgtsca 和 effres 算法：**
1. 按绝对权重降序排列以确定比特显著性
2. 通过检测比值跳变 >= 3 找到"有效"比特（大跳变表示噪声/冗余位）
3. 初始缩放将最小有效权重归一化为 1
4. 精化缩放以最小化有效权重的取整误差
5. 计算 effres = `log2(sum(absW_sig)/absW_LSB + 1)`

**参数：**
- `weights` - 从 MSB 到 LSB 的比特权重，向量 (1 x B)
- `disp` - 显示标志（可选，默认：1）。设为 0 禁用绘图。

**输出：**
- `radix` - 相邻比特间的 radix，向量 (1 x B-1)
  - `radix(i) = |weight(i) / weight(i+1)|`
  - 二进制 ADC：所有比特 radix ≈ 2.00
  - 子基数 ADC：radix < 2.00（如 1.5-bit/stage → ~1.90）
- `wgtsca` - 最优权重缩放因子，归一化权重以最小化取整误差
- `effres` - 有效分辨率（比特），从有效权重比估计

**示例：**
```matlab
% 可视化理想 12 位二进制权重
weights_ideal = 2.^(11:-1:0);
[radix, wgtsca, effres] = plotwgt(weights_ideal);

% 可视化 CDAC 权重（6 位，3+3 段）
cd = [4 2 1 4 2 1];       % 两个 3 位段 [MSB ... LSB]
cb = [0 0 0 8/7 0 0];     % 段间桥接电容
cp = [0 0 0 0 0 1];       % LSB 处寄生电容
weight = cdacwgt(cd, cb, cp);
radix = plotwgt(weight);

% 仅计算缩放因子，不显示图形
[~, wgtsca, effres] = plotwgt(weights, 0);
```

### plotres

**功能：** 绘制 ADC 比特矩阵的部分和残差散点图，揭示比特级之间的相关性、非线性模式和冗余。

**语法：**
```matlab
plotres(sig, bits)
plotres(sig, bits, wgt)
plotres(sig, bits, wgt, xy)
plotres(sig, bits, wgt, xy, alpha)
plotres(sig, bits, 'Name', Value)
```

**主要特性：**
- 不同比特级残差的瓦片散点图
- 第 k 级残差 = sig - bits(:,1:k) * wgt(1:k)'
- 支持自动或手动 alpha 控制的半透明标记
- 支持自定义比特权重和任意比特对选择

**参数：**
- `sig` - 理想输入信号（N x 1 或 1 x N）
- `bits` - 原始 ADC 比特矩阵（N x M），MSB 在前
- `wgt` - 比特权重（可选，默认：二进制权重 `[2^(M-1), ..., 1]`）
- `xy` - 要绘制的比特对索引（可选，默认：`[(0:(M-1))', ones(M,1)*M]`）
- `alpha` - 标记透明度（可选，默认：`'auto'`）
  - `'auto'`：缩放为 `clamp(1000/N, 0.1, 1)`
  - 数值标量 (0, 1]：固定透明度

**示例：**
```matlab
% 使用二进制权重的基本残差图
N = 1024; M = 6;
sig = (sin(2*pi*(0:N-1)'/N * 3)/2 + 0.5) * (2^M - 1);
code = round(sig);
bits = dec2bin(code, M) - '0';
plotres(sig, bits);

% 指定比特对和自定义透明度
plotres(sig, bits, 2.^(M-1:-1:0), [2 4; 4 6], 0.3);
```

**另见：** bitchk, plotwgt, plotressin

---

### plotressin

**功能：** 便捷封装函数，通过 `wcalsin` 校准比特权重后将结果转发给 `plotres`，省去手动校准步骤。

**语法：**
```matlab
plotressin(bits)
plotressin(bits, xy)
plotressin(bits, ..., 'Name', Value)
```

**主要特性：**
- 内部调用 `wcalsin` 恢复校准权重和理想信号
- 将重建的参考信号（`ideal + offset`）和权重转发给 `plotres`
- 接受与 `plotres` 相同的 `xy` 格式
- 将 `freq`、`order`、`verbose` 参数转发给 `wcalsin`
- 将 `alpha` 参数转发给 `plotres`

**参数：**
- `bits` - 原始 ADC 比特矩阵（N x M），MSB 在前
- `xy` - 要绘制的比特对索引（可选，与 `plotres` 格式相同）
- `'freq'` - `wcalsin` 的归一化输入频率（默认：0，自动搜索）
- `'order'` - 拟合模型中的谐波数（默认：1）
- `'verbose'` - 详细输出标志（默认：0）
- `'alpha'` - 标记透明度，转发给 `plotres`（默认：`'auto'`）

**示例：**
```matlab
% 基本用法（自动频率搜索和校准）
N = 1024; M = 6;
sig = (sin(2*pi*(0:N-1)'/N * 3)/2 + 0.5) * (2^M - 1);
code = round(sig);
bits = dec2bin(code, M) - '0';
plotressin(bits)

% 指定比特对和已知频率
plotressin(bits, [0 6; 3 6], 'freq', 3/1024)

% 转发校准参数
plotressin(bits, 'order', 3)
```

**另见：** plotres, wcalsin, plotwgt

---

### errsinv

**功能：** `errsin` 的快捷方式，默认使用值模式分箱（`xaxis='value'`），无需指定 `xaxis` 参数即可快速进行类 INL 误差可视化。

**语法：**
```matlab
[emean, erms, xx, anoi, pnoi, err, errxx] = errsinv(sig)
[emean, erms, xx, anoi, pnoi, err, errxx] = errsinv(sig, 'Name', Value)
```

**主要特性：**
- 调用 `errsin` 并预设 `'xaxis', 'value'`
- 无输出参数时自动启用显示（`'disp', 1`）
- 其他所有 `errsin` 名值参数原样传递

**参数：**
- `sig` - 输入信号（与 `errsin` 相同）
- `errsin` 接受的所有名值参数（如 `'bin'`、`'fin'`、`'erange'`、`'disp'`）

**输出：**
- 与 `errsin` 相同：`emean`、`erms`、`xx`、`anoi`、`pnoi`、`err`、`errxx`

**示例：**
```matlab
% 快速值模式误差图（自动显示）
sig = sin(2*pi*0.12345*(0:999)') + 0.01*randn(1000,1);
errsinv(sig)

% 自定义分箱数
[emean, erms, xx] = errsinv(sig, 'bin', 50);
```

**另见：** errsin

---

### inlsin

**功能：** 使用正弦波直方图法计算 ADC 的 INL（积分非线性）和 DNL（微分非线性）。

**语法：**
```matlab
[inl, dnl, code] = inlsin(data)
[inl, dnl, code] = inlsin(data, excl, disp)
[inl, dnl, code] = inlsin(data, 'name', value)
```

**主要特性：**
- 基于直方图的方法，假设理想正弦波输入
- 自动端点排除以避免削波噪声
- 检测并突出显示缺失码（DNL ≤ -1）
- 零均值归一化 DNL 输出
- INL 由 DNL 累积求和计算

**算法：**
1. 对 ADC 输出码进行直方图统计
2. 应用余弦变换对正弦分布进行线性化
3. 从线性化直方图的差值计算 DNL
4. 归一化到 LSB = 1 并去除直流偏移
5. INL 由 DNL 累积求和得到

**参数：**
- `excl` - 端点排除比例（默认：0.01 = 1%）
- `disp` - 显示图形（默认：无输出时自动显示）

**示例：**
```matlab
% 生成 8 位 ADC 输出并分析
t = linspace(0, 2*pi, 10000);
data = round(127.5 + 127.5*sin(t));
[inl, dnl, code] = inlsin(data);

% 自定义排除比例
[inl, dnl, code] = inlsin(data, 0.05);  % 两端各排除 5%
```

### errsin

**功能：** 带直方图分箱的正弦拟合误差分析，用于识别幅度和相位噪声。

**语法：**
```matlab
[emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig)
[emean, erms, xx, anoi, pnoi, err, errxx] = errsin(sig, 'Name', Value)
```

**主要特性：**
- 两种分箱模式：相位（默认）或值
- **相位模式**：估计幅度和相位噪声分量
- **值模式**：适用于 INL 分析
- 可配置分箱数和误差范围过滤
- 噪声源的最小二乘分解

**噪声估计（相位模式）：**
- 幅度噪声等幅影响所有相位（cos² 模式）
- 相位噪声产生与斜率成正比的误差（sin² 模式）
- 拟合：`erms² = anoi²·cos²(θ) + pnoi²·sin²(θ)`

**参数：**
- `'bin'` - 直方图分箱数（默认：100）
- `'fin'` - 归一化输入频率（默认：0，自动检测）
- `'xaxis'` - 分箱模式：'phase'（默认）或 'value'
- `'erange'` - 误差范围过滤 [min, max]（默认：[]）
- `'disp'` - 显示图形（默认：无输出时自动显示）

**示例：**
```matlab
% 相位模式分析及噪声估计
sig = sin(2*pi*0.12345*(0:999)') + 0.01*randn(1000,1);
[emean, erms, xx, anoi, pnoi] = errsin(sig);

% 值模式用于 INL 可视化
[emean, erms, xx] = errsin(sig, 'xaxis', 'value', 'bin', 50);

% 过滤到特定相位范围
[~, ~, ~, ~, ~, err, phase] = errsin(sig, 'erange', [90, 180]);
```

### tomdec

**功能：** 将单频信号进行 Thompson 分解，分离为基波正弦、谐波失真和其他误差。

**语法：**
```matlab
[sine, err, har, oth, freq] = tomdec(sig)
[sine, err, har, oth, freq] = tomdec(sig, freq, order, disp)
[sine, err, har, oth, freq] = tomdec(sig, 'name', value)
```

**主要特性：**
- 最小二乘拟合分离信号分量
- 未指定时自动检测频率
- 分解为：基波 + 谐波 + 残差
- 可配置谐波阶数（默认：10）

**分解关系：**
- `sig = sine + err`
- `sine` = 直流 + 仅基波
- `err = har + oth`
- `har` = 2 次到第 n 次谐波
- `oth` = 所有剩余误差（噪声、非谐波失真）

**参数：**
- `freq` - 归一化信号频率（默认：自动检测）
- `order` - 拟合的谐波数（默认：10）
- `disp` - 显示结果（默认：无输出时自动显示）

**示例：**
```matlab
% 自动检测频率并分解
[sine, err, har, oth] = tomdec(sig);

% 已知频率仅拟合 5 次谐波
[sine, err, har, oth] = tomdec(sig, 0.123, 5);
```

### ntfperf

**功能：** 分析噪声整形 ADC（Delta-Sigma 调制器）的噪声传递函数性能。

**语法：**
```matlab
snr = ntfperf(ntf, fl, fh)
snr = ntfperf(ntf, fl, fh, disp)
```

**主要特性：**
- 评估噪声整形和过采样带来的 SNR 改善
- 支持低通和带通 NTF 分析
- 高分辨率频率评估（100 万点）
- 自动生成带信号频带标记的图形

**参数：**
- `ntf` - 噪声传递函数（tf、zpk 或 ss 对象）
- `fl` - 低频边界 [0, 0.5]
- `fh` - 高频边界 (fl, 0.5]
- `disp` - 显示图形（默认：0）

**示例：**
```matlab
% 1 阶低通 Delta-Sigma，16 倍 OSR
ntf = tf([1 -1], [1 0], 1);  % NTF = 1 - z^-1
snr = ntfperf(ntf, 0, 0.5/16);  % 返回约 31 dB 改善

% 带通 NTF 并可视化
ntf = tf([1 0 1], [1 0 0], 1);  % NTF = 1 + z^-2
snr = ntfperf(ntf, 0.24, 0.26, 1);
```

### alias

**功能：** 计算采样后的混叠频率，考虑奈奎斯特区效应。

**语法：**
```matlab
fal = alias(fin, fs)
```

**主要特性：**
- 处理任意奈奎斯特区的信号
- 偶数区（0,2,4,...）：正常混叠
- 奇数区（1,3,5,...）：镜像混叠
- 支持标量或向量输入

**示例：**
```matlab
% 0.7*fs 的信号混叠到 0.3*fs（镜像）
fal = alias(70, 100);  % 返回 30

% 1.3*fs 的信号混叠到 0.3*fs（正常）
fal = alias(130, 100);  % 返回 30

% 多个频率
fal = alias([30 70 130], 100);  % 返回 [30 30 30]
```

### ifilter

**功能：** 基于 FFT 的理想砖墙滤波器，提取指定频带。

**语法：**
```matlab
sigout = ifilter(sigin, passband)
```

**主要特性：**
- 基于 FFT 的滤波，过渡带陡峭
- 支持多个通带（频带并集）
- 独立滤波每一列
- 保持厄米对称性以确保实数输出

**参数：**
- `sigin` - 输入信号矩阵（每列独立滤波）
- `passband` - 频带 [fLow, fHigh] 按行排列（归一化到 [0, 0.5]）

**示例：**
```matlab
% 从 0.1*Fs 到 0.2*Fs 的单通带
sigout = ifilter(sigin, [0.1, 0.2]);

% 多通带
sigout = ifilter(sigin, [0.05, 0.15; 0.25, 0.35]);
```

### bitchk

**功能：** 通过分析比特段残差分布检查 ADC 溢出。

**语法：**
```matlab
bitchk(bits)
bitchk(bits, wgt, chkpos)
bitchk(bits, 'name', value)
```

**主要特性：**
- 可视化每个比特的归一化残差分布
- 检测溢出（≥1）和下溢（≤0）条件
- 颜色编码：蓝色（正常）、红色（溢出）、黄色（下溢）
- 显示边界处的样本百分比

**参数：**
- `bits` - 原始 ADC 比特矩阵 [MSB ... LSB]（N×M）
- `wgt` - 比特权重（默认：二进制权重）
- `chkpos` - 检查的比特位置（默认：MSB）

**示例：**
```matlab
% 使用默认二进制权重检查
bits = randi([0 1], 10000, 10);
bitchk(bits);

% 自定义权重和检查位置
wgt = 2.^(9:-1:0);
bitchk(bits, wgt, 8);  % 检查从第 8 位到 LSB 的段
```

## 使用示例

### 示例 1：使用 adcpanel 进行快速综合分析

```matlab
% 加载 ADC 数据
load('adc_capture.mat');  % 包含 'data' 变量

% 一个函数调用完成综合分析
rep = adcpanel(data, 'fs', 100e6);

% 从报告结构体中获取结果
fprintf('=== ADC 性能总结 ===\n');
fprintf('ENOB: %.2f bits\n', rep.spectrum.enob);
fprintf('SNDR: %.2f dB\n', rep.spectrum.sndr);
fprintf('SFDR: %.2f dB\n', rep.spectrum.sfdr);
fprintf('SNR: %.2f dB\n', rep.spectrum.snr);
fprintf('THD: %.2f dB\n', rep.spectrum.thd);
fprintf('Max INL: %.3f LSB\n', max(abs(rep.linearity.inl)));
fprintf('Max DNL: %.3f LSB\n', max(abs(rep.linearity.dnl)));
fprintf('幅度噪声: %.2e\n', rep.errorPhase.anoi);
fprintf('相位噪声: %.2e rad\n', rep.errorPhase.pnoi);
```

### 示例 2：完整 ADC 表征

```matlab
% 加载 ADC 数据（假设 12 位 ADC，100 MHz 采样）
load('adc_capture.mat');  % 包含 'data' 变量

% 1. 频谱分析
[enob, sndr, sfdr, snr, thd] = plotspec(data, 100e6, 2^12, 5);
fprintf('ADC 性能：\n');
fprintf('  ENOB: %.2f bits\n', enob);
fprintf('  SNDR: %.2f dB\n', sndr);
fprintf('  SFDR: %.2f dB\n', sfdr);
fprintf('  SNR: %.2f dB\n', snr);
fprintf('  THD: %.2f dB\n', thd);

% 2. 查找输入频率
freq = findfreq(data, 100e6);
fprintf('  输入频率: %.2f MHz\n', freq/1e6);

% 3. 线性度分析
[inl, dnl, code] = inlsin(data, 0.02);  % 两端各排除 2%
fprintf('  最大 INL: %.3f LSB\n', max(abs(inl)));
fprintf('  最大 DNL: %.3f LSB\n', max(abs(dnl)));
fprintf('  缺失码: %d\n', sum(dnl <= -1));

% 4. 误差分析
[emean, erms, phase, anoi, pnoi] = errsin(data, 'bin', 200);
fprintf('  幅度噪声: %.2e\n', anoi);
fprintf('  相位噪声: %.2e rad\n', pnoi);
```

### 示例 3：过采样 ADC 分析

```matlab
% 分析 16 位 Delta-Sigma ADC，64 倍过采样
OSR = 64;
[enob, sndr, ~, snr] = plotspec(data, 1e6, 2^16, 'OSR', OSR, ...
                                  'window', @blackman, ...
                                  'averageMode', 'coherent');

% 分析噪声传递函数
ntf = tf([1 -1], [1 -0.5], 1);  % 1 阶 NTF
snr_gain = ntfperf(ntf, 0, 0.5/OSR, 1);
fprintf('NTF 提供 %.2f dB SNR 改善\n', snr_gain);
```

### 示例 4：SAR ADC 校准

```matlab
% 加载 SAR ADC 原始比特数据
load('sar_bits.mat');  % 包含 'bits' 矩阵

% 1. 使用正弦波校准权重
[weight, offset, postcal] = wcalsin(bits, 'freq', 0.1234, 'order', 3);

% 2. 检查校准数据的溢出
bitchk(bits, weight);

% 3. 计算理论 CDAC 权重用于比较
% 假设 12 位，6+6 分裂电容阵列
cd = [32 16 8 4 2 1,  32 16 8 4 2 1];
cb = [0 0 0 0 0 32,  0 0 0 0 0 0];
cp = zeros(1, 12);
[weight_ideal, ctot] = cdacwgt(cd, cb, cp);

% 4. 比较校准权重与理想权重
figure;
subplot(2,1,1);
bar(weight);
title('校准权重');
subplot(2,1,2);
bar(weight_ideal);
title('理想权重');

% 5. 分析校准后的性能
[enob_cal, sndr_cal] = plotspec(postcal, 'disp', true);
fprintf('校准后: ENOB = %.2f, SNDR = %.2f dB\n', enob_cal, sndr_cal);
```

### 示例 5：多频测试

```matlab
% 在多个输入频率下测试 ADC
frequencies = [1e6, 5e6, 10e6, 20e6, 40e6];  % 1 到 40 MHz
fs = 100e6;  % 100 MHz 采样
N = 8192;    % FFT 长度

results = struct();
for i = 1:length(frequencies)
    fin = frequencies(i);

    % 查找相干 bin
    bin = findbin(fs, fin, N);
    fin_coherent = bin * fs / N;

    % 生成测试信号（仿真）
    t = (0:N-1)'/fs;
    sig = sin(2*pi*fin_coherent*t) + 0.001*randn(N,1);

    % 测量性能
    [enob, sndr, sfdr] = plotspec(sig, fs, 2, 'disp', false);

    % 存储结果
    results(i).freq = fin;
    results(i).freq_actual = fin_coherent;
    results(i).enob = enob;
    results(i).sndr = sndr;
    results(i).sfdr = sfdr;

    fprintf('频率: %.1f MHz -> ENOB: %.2f, SNDR: %.2f dB\n', ...
            fin/1e6, enob, sndr);
end

% 绘制 ENOB vs. 频率
figure;
plot([results.freq]/1e6, [results.enob], 'o-');
xlabel('输入频率 (MHz)');
ylabel('ENOB (bits)');
title('ADC 性能 vs. 频率');
grid on;
```

### 示例 6：Thompson 分解分析

```matlab
% 将 ADC 输出分解为各分量
[sine, err, har, oth, freq] = tomdec(data, 'order', 10, 'disp', true);

% 分析各分量功率
sig_power = rms(sine)^2;
har_power = rms(har)^2;
oth_power = rms(oth)^2;

fprintf('功率分解：\n');
fprintf('  信号: %.2e (%.1f dB)\n', sig_power, 10*log10(sig_power));
fprintf('  谐波: %.2e (%.1f dB)\n', har_power, 10*log10(har_power));
fprintf('  其他: %.2e (%.1f dB)\n', oth_power, 10*log10(oth_power));
fprintf('  THD: %.2f dB\n', 10*log10(har_power/sig_power));
fprintf('  SNR: %.2f dB\n', 10*log10(sig_power/oth_power));
```

### 示例 7：带通滤波

```matlab
% 提取特定频带用于噪声分析
% 假设 100 MHz ADC，信号在 20 MHz

% 定义多个通带
passbands = [
    0.15, 0.25;   % 信号频带（15-25 MHz）
    0.40, 0.45;   % 高频噪声频带
];

% 将信号滤波到这些频带
sig_filtered = ifilter(data, passbands);

% 分析滤波后的内容
plotspec(sig_filtered, 100e6, 2^12);
```

## 旧版函数

`legacy/` 目录包含旧版函数名，用于向后兼容。这些函数调用更名后的新实现：

| 旧版函数 | 新函数 | 说明 |
|---------|--------|------|
| `specPlot.m` | `plotspec.m` | 频谱分析（旧 camelCase 命名） |
| `specPlotPhase.m` | `plotphase.m` | 相位谱 |
| `findBin.m` | `findbin.m` | 相干 bin 查找 |
| `findFin.m` | `findfreq.m` | 频率查找 |
| `FGCalSine.m` | `wcalsin.m` | 权重校准 |
| `cap2weight.m` | `cdacwgt.m` | CDAC 权重计算 |
| `weightScaling.m` | `plotwgt.m` | 权重可视化 |
| `INLsine.m` | `inlsin.m` | INL/DNL 分析 |
| `errHistSine.m` | `errsin.m` | 误差直方图 |
| `sineFit.m` | `sinfit.m` | 正弦拟合 |
| `tomDecomp.m` | `tomdec.m` | Thompson 分解 |
| `NTFAnalyzer.m` | `ntfperf.m` | NTF 性能分析 |
| `overflowChk.m` | `bitchk.m` | 溢出检查 |
| `bitInBand.m` | N/A | 逐位滤波器（已弃用） |

**注意：** 建议在新代码中使用新函数名。旧版函数仅为兼容性保留。

## 环境要求

### 最低要求
- MATLAB R2016b 或更高版本
- 核心功能无需额外工具箱

### 可选（扩展功能）
- **Signal Processing Toolbox**：仅在使用自定义窗函数时需要（如 `@blackman`、`@kaiser`）
  - 内置 Hanning 和矩形窗不需要此工具箱
- **Control System Toolbox**：`ntfperf` 函数（传递函数分析）需要

## 文件结构

```
matlab/
├── README.md                 # 英文文档
├── README.zh-CN.md           # 本文件（中文文档）
├── setupLib.m               # 路径添加脚本
├── src/                     # 源代码目录
│   ├── adcpanel.m          # 综合 ADC 分析面板
│   ├── plotspec.m          # 频谱分析
│   ├── plotphase.m         # 相位谱分析
│   ├── plotwgt.m           # 权重可视化
│   ├── sinfit.m            # 正弦拟合
│   ├── findfreq.m          # 频率查找
│   ├── findbin.m           # 相干 bin 查找
│   ├── wcalsin.m          # 权重校准
│   ├── cdacwgt.m           # CDAC 权重计算
│   ├── inlsin.m            # INL/DNL 分析
│   ├── errsin.m            # 误差直方图分析
│   ├── tomdec.m            # Thompson 分解
│   ├── perfosr.m           # 性能 vs OSR 扫描
│   ├── ntfperf.m           # NTF 性能分析
│   ├── alias.m             # 混叠计算
│   ├── ifilter.m           # 理想滤波器
│   ├── plotres.m           # 部分和残差散点图
│   ├── bitchk.m            # 溢出检查
│   ├── shortcut/           # 便捷封装函数
│   │   ├── plotressin.m    # plotres + wcalsin 一步调用
│   │   └── errsinv.m       # errsin 值模式快捷方式
│   ├── legacy/             # 旧版函数名（兼容用）
│   │   ├── specPlot.m
│   │   ├── specPlotPhase.m
│   │   ├── findBin.m
│   │   ├── findFin.m
│   │   ├── FGCalSine.m
│   │   ├── cap2weight.m
│   │   ├── weightScaling.m
│   │   ├── INLsine.m
│   │   ├── errHistSine.m
│   │   ├── sineFit.m
│   │   ├── tomDecomp.m
│   │   ├── NTFAnalyzer.m
│   │   ├── overflowChk.m
│   │   └── bitInBand.m
│   └── toolbox.ignore
├── toolbox/                 # 工具箱打包文件
│   ├── ADCToolbox_1v30.mltbx  # 最新工具箱包
│   ├── ADCToolbox_1v21.mltbx  # 历史版本
│   ├── ADCToolbox_1v2.mltbx
│   ├── ADCToolbox_1v1.mltbx
│   ├── ADCToolbox_1v0.mltbx
│   ├── ADCToolbox_0v12.mltbx
│   ├── ADCToolbox_0v11.mltbx
│   ├── ADCToolbox_0v1.mltbx
│   ├── deploymentLog.html
│   └── icon.png
└── .gitignore
```

## 常用工作流

### 1. 快速性能检查（面板方式）
```matlab
% 使用 adcpanel 进行一站式综合分析
rep = adcpanel(data, 'fs', fs);
% 所有指标保存在 rep 结构体中，配有整齐的图形面板
```

### 2. 快速性能检查（单独函数方式）
```matlab
% 仅执行频谱分析
[enob, sndr, sfdr] = plotspec(data);
```

### 3. 详细表征（手动工作流）
```matlab
% 使用单独函数进行综合分析
freq = findfreq(data, fs);
[enob, sndr, sfdr, snr, thd] = plotspec(data, fs, 2^bits, 5);
[inl, dnl] = inlsin(data);
[emean, erms, phase, anoi, pnoi] = errsin(data);
```

### 4. 校准工作流（逐位数据）
```matlab
% 使用 adcpanel 进行自动校准和分析
rep = adcpanel(bits, 'dataType', 'bits');
% 或手动校准工作流：
[weight, offset, postcal] = wcalsin(bits);
bitchk(bits, weight);
[enob_after_cal, ~] = plotspec(postcal, 'disp', false);
```

### 5. 频率扫描测试
```matlab
% 在多个频率下测试
for fin = test_frequencies
    bin = findbin(fs, fin, N);
    [enob(i), sndr(i)] = specplot(data{i}, fs, maxCode, 'disp', false);
end
plot(test_frequencies, enob);
```

## 技巧与最佳实践

1. **相干采样**：使用 `findbin` 确保相干采样，以获得准确的 FFT 分析
2. **过采样**：分析噪声整形 ADC 时在 `plotspec` 中设置 `'OSR'` 参数
3. **平均**：对重复测量使用 `'averageMode', 'coherent'` 以获得更好的噪底
4. **窗函数选择**：Hanning 窗（默认）适用于一般场景；相干信号使用矩形窗
5. **端点排除**：如果数据有削波或饱和，增大 `inlsin` 中的 `excl` 参数
6. **频率精度**：校准时让 `wcalsin` 自动搜索频率或使用精细搜索
7. **多组数据校准**：使用 cell 数组输入 `wcalsin` 以获得更好的统计收敛性
8. **秩亏**：如果 `wcalsin` 警告秩亏，根据预期比特权重调整 `'nomWeight'`

## 故障排除

### 问题：wcalsin 中出现 "Rank deficiency detected"
**解决方案：**
- 检查比特数据是否有足够的变化
- 调整 `'nomWeight'` 参数以匹配实际比特权重
- 确保输入数据覆盖全码范围

### 问题：wcalsin 中出现 "SNR below 20 dB" 警告
**解决方案：**
- 确认输入数据包含干净的正弦波信号
- 检查数据中是否存在过多噪声或削波
- 确保信号幅度适合 ADC 量程
- 此警告表明校准可能未能正确提取正弦波

### 问题：plotspec 中 ENOB 较差
**可能原因：**
- 非相干采样（使用 `window` 和 `sideBin` 应用适当的窗函数）
- 削波或饱和（使用 `errsin` 或 `bitchk` 检查）

### 问题：结果为 NaN 或 Inf
**解决方案：**
- 检查输入数据是否为全零或常数
- 确保数据缩放正确
- 验证采样频率为正值

### 问题：inlsin 返回异常 DNL
**解决方案：**
- 确保输入为整数码值（会自动取整并发出警告）
- 如果端点噪声较大，增大排除比例
- 确认输入来自正弦波（非其他波形）
- 使用足够的数据量进行分析（通常需要量化级数的 64 倍以上）

## 贡献指南

欢迎贡献！请遵循以下准则：
1. 保持与旧版函数名的向后兼容
2. 按照 MATLAB 标准添加完整的帮助文档
3. 包含输入验证和错误检查
4. 在函数帮助中添加示例
5. 添加新函数时更新本 README

## 版本历史

- **v1.30**（当前版本，2026-02-09）
  - 新增 `plotres` 和 `plotressin` 函数，支持半透明散点图
  - 在 `adcpanel` 中新增整数向量到二进制分解功能（`bits` 数据类型）
  - 在 `plotspec` 中为截尾均值索引添加边界保护
  - 修复 `tomdec` 中的直流拟合并改进绘图
  - 在 `sinfit` 中为频谱索引添加边界保护
  - 在 `wcalsin` 中为修补/常数列新增详细输出

- **v1.21**（2026-02-02）
  - 增强 `plotwgt`，新增权重缩放和有效分辨率显示
  - 修复 `effres` 公式：将 +1 移入 log2() 内
  - 改进 `maxCode` 计算和逐位数据的权重缩放
  - 在 `adcpanel` 中将所有绑图操作包裹在 `dispFlag` 检查中

- **v1.2**（2026-01-29）
  - 新增 `adcpanel` — 集成式 ADC 分析面板（初始版本）
  - 更新 `sinfit`：新增 `fsearch` 选项用于迭代频率细化、`verbose` 选项、以及使用 `inputParser` 处理可选输入
  - 在 `plotspec` 中实现窗函数感知的自动 `sideBin` 检测
  - 更新 `adcpanel` 以更好支持过采样
  - 重命名 `wcalsine` 为 `wcalsin`；重命名 `ovfchk` 为 `bitchk`；将 `bitact` 和 `bitsweep` 移至 legacy
  - 新增 `errsinv` 快捷方式用于 `errsin` 的数值横轴模式
  - 在 `plotspec` 中新增噪底估计的自动模式

- **v1.1**（2025-12-23）
  - 重大重构：重命名 11 个核心函数，建立清晰的命名规范（`analyze_*`、`plot_*`、`calc_*`）
  - 完成全部 21 个示例，按类别组织（b01–b04、a01–a14、d01–d05）
  - 修复 `plotspec` 中的 SNR 计算
  - 优化按相位分析误差的算法
  - 将 `weightScaling` 重命名为 `plotwgt` 并改进显示
  - 改进 `errsin` 的显示效果

- **v1.0**（2025-12-02）
  - 首个正式发布版
  - 统一核心函数命名：`errHistSine`→`errsin`、`inlsine`→`inlsin`、`sinefit`→`sinfit`、`specPlot`→`plotspec`、`specPlotPhase`→`plotphase`
  - 为所有函数添加完整文档
  - 添加 legacy 兼容包装函数
  - 重构测试套件，采用新的 runner 模式
  - 新增基于 LMS 的相位绘图算法

- **v0.12**（2025-11-26）
  - 函数重命名：`cap2weight`→`cdacwgt`、`findBin`→`findbin`、`sineFit`→`sinefit`、`findFin`→`findfreq`、`tomDecomp`→`tomdec`、`NTFAnalyzer`→`ntfperf`、`overflowChk`→`ovfchk`、`FGCalSine`→`wcalsine`、`bitInBand`→`ifilter`、`INLsine`→`inlsine`
  - 为所有重命名函数添加完整文档
  - 为所有重命名函数添加 legacy 兼容包装
  - 新增 `bitActivity` 工具和 `ENoB_bitsweep` 工具
  - 实现三层数据结构
  - 新增 Python 版本，15 项测试中实现 100% MATLAB–Python 一致性

- **v0.11**（2025-11-26）— 文档和测试更新
- **v0.1**（2025-11-26）— 初始工具箱打包

## 联系方式

如有问题、反馈或功能请求，请联系 jielu@tsinghua.edu.cn 或 zhangzs21@mails.tsinghua.edu.cn。

## 另见

- Python 实现：`../python/` - ADC 分析工具的 Python 版本
- 文档：`../doc/` - 附加文档和理论说明
- 测试数据集：`../dataset/` - 用于测试的 ADC 采集示例数据

---

**注意：** 本工具箱专为 ADC 测试和表征设计。在生产环境中使用时，请确保针对您的具体 ADC 架构和测试配置进行适当验证。
