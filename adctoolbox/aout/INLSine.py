import numpy as np

def INLsine(data, clip=0.01):
    """
    使用正弦拟合直方图法计算ADC的INL(积分非线性)和DNL(微分非线性)
    
    该函数通过对ADC输出数据的直方图进行反余弦变换来估计INL和DNL。
    基本原理：对于理想的正弦波输入，ADC输出码的累积分布函数(CDF)应该
    遵循反正弦函数。通过对累积直方图应用反余弦变换，可以得到理想的
    线性码分布，从而计算出非线性误差。
    
    Parameters
    ----------
    data : ndarray
        ADC输出数据 (1D数组)
    clip : float, optional
        裁剪比例，用于排除数据边缘的异常值，默认为0.01 (1%)
        
    Returns
    -------
    INL : ndarray
        积分非线性误差 (以LSB为单位)
    DNL : ndarray
        微分非线性误差 (以LSB为单位)
    code : ndarray
        对应的输出码值
        
    Notes
    -----
    INL表示实际转换特性曲线与理想直线的最大偏差
    DNL表示相邻码步长与理想步长(1 LSB)的偏差
    """
    
    # 确保data是列向量 (转置如果需要)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    S = data.shape
    if S[0] < S[1]:  # 如果行数小于列数，进行转置
        data = data.T
    
    # 展平为一维数组
    data = data.flatten()
    
    # 计算数据范围并应用裁剪
    # 裁剪是为了排除输入信号饱和区域的数据点，这些点会导致统计偏差
    max_data = np.ceil(np.max(data))
    min_data = np.floor(np.min(data))
    
    # 对数据范围两端各裁剪clip比例
    max_data = round(max_data - clip * (max_data - min_data) / 2)
    min_data = round(min_data + clip * (max_data - min_data) / 2)
    
    # 生成码值范围
    code = np.arange(min_data, max_data + 1)
    
    # 将数据限制在裁剪后的范围内
    data = np.clip(data, min_data, max_data)
    
    # 计算直方图 - 统计每个码值出现的次数
    # DCC: Digital Code Count，每个码的计数
    # MATLAB's hist(data, code) uses code values as bin centers
    # For consecutive integers, bins are [code[i]-0.5, code[i]+0.5)
    bins = np.append(code - 0.5, code[-1] + 0.5)
    DCC, _ = np.histogram(data, bins=bins)
    
    # 计算累积分布函数(CDF)并应用反余弦变换
    # 理论：对于理想正弦波，CDF应该是反正弦函数
    # 使用 -cos(π * CDF) 变换将其线性化
    # cumsum(DCC)/sum(DCC) 计算归一化的CDF (范围0到1)
    cumulative_prob = np.cumsum(DCC) / np.sum(DCC)
    DCC = -np.cos(np.pi * cumulative_prob)
    
    # 计算微分：相邻码之间的差值
    # DNL是每个码步长相对于理想1 LSB的偏差
    DNL = DCC[1:] - DCC[:-1]
    
    # 更新码值范围（因为差分减少了一个点）
    code = code[:-1]
    
    # 对DNL和码值再次进行边缘裁剪
    # 这是为了排除边缘效应，因为边缘处的统计可能不准确
    clip_points = int(np.floor(clip * (max_data - min_data + 1) / 2))
    
    if clip_points > 0:
        code = code[clip_points:-clip_points]
        DNL = DNL[clip_points:-clip_points]
    
    # DNL归一化处理
    # 1. 首先归一化使总和为1（概率归一化）
    DNL = DNL / np.sum(DNL)
    
    # 2. 缩放到实际码范围（以LSB为单位）
    # 理想情况下，每个码步长应该是1 LSB
    num_codes = max_data - min_data - clip_points * 2 + 1
    DNL = DNL * num_codes - 1
    
    # 3. 去除均值（去除增益误差的影响）
    # 这样DNL只反映非线性，不包含整体增益误差
    DNL = DNL - np.mean(DNL)

    # 计算INL：DNL的累积和
    # INL表示每个码点相对于理想直线的偏差
    # 通过对DNL积分(累加)得到
    INL = np.cumsum(DNL)

    return INL, DNL, code


