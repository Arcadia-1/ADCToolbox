# 访问 ADCToolbox 示例代码

**重要说明：** Examples **不包含在 pip 安装包中**，以保持包的轻量化。
所有示例代码都在 GitHub 仓库中提供。

## 🌐 在线查看示例（推荐）

最简单的方式是直接在 GitHub 上浏览：

**示例目录：** https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples

在线查看的优势：
- ✅ 无需下载
- ✅ 始终最新
- ✅ 可直接复制代码
- ✅ 有语法高亮

## 📥 方法 1：克隆仓库（推荐）

克隆完整仓库以获取所有示例：

```bash
# 克隆仓库
git clone https://github.com/Arcadia-1/ADCToolbox.git

# 进入示例目录
cd ADCToolbox/python/examples

# 运行示例
python quickstart/basic_workflow.py
```

## 📥 方法 2：仅下载 examples 文件夹

如果不想克隆整个仓库，可以使用以下方式：

### 使用 svn (如果已安装)
```bash
svn export https://github.com/Arcadia-1/ADCToolbox/trunk/python/examples
cd examples
python quickstart/basic_workflow.py
```

### 使用 GitHub CLI
```bash
gh repo clone Arcadia-1/ADCToolbox -- --depth=1 --single-branch
cd ADCToolbox/python/examples
```

### 手动下载
1. 访问 https://github.com/Arcadia-1/ADCToolbox
2. 点击 "Code" → "Download ZIP"
3. 解压后进入 `ADCToolbox-main/python/examples`

## 🔍 方法 3：使用辅助工具

安装 adctoolbox 后，可以使用内置工具获取 GitHub 链接：

```bash
# 获取示例 URL
python -m adctoolbox.examples_util url
```

输出：
```
Examples URL: https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples
Repository URL: https://github.com/Arcadia-1/ADCToolbox
```

## 💡 快速开始示例

如果只是想快速试用，可以直接复制以下代码：

```python
import numpy as np
from adctoolbox.aout import spec_plot
from adctoolbox.common import find_bin

# 生成测试信号
N = 2**12
J = find_bin(1, 0.1, N)
signal = 0.5 * np.sin(2 * np.pi * J / N * np.arange(N)) + 0.5

# 分析频谱
enob, sndr, sfdr, snr, thd = spec_plot(signal, label=True)
print(f"ENoB: {enob:.2f}, SNDR: {sndr:.2f} dB")
```

## 📂 示例目录结构

```
examples/
├── README.md                # 示例说明文档
├── INDEX.md                 # 所有示例索引
│
├── quickstart/              # 快速入门
│   └── basic_workflow.py
│
├── aout/                    # 模拟输出分析工具
│   ├── example_spec_plot.py
│   ├── example_tom_decomp.py
│   └── ...
│
├── dout/                    # 数字输出校准工具
│   ├── example_fg_cal_sine.py
│   └── ...
│
├── common/                  # 通用工具函数
│   ├── example_sine_fit.py
│   └── ...
│
└── workflows/               # 完整分析流程
    └── complete_adc_analysis.py
```

## ❓ 为什么 examples 不包含在 pip 包中？

这是 Python 包的标准做法，原因包括：

1. **保持包轻量** - pip 包应该只包含运行所需的代码
2. **避免额外依赖** - examples 可能使用额外的可视化、数据文件等
3. **遵循业界规范** - requests、pandas、numpy、fastapi 等都采用此方式
4. **便于维护** - examples 在 GitHub 上可以独立更新

## 🔧 开发模式

如果你是从源码安装（开发模式），examples 会在本地可用：

```bash
# 安装开发模式
cd ADCToolbox/python
pip install -e .

# 列出本地示例
python -m adctoolbox.examples_util list
```

## 📚 更多资源

- **文档：** https://github.com/Arcadia-1/ADCToolbox
- **问题反馈：** https://github.com/Arcadia-1/ADCToolbox/issues
- **示例在线浏览：** https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples
