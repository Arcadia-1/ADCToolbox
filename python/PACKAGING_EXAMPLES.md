# Examples 打包策略 - 最终方案

## ✅ 最终决定：Examples 不包含在 pip 包中

遵循 Python 包的标准做法，examples 保留在 GitHub 仓库，但**不随 pip 包一起发布**。

## 📋 实现清单

### 已完成的配置：

1. **`MANIFEST.in`** - 排除 examples
   ```
   prune examples
   prune tests
   ```

2. **`pyproject.toml`** - 无特殊配置
   ```toml
   [tool.setuptools]
   include-package-data = true  # 只包含必需文件
   ```

3. **`src/adctoolbox/examples_util.py`** - 辅助工具
   - 提供 GitHub 链接
   - 仅在开发模式下可用
   - 提供清晰的错误信息

4. **`examples/INSTALL.md`** - 用户指南
   - 说明如何访问 examples
   - 提供多种获取方式
   - 解释为什么不包含在 pip 包中

## 🎯 用户如何访问 Examples

### 方法 1：在线查看（推荐）
```
https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples
```

### 方法 2：克隆仓库
```bash
git clone https://github.com/Arcadia-1/ADCToolbox.git
cd ADCToolbox/python/examples
python quickstart/basic_workflow.py
```

### 方法 3：使用辅助工具获取链接
```bash
pip install adctoolbox
python -m adctoolbox.examples_util url
```

输出：
```
Examples URL: https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples
Repository URL: https://github.com/Arcadia-1/ADCToolbox
```

## 📦 包大小对比

| 配置 | 包大小 (估算) |
|------|--------------|
| 不含 examples | ~500 KB |
| 包含 examples | ~1.5 MB |
| **节省空间** | **~1 MB (67% 减小)** |

## ✅ 优势

1. **保持包轻量** - 用户 `pip install` 快速完成
2. **避免额外依赖** - examples 可能需要额外的可视化库
3. **遵循业界标准** - 与 requests、pandas、numpy 等保持一致
4. **便于维护** - examples 可以独立更新，不需要发布新版本
5. **离线可用** - 用户可以 `git clone` 后离线使用

## 🔍 业界参考

主流 Python 包的做法：

| 包名 | Examples 位置 |
|------|--------------|
| **requests** | GitHub only |
| **pandas** | GitHub only (tutorials) |
| **numpy** | GitHub only |
| **fastapi** | GitHub only (tutorials) |
| **scikit-learn** | GitHub only |
| **matplotlib** | 部分内置 (gallery)，完整示例在 GitHub |

## 🧪 测试打包结果

### 1. 构建包
```bash
cd python/
python -m build
```

### 2. 检查内容
```bash
# 检查 tar.gz 内容
tar -tzf dist/adctoolbox-*.tar.gz | grep -E "(examples|tests)"

# 应该返回空（examples 和 tests 已被排除）
```

### 3. 测试安装
```bash
# 创建虚拟环境
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# 从 wheel 安装
pip install dist/adctoolbox-*.whl

# 验证 examples 不在包中
python -m adctoolbox.examples_util list

# 应输出：
# ✗ Examples not included in pip installation
# Examples are available on GitHub:
#   https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples
```

## 📝 文档更新建议

在 `python/README.md` 中添加：

````markdown
## Examples

ADCToolbox provides comprehensive examples in the GitHub repository.

**View examples online:**
https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples

**Clone repository to run examples:**
```bash
git clone https://github.com/Arcadia-1/ADCToolbox.git
cd ADCToolbox/python/examples
python quickstart/basic_workflow.py
```

**Get examples URL:**
```bash
python -m adctoolbox.examples_util url
```

### Why aren't examples included in pip package?

Following Python packaging best practices:
- Keeps package lightweight (~500KB vs ~1.5MB)
- Avoids bundling development files
- Examples stay up-to-date on GitHub
- Matches standard practice (requests, pandas, numpy, etc.)

See `examples/INSTALL.md` for detailed instructions.
````

## 🚀 发布到 PyPI

准备发布时：

```bash
# 1. 构建包
python -m build

# 2. 检查包内容
twine check dist/*

# 3. 上传到 TestPyPI
twine upload --repository testpypi dist/*

# 4. 测试安装
pip install --index-url https://test.pypi.org/simple/ adctoolbox
python -m adctoolbox.examples_util url

# 5. 确认无误后上传到 PyPI
twine upload dist/*
```

## 🔧 开发模式

对于开发者，examples 仍然可用：

```bash
# 克隆仓库
git clone https://github.com/Arcadia-1/ADCToolbox.git
cd ADCToolbox/python

# 安装开发模式
pip install -e .

# 列出本地 examples
python -m adctoolbox.examples_util list

# 输出：
# ✓ Running in development mode
#   Examples location: /path/to/ADCToolbox/python/examples
```

## 📊 总结

| 方面 | 决策 |
|------|------|
| **Examples 包含在 pip 包中？** | ❌ 否 |
| **Examples 在 GitHub 仓库中？** | ✅ 是 |
| **用户如何访问？** | 在线查看或 git clone |
| **开发模式可用？** | ✅ 是 |
| **包大小节省** | ~1 MB (67%) |
| **符合业界标准？** | ✅ 是 |

这个方案平衡了以下需求：
- ✅ 包的轻量化
- ✅ 用户便利性（通过 GitHub）
- ✅ 开发者友好（开发模式仍可用）
- ✅ 遵循 Python 生态系统标准
