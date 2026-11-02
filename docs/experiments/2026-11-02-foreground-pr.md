# 独立前台校准 PR 验证记录

日期：2026-11-02。基于上游 `main` 的 `cd72061`，分支 `codex/foreground-calibration`。

## 提交范围

本 PR 提供四个公开接口：`build_calibration_features`、`fit_calibration_polynomial`、
`fit_calibration_mlp`、`apply_calibration`。训练接收显式对齐参考，运行时使用冻结映射，
拟合参数不解释为内部电容值或级间增益。历史特征支持连续块状态，独立采集须重置。

示例放在独立的 `12_foreground/`，保留原编号 exp_cal03、05、10、11。
非线性／记忆实验使用原有 `siggen` 接口；SAR 及联合误差使用原有 SAR 模型。
已移除自建 FIR 前端。监督数据划分、验证选择、SNDR/SFDR、多音口径、位权观测条件、
独立参考与 SAR 输入旁路条件均在包内文档中说明。

本分支不依赖 [Dither PR #102](https://github.com/Arcadia-1/ADCToolbox/pull/102)，
不重复提交其模型、跟踪器或实验。原共享 FFT 指标函数移入本示例目录，数值算法不变。
混合初始化的旧实验属于同一 Dither 增益跟踪比较，不另增公共后台 API。
既有 `models`、`siggen` 和频谱数值实现不变；保留上游最新 SFDR 修复。
示例之外的 `analyze_error_spectrum.py` 仅补一行 docstring 空行，以通过严格文档构建，
与 Dither PR 的同一文档修复一致。旧 dout API 页面改为引用统一 calibration API 页面，
避免重复 autodoc 条目。

## 实验与提取验证

- 独立分支相关测试：**43 passed、1 skipped**。跳过项为 Windows 目录符号链接权限；
  两条已有正弦拟合示例的迭代警告不属于本次失败。
- 严格 Sphinx：`-W --keep-going` 通过。
- Wheel、sdist 构建通过，不含缓存或实验 output 目录；未打包 Dither 示例。
- 独立虚拟环境实际安装 wheel，确认导入来自 `site-packages`。两种原仓库器件的
  单音／混合／多音短实验共 **842 行**冻结核对；二进制／冗余 SAR 位权比较
  **238 行**；两种结构联合校准 **324 行**。示例复制和临时技能安装通过。
- 用提取后的代码回放原正式实验：前台 **2904 行**、位权 **612 行**、联合校准
  **972 行**，合计 **4488 行**，最大独立 FFT 指标差 **4.21e-11 dB**，未重新训练。

原实验完整训练预算及性能见[原仓库模型重跑](2026-11-02-foreground-repository-models.md)
和[联合校准](2026-11-02-combined-foreground.md)。目录调整不是一次新的正式训练。
历史 CSV/JSON 保留原字节与原路径、源码哈希；本 PR 的源码回放哈希单独记录。
旧自建模型数据明确标为历史，不作为当前模型成绩。

原仓库记忆模型含粗／细量化和码反馈，其结果显著低于旧线性 FIR 的成绩。
前台单音／混合训练的 12 次 MLP 均达到 200 次迭代预算，未满足优化器收敛条件。
联合校准 MLP 同样保留其预算终止状态。所有结果是浮点行为验证，不是实测芯片、
晶体管级、RTL、定点或硬件吞吐验证，也不证明某类算法普遍优胜。

## 可复现命令

在本分支 `python/` 运行：

```powershell
uv run --extra docs --with pytest pytest tests/unit/test_foreground_workflows.py tests/unit/test_foreground_bitweights_example.py tests/unit/test_combined_foreground_example.py tests/unit/test_skill_cli.py tests/integration/test_user_guide_skill_examples.py -q -rs
uv run --extra docs sphinx-build -b html -W --keep-going docs/source <new-html-directory>
uv build --out-dir <new-dist-directory>
```

完整训练与新目录的原生核对命令见 `12_foreground/README.md` 及邻近指南。
如果持有原五组完整 output 目录，可执行下面的迁移回放，不修改历史配置或源码哈希：

```powershell
uv run python ../docs/experiments/replay_foreground_extraction.py --source-output <original-output-root> --output <new-replay-json>
```

[正式数据迁移回放证据](foreground-pr-replay-final-20261102.json)记录当前源码哈希和历史输入哈希；
[安装包验证记录](foreground-pr-distribution-20261102.json)记录发行包哈希与安装后测试。
完整模型和源码快照位于原实验输出目录；提交的 CSV/JSON 提供可审查的数值和出处，
不声称这些汇总文件单独包含全部训练数据或模型权重。
