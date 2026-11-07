# 冻结 NLNS Repair 的精英信号预检

独立、固定预算的训练数据诊断工具。不训练网络，不生成精英，不声称改进完整 NLNS 搜索。当前实现 CPU 路径；正式模型运行需要服务器现有 NumPy/PyTorch 和经过核实的资产。

当前协议版本为 `frozen-repair-precheck-v2`，须在服务器配置中显式填写。此次更新仅加强可靠性及说明，采样、Destroy、Repair、统计估计方式不变。旧运行/账本不自动迁移，应使用新运行目录。

官方源码：<https://github.com/ahottung/NLNS>，commit `8fd1e83faeb0ecff7986c4e5993f7398e6b6b6f8`。`vendor/NLNS` 保留原始源码及 GPL-3.0 许可证；运行时兼容改动见 [协议](docs/PROTOCOL.md)。

本地验证证据与尚未完成的真实模型检查见 [VALIDATION.md](docs/VALIDATION.md)。

## 上传与环境

上传 frozen_repair、vendor/NLNS、vendor/UPSTREAM.json、configs、scripts、tests、docs、requirements.txt、README.md、LICENSE。不要上传 runs、test_artifacts、vendor/upstream_checkout（源码下载用的临时完整 checkout，包含未选用的官方资产及 Git 数据）。无需临时 checkout 即可运行。

也可直接上传 `dist/frozen_repair_precheck_v1.zip`，解压后进入 frozen_repair_precheck 目录。压缩包只含上述源码与说明及逐文件校验清单，不包含权重、实例、精英或任何实验结果。重建命令为 `python -B scripts/package_project.py`。

使用已有 Python 3.11+ 环境。程序不自动安装依赖或联网。`requirements.txt` 是依赖要求，安装需用户单独执行。当前本机缺少 NumPy/PyTorch，不能把纯 Python 检查报告成真实模型 smoke 已通过。

## 准备资产

按 [ASSETS.md](docs/ASSETS.md) 准备真实 checkpoint 元数据、至少 64 个兼容训练实例及已有精英的全集清单。复制示例配置为服务器上的 `configs/server.json`，替换路径和哈希。示例中的规模/分布不能作为证据，未知资产阻塞运行。相对路径按配置文件/清单所在目录解析。

## 命令

在项目根目录执行；`-B` 防止 Python 写入 bytecode，尤其是只读 inspect/status。

```bash
python -B -m frozen_repair inspect --config configs/server.json
python -B -m unittest discover -s tests -v
python -B -m frozen_repair smoke --config configs/server.json --run-dir runs/smoke_001
python -B -m frozen_repair collect --config configs/server.json --smoke-run runs/smoke_001 --run-dir runs/precheck_001
```

smoke 已包含四实例真实语义检查；无需在 smoke 前另跑 verify。若只想检查适配一致性，可单独运行：

```bash
python -B -m frozen_repair verify --config configs/server.json --run-dir runs/verify_001
```

verify 不替代完整 smoke，不能直接作为 collect 凭据。真实适配单测也可通过 PRECHECK_CONFIG 显式启用，费用另计；默认会明确跳过。

恢复与报告：

```bash
python -B -m frozen_repair status --run-dir runs/precheck_001
python -B -m frozen_repair resume --run-dir runs/precheck_001
python -B -m frozen_repair report --run-dir runs/precheck_001
```

初次运行目录必须不存在，防止覆盖。resume 从 manifest 读取原配置并重验代码/资产/环境；不重新选样，不重复累计已完成调用。未提交的中断调用仍计费。达到固定预算则停止；INVALID 不可恢复。退出码：0 完成、2 资产/预算阻塞、130 中断、1 实现或数据异常。

Linux 包装器：`bash scripts/run_server.sh <上述子命令和参数>`；`PRECHECK_PYTHON` 可指定已有解释器。包装器前台运行，日志直接输出到终端，不暗中开启后台任务或安装依赖。

## 输出

- manifest.json：完整兼容全集、抽样名单、配置、资产/源码哈希、模型结构、设备版本与随机规则。
- ledger.sqlite：任务状态、逐调用计费、缓存校验；完整 Repair 与部分补全分开。
- raw/calls：每次已完成模型调用及状态/轨迹缓存。
- raw/<task>.json：每实例状态、自然动作、精英资格、干预快照、三分支补全及耗时。
- checks.json：四实例真实适配检查，仅 verify/smoke 写入。
- completion.json / interruption.json / failure.json：完成、中断/预算阻塞、INVALID 证据。
- summary.json / report.md：从校验后的原始轨迹和分支成本重算；可报告未完成状态，但不据部分结果做科学决策。

自然辅助信号与冻结策略的反事实比较分开呈现。保留全部重合零值，另报分歧子集；按实例等权汇总，bootstrap 2000 次。δ=.001 保留为实际意义参照，不作为训练准入硬门槛。后续训练 A/B/C 和无精英验证是独立实验，不在本轮自动执行。

自然主统计以合法目标数大于 1 的自主决策步为分母；原始 Repair 步全部保留，total_steps/deterministic_steps 记录全部/确定性步，natural.all_steps 提供全步辅助口径。无自主步的比例为 null。natural.new_edge_geometry 汇总精英与非精英新边的边长及全客户近邻排名，并按实例/深度提供；排名不局限于合法目标。当前统计标识为 natural-autonomous-v2。代码身份随本次修改变化，旧 smoke/run 不能直接与新代码拼接恢复，应使用新的运行目录。

报告区分 INVALID / INCOMPLETE / UNFINALIZED / COMPLETE。任务齐全但未写有效完成凭据时只能描述；恢复最终核验后才可 COMPLETE。完成凭据绑定任务文件、账本计费、检查和参数哈希，smoke 检查另有账本提交记录。缺表、孤立缓存、损坏检查凭据都拒绝恢复，不自动修补。检查阶段未提交就中断时，保守要求新运行目录；正式采集的已提交缓存仍可复用。若完成后仅报告生成失败，执行 `report` 即可重建，无需重跑采集。

初始化沿用固定上游的坐标乘积平方排序，非欧氏最近邻。此限制在 manifest 与报告中记录；结论只适用于该初始化及其后续状态链，未来修正需新协议重新实验。
