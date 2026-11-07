# 本地验证状态

日期：2026-11-05。

## 已完成

- 官方源码固定至 8fd1e83faeb0ecff7986c4e5993f7398e6b6b6f8；17 个上游文件逐个从该 commit 的 Git blob 取出，SHA256 写入 vendor/UPSTREAM.json。
- Python 3.11.1 下纯标准库检查：28 项通过，1 项真实资产检查明确跳过（共 29 项）。命令：`python -B -m unittest discover -s tests -v`。
- 检查包括无向边与翻转、仓库排除、恢复边资格、保留原策略概率分母、同成本门控、RLOO、实例等权、重合零值、小样本描述标签、缓存复用/篡改检测、失败调用不退款、单写入锁、身份变化拒绝恢复，以及从分支原始成本重算报告和 INVALID/未完成标签。
- 已补充检查：保留原始确定性末步、自主/全步分母及零填充均值差异、无自主步时比例为 null、精英/非精英新边的边长与全客户近邻排名，以及报告重算后按实例/深度输出新口径。
- 可靠性检查覆盖缺数据库/表/字段、旧账本版本、孤立缓存与错误 call_id 关联、异常任务/调用类型及超预算、正常恢复不重复计费；最后任务提交后的 UNFINALIZED 窗口和恢复最终核验；任务集合替换、完成凭据/调用记录损坏；检查实例记录删减、参数篡改、替换为 passed-only、删除或未提交检查文件；完成后报告失败仍可重建；协议版本变化拒绝复用。相关记录为构造的可靠性夹具，不是 smoke 或求解器结果。
- 项目 Python 文件使用 AST 只读语法检查；CLI help 可正常显示。
- 示例配置因未提供真实 checkpoint SHA256 被拒绝，没有创建实验 run，没有自动安装依赖。

## 尚未完成

Windows 本机没有 NumPy、PyTorch，不能在本机执行真实模型检查。下述 2026-11-06 服务器检查现已完成实际 actor 加载、原版/适配动作一致性、真实状态恢复和四实例 smoke；正式 64 实例采集仍未执行。单元测试中的构造记录只验证协议与工程行为，不是模型或求解器实验结果。

真实适配检查由服务器 verify/smoke 执行，collect 必须绑定相同源码、资产、配置、环境的已完成 smoke。没有这些证据，不声称预检已经可在真实模型上通过。

实现的当前设备范围为 CPU；CUDA 不受支持，不能仅通过切换配置绕过一致性检查。

## 2026-11-06 服务器检查与修复

已复用服务器 Python 3.11.16、NumPy 1.26.4、PyTorch 2.5.1+cu121（CPU 路径），绑定原版 ALTR-100/P20% checkpoint 和 64 个旧 development 实例。用户明确授权这些实例用于本轮训练诊断，来源不改称独立验证或测试。每实例固定使用已有 HGS seed=271828 参考，不调用求解器生成精英。

首次 smoke_server_20261106_001 的四实例语义检查通过，但在首个实例反事实阶段因 Branch snapshot differs 停止，状态 INVALID。实际已计费 evolution_full=20、natural_full=4、partial=1、semantic_full=8、semantic_partial=8，记录保留，不参与科学分析。

原因是 instance_state 的 open 列表引用了活动环境的开放端点列表，后续自然动作会修改已选快照。现改为复制列表；回归测试验证源环境列表/路线继续修改时，快照内容及哈希保持不变。此修改不改变采样、Destroy、Repair 或统计协议，但代码身份变化，必须使用新运行目录，不能恢复 INVALID 运行。

修复后服务器单元检查 28 项通过、1 项 opt-in 真实资产单测跳过；真实适配检查由随后 smoke 显式执行，不代表实际跳过模型验证。smoke_server_20261106_002 四实例全部完成，audit、report、smoke_proof 均通过，参数哈希前后一致。实际调用 evolution_full=80、natural_full=48、partial=176，另有 semantic_full=8、semantic_partial=8。J=12 仅用于检查实现，scientific_conclusions_allowed=false，不据 smoke 宣称方法收益。

服务器 run：/data/kcx/projects/study_02_whole/frozen_repair_precheck/runs/smoke_server_20261106_002。完成凭据 SHA256 为 43d8685e6ba962b72a0f0e4cc1110cd61509cbf11a8cf849c303abf70d60fbc2；manifest SHA256 为 ab7436093e8c4072c510813424b7e3ff9050425085e57f7020fec4c513abffb7；checks SHA256 为 7664a26cef76fa3ac201e15cade13580ff1d23119a30d4bca76b991b49d72aba。失败的 smoke_001 保留且不复用。正式 64 实例实验未启动。
