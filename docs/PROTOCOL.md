# 固定协议与实现边界

冻结网络、训练数据诊断，未运行训练或完整搜索收益评估。

协议版本为 frozen-repair-precheck-v2，在配置、manifest、报告中一致记录；账本结构版本为 2（SQLite user_version），检查凭据格式版本为 1。版本不兼容直接拒绝，不自动迁移旧账本或混合旧运行。相对桌面编码 Prompt，保留各比较的冻结策略局部效果标签，不自动输出 READY_FOR_TRAINING_PILOT；是否开展 A/B/C 训练是独立决策。

初始化仍直接使用固定上游 create_initial_solution。其排序函数为 sum((locations[candidate]*locations[origin])**2)，是坐标乘积平方排序，并非欧氏最近邻。此已知限制写入 manifest 的 identity.initialization 和报告，结论只适用于该初始化及后续状态链。vendor 文件未改动；将来修正公式需新协议、重新 smoke 和正式采集，不能与当前结果混合。

## 采样

64 个互异训练实例；官方 create_initial_solution 构造 S0。20 次原版 Destroy+冻结 categorical Repair，每次无条件回写。深度 0/5/20 各做一次独立 Destroy，四条自然轨迹来自同一残余 P。诊断与干预结果不回写。不是原版只接受改进的搜索收益实验。

随机种子由 SHA256(canonical JSON([20261105, instance_id, task parts...])) 前 32 bit 派生。状态演化、诊断 Destroy、自然轨迹、决策点蓄水池、精英目标、反事实分支各自隔离。CPU NumPy/Torch 全局 RNG 在子流结束后恢复；不使用进程随机 hash。bootstrap 使用独立种子 20261106。

原始 actions 保留全部 Repair 策略调用步；环境选参考端点不计步。统计中明确以执行当时合法目标数大于 1 定义 autonomous，唯一合法目标步标记为确定性步。主统计的 actions、非空率、概率质量零填充均值、命中率和系数非零比例均以自主决策步作分母；natural.all_steps 保留全部 Repair 步口径的辅助统计。total_steps 和 deterministic_steps 分别计全部步与确定性步。没有自主决策步时相关比例记 null，不用 0 或分母 1 代替。此口径仅用于诊断，不跳过原版采样，不改变 RNG、轨迹或补全。动作前记录原客户身份，避免环境拼接/翻转改变映射。末步及其原始 log 概率完整保存，不沿用原版末步乘零的日志值；末步是否自主仍由合法目标数决定。每条轨迹内无向客户边去重，并检查最终保留，恢复边 n=0，仓库边 n=0，确定性步 n=0。

natural.new_edge_geometry 分别汇总自主建立且最终保留的精英新边（n=1,h=1）和非精英新边（n=1,h=0）的数量、边长与近邻排名分布；natural_by_instance、natural_by_depth 同样提供。边长为原坐标未取整欧氏距离；排名在全体其他客户中计算，排除仓库和自身，等距按原客户 ID 升序，不是合法目标集合内排名。这些是描述性诊断，不证明已控制几何因素。natural.statistics_schema=natural-autonomous-v2 标记当前统计口径。

Vstar/Vnew 资格仅含合法客户；概率仍以包含仓库的原策略分布为分母。q 的容差为 1e-8*max(1,abs(C(S)))，与实际意义 δ 不同；同成本 q 全为 0。alpha=0.5 仅作系数诊断，未计算梯度范数或反向传播。

## 反事实

每个状态只从轨迹 1 的 Vnew 非空决策点均匀选一个。E 从该点 Vnew 均匀选一次；P 是该点原采样动作；N 是最近合法客户，用原始坐标未取整欧氏距离，距离相同时按原客户 ID。各目标保持已确定参考端点，每个不同动作补全 8 次。重合标签共享结果，不删除样本。不同动作使用独立随机流，未使用共同随机数。

主分析包含全部抽样点，分歧子集为额外探索分析。按实例内点均值再实例等权，bootstrap 对有点实例重采样并保留其内部观测，95% percentile CI，2000 次。机会覆盖以全部 64 个实例报告；各深度单列。

δ=.001 是成本比例参照，不是训练准入门槛。MEANINGFUL_LOCAL_ADVANTAGE_SUPPORTED、NO_MEANINGFUL_ADVANTAGE_DETECTED、HARM_SUPPORTED、INCONCLUSIVE 描述冻结策略的局部比较，两项分别分类。HARM_SUPPORTED 优先；不足 32 个有点实例统一只作描述并标 INCONCLUSIVE；J=0 标 NO_ELIGIBLE_POINTS。自然信号覆盖与关联另报，不使用任意 15% 门槛，不自动发出 READY_FOR_TRAINING_PILOT。必须另立有限 A/B/C 训练实验才可评估学习收益。

## 预算与恢复

正式上限分别为 evolution_full=1280、natural_full=768、partial=4608；不把部分补全计为完整 Repair。smoke 固定取抽中名单前四个实例，运行同一协议，上限为 80、48、288。smoke 内语义检查额外计 semantic_full/semantic_partial，正常各 8 次；独立 verify 也是这两项，均不进入正式统计。重试的硬上限各为 64，防止反复验证无界执行。

每个模型调用前在 SQLite 持久记账，完成调用后原子提交缓存及校验。状态演化、自然轨迹和每次反事实补全独立缓存，恢复不会重复调用已提交且校验通过的结果；中断调用不退款，重做仍计费。因此极端中断可能耗尽固定预算，程序停止，不暗中增加额度。账本、缓存或完成产物缺失/损坏拒绝复用。正式任务以实例为提交单位，原始逐调用缓存保留。

配置、源码、资产、运行环境哈希固定；恢复时重新核验。操作系统锁保证单写入者，进程退出后释放，锁文件可保留。INVALID 是终态，不允许修补代码后拼接旧记录。仅凭预算不足或中断不会认定方法无效。

新运行才创建数据库/表。恢复以 SQLite mode=rw 打开，不执行 CREATE TABLE；报告与 smoke 复用以 mode=ro 校验。检查结构版本、表/字段类型、外键、数据库完整性、任务名单、调用归属/类型及预算。缓存必须关联唯一已计费 call_id，且 kind/task/detail 与缓存键一致。损坏、缺失或旧结构不自动修补。

checks.json 保存格式版本、identity 哈希、指定四实例、前后参数哈希、逐项检查结果和带实例/角色的语义调用 ID；成功写入后将文件哈希提交到 proofs 表。恢复和 smoke_proof 共用严格验证函数。文件缺失、被替换、记录不全或参数不符拒绝复用；文件已写但未提交，或检查中途只留下计费记录时，保守拒绝恢复检查阶段，要求新运行目录，不默默重跑覆盖证据。已提交的正常模型调用缓存恢复不重复计费。

完成顺序为：全部任务提交 → 核对任务 ID 集合、原始文件与账本 → 重新核验源码、资产、模型参数 → 原子写 completion.json → 生成报告。完成凭据绑定 manifest 哈希、逐任务文件哈希、调用计数、账本记录摘要、检查/引用 smoke 凭据和模型参数哈希。报告只验证这些已提交证据，不重新加载模型或源数据。

报告状态：失败记录或完整性异常为 INVALID；任务未齐全为 INCOMPLETE；任务齐全但缺完成凭据为 UNFINALIZED；全部绑定产物与完成凭据有效才为 COMPLETE。只有 COMPLETE 才给比较效果标签，smoke 仍只作描述。存在但损坏的完成凭据判 INVALID，不降为 UNFINALIZED。最后任务提交后中断可恢复最终核验，无需重复采集；完成后单纯报告生成失败不更改任务/完成凭据，可单独执行 report 重建。

## 官方源码兼容改动

vendor/NLNS 原文件与固定 Git blob 完全一致，由 vendor/UPSTREAM.json 记录哈希。运行时仅适配：

1. Pointer.forward 的两个 squeeze 明确维度，修复 batch=1 维度消失；验证与未修改版本 batch=2 输出逐值一致。
2. 执行参考 Repair 循环时，在内存将已移除别名 np.long 改为 np.int64。

环境动作、掩码、初始化和 Destroy 直接调用原方法。中间状态采用完整引用图编码；不用官方 __deepcopy__，因为它丢失 incomplete_tours、端点映射及 open_nn_input_idx。快照同时包含被 actor 原地修改的 static、dynamic、当前参考端点与步数。

4 实例真实检查比较原循环和适配循环的动作、成本、路线、非末步 logp，检查末步记录、快照回放、分支隔离与参数哈希。真实检查未通过，不允许启动正式采集。CPU 一致性不能外推到 CUDA 或其他软件环境。
