# 服务器输入资产

当前没有已确认兼容的 checkpoint、64 个训练实例与精英映射。官方源码已固定；官方仓库中的模型不自动视为与现有训练精英数据兼容。不能用旧项目的自定义网络权重代替原版 actor。

所有输入使用 UTF-8 JSON（可带 BOM）。路径可为绝对路径；配置路径相对配置文件，清单内实例/精英路径相对清单文件。资产检查不写文件，不加载不安全的 pickle 数据。

## 1. 运行配置

复制 `configs/experiment.example.json` 为自己的服务器配置，替换全部占位路径及 SHA256。支持 CPU，默认一个计算线程。当前不支持 CUDA；不能把 CPU 检查当作 GPU 一致性证据。固定协议由代码定义，不通过配置修改样本量或预算。

## 2. Checkpoint 与元数据

checkpoint 必须为官方 `torch.save` 字典，包含 `parameters`、`destroy_operation`、`p_destruction`。使用 `weights_only=True` 加载，严格匹配 actor 参数键与形状；不回退到任意 pickle 反序列化。隐藏维度从参数推断，不猜测规模或分布。

元数据结构如下。尖括号内容必须由真实资产证据替换；下面不是已确认的资产配置。

```json
{
  "checkpoint_sha256": "<真实 SHA256>",
  "problem": "CVRP",
  "nb_customers": 100,
  "distribution": "<可核实的训练分布标识>",
  "capacity": 50,
  "coordinate_scale": 1,
  "destroy_operation": "P",
  "p_destruction": 0.3,
  "round_distances": false,
  "evidence": "<训练配置/日志/官方模型路径等证据出处>"
}
```

这里的 100、50、P、0.3 都是格式示例，不是本项目对现有权重的兼容性声明。`coordinate_scale` 是原始坐标到网络坐标的除数。网络坐标须在 [0,1]；成本始终使用 original_locations。round_distances=true 使用 NumPy 原版逐边四舍六入五取偶约定，独立成本重算对应 Python round；不替换成 VRPLIB 的其他取整规则。分布、规模与距离证据无法从 state_dict 自动推断，需要提供真实来源，未知则不运行。

## 3. 实例文件

```json
{
  "instance_id": "train-000001",
  "distribution": "<与 checkpoint 元数据相同的真实标识>",
  "customer_ids": [0, 11, 27],
  "original_locations": [[0.5, 0.5], [0.1, 0.2], [0.8, 0.9]],
  "demand": [0, 2, 3],
  "capacity": 50
}
```

上例只有两个客户用于说明格式，不能用于声称 CVRP-100 兼容。数组第 0 项必须是仓库，原 ID 为整数 0；其余客户原 ID 是互异正整数，可以不连续。数组顺序是 NLNS 内部索引，日志另外记录原 ID。需求为正整数，单客户需求不超容量。不支持 SDVRP。

## 4. 精英文件

```json
{
  "instance_id": "train-000001",
  "instance_sha256": "<对应实例文件的精确 SHA256>",
  "source": "<真实求解器、版本、已有运行/文件出处>",
  "routes": [[0, 11, 27, 0]]
}
```

routes 使用原客户 ID，每条路线首尾为仓库。检查全部客户恰好一次、容量与身份，并按当前成本规则重算。精英原文件中额外的成本或最优性声明不作为当前目标下的证明。精英只由训练诊断流程读取。

## 5. 兼容资产全集清单

```json
{
  "schema": "frozen-repair-assets-v1",
  "instances": [
    {
      "instance_id": "train-000001",
      "split": "train",
      "split_evidence": "<原始训练划分文件/生成记录出处>",
      "instance": "instances/train-000001.json",
      "instance_sha256": "<真实 SHA256>",
      "elite": "elites/train-000001.json",
      "elite_sha256": "<真实 SHA256>"
    }
  ]
}
```

提供全部已核实兼容的训练资产，至少 64 个。检查拒绝非训练条目、坏资产及重复实例，不会静默过滤后挑出好样本。选样对排序后的 ID 使用固定独立子流，不根据精英质量或模型结果选样。指纹检测同一仓库及同一客户坐标/需求集合，避免通过更名或重排客户重复采样。原始划分来源的真实性仍需要资产提供者核实，程序无法仅凭一个标签证明训练归属。

已有 .pkl/.vrp 数据需要先按照这些明确字段导出；当前程序不根据文件名猜测缩放、客户映射或训练归属，不会自动生成新数据和精英。
