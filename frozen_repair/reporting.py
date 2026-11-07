import sqlite3
from pathlib import Path
from .common import read_json, file_hash, Invalid, Blocked, atomic_json, digest
from .statistics import clustered, distribution, mean
from .diagnostics import Diagnostic, natural_summary
from .evidence import audit


def generate(directory):
    directory = Path(directory)
    try:
        evidence = audit(directory)
    except Exception as exc:
        result = dict(status="INVALID", scientific_conclusions_allowed=False,
                      reason=f"Evidence integrity failed: {type(exc).__name__}: {exc}")
        atomic_json(directory / "summary.json", result)
        (directory / "report.md").write_text(
            "# 冻结 Repair 预检\n\n冻结网络、训练数据诊断；未运行训练或完整搜索收益评估。\n\n"
            "状态：INVALID；完整性检查失败，禁止科学结论。\n\n"+result["reason"]+"\n", encoding="utf-8")
        return result
    manifest = read_json(directory / "manifest.json")
    if manifest["kind"] == "verify":
        raise Blocked("verify contains semantic checks only; inspect checks.json, not an experimental report")
    if digest(manifest["identity"]) != manifest["identity_sha256"]:
        raise Invalid("Manifest identity checksum mismatch")
    db = sqlite3.connect((directory / "ledger.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT id,status,hash FROM tasks ORDER BY id").fetchall()
        counts = dict(db.execute("SELECT kind,count(*) FROM calls GROUP BY kind"))
    finally:
        db.close()
    points, summaries, missing, per_instance, per_depth = [], [], [], {}, {}
    for task, status, checksum in rows:
        if status != "completed":
            continue
        path = directory / "raw" / (task+".json")
        if file_hash(path) != checksum:
            raise Invalid("Raw artifact checksum mismatch")
        result = read_json(path)
        ident = result["instance_id"]
        local_summaries = []
        for state in result["states"]:
            previous = [[int(e[0]) for e in state["complete_state"]["pool"][i]]
                        for i in state["complete_state"]["solution"]]
            Diagnostic(result["elite_routes"], previous, result["instance"]).annotate(
                state["trajectories"], state["before_cost"])
            s = natural_summary(state["trajectories"])
            summaries.append(s)
            local_summaries.append(s)
            per_depth.setdefault(str(state["depth"]), []).append(s)
            if state["point"]:
                point = state["point"]
                point["mean_costs"] = {b: sum(r["cost"] for r in point["results"][point["aliases"][b]])/8
                                       for b in ("E", "P", "N")}
                point["dEP"] = (point["mean_costs"]["E"]-point["mean_costs"]["P"])/state["before_cost"]
                point["dEN"] = (point["mean_costs"]["E"]-point["mean_costs"]["N"])/state["before_cost"]
                points.append(point)
            else:
                missing.append(dict(instance_id=ident, depth=state["depth"]))
        per_instance[ident] = aggregate_natural(local_summaries)
    total = len(manifest["task_ids"])
    run_status = evidence["status"]
    complete = run_status == "COMPLETE"
    comparisons = {key: clustered(points, key, total) for key in ("dEP", "dEN")}
    if not complete:
        for c in comparisons.values():
            c["comparison_label"] = run_status
    summary = dict(scope=manifest["identity"]["scope"], run_kind=manifest["kind"],
                   status=run_status, scientific_conclusions_allowed=complete and manifest["kind"] == "collect",
                   protocol_version=manifest["protocol_version"],
                   initialization=manifest["identity"]["initialization"], tasks_completed=len(per_instance),
                   total_instances=total, J=len(points), instances_with_points=len({p["instance_id"] for p in points}),
                   missing_opportunities=missing, comparisons=comparisons,
                   divergent_exploratory={k: clustered(points, k, total, True) for k in ("dEP", "dEN")},
                   by_depth={str(d): {k: clustered([p for p in points if p["depth"] == d], k, total)
                                     for k in ("dEP", "dEN")} for d in (0, 5, 20)},
                   natural=aggregate_natural(summaries), natural_by_instance=per_instance,
                   natural_by_depth={d: aggregate_natural(s) for d, s in per_depth.items()},
                   budget=dict(charged_calls=counts, limits=manifest["limits"],
                               requested_partial=sum(p["requested_completions"] for p in points),
                               completed_partial=sum(p["executed_completions"] for p in points),
                               note="Charged calls include interrupted attempts; full and partial calls are separate"),
                   raw_points=[{k: p[k] for k in ("point_id", "instance_id", "depth", "dEP", "dEN", "actions", "aliases",
                                                       "before_cost", "mean_costs", "targets_original_ids")} for p in points],
                   training_decision="NOT_TESTED; a separate budgeted A/B/C comparison is required",
                   interpretation="Local frozen-policy evidence is auxiliary, not a necessary condition for training benefit")
    if not complete:
        for c in summary["divergent_exploratory"].values():
            c["comparison_label"] = run_status
        for groups in summary["by_depth"].values():
            for c in groups.values():
                c["comparison_label"] = run_status
    atomic_json(directory / "summary.json", summary)
    text = ["# 冻结 Repair 训练数据预检", "",
            "冻结网络、训练数据诊断；未运行训练或完整搜索收益评估。", "",
            f"运行类型：{manifest['kind']}；状态：{summary['status']}。",
            f"协议版本：{manifest['protocol_version']}。",
            "初始化沿用固定上游坐标乘积平方排序，非欧氏最近邻；结论仅适用于该初始化及后续状态链。",
            f"完成 {len(per_instance)}/{total} 个实例；可用点 J={len(points)}；"
            f"有机会实例 {summary['instances_with_points']}/{total}。", "",
            "## 冻结策略反事实", "",
            "负差值表示精英动作更好；按实例等权，实例聚类 bootstrap 2000 次。",
            "δ=0.001 仅为实际意义参照，不是训练准入门槛。", ""]
    for key, c in comparisons.items():
        text.append(f"- {key}: estimate={c['estimate']}, 95% CI={c['ci95']}, "
                    f"instances={c['available_instances']}, label={c['comparison_label']}。")
    text.extend(["", "少于 32 个可用实例仅作描述；smoke 不能用来作科学决策。",
                 "全部抽样点保留重合零值；分歧子集及各深度另见 summary.json。", "",
                 "## 自然辅助信号", "",
                 f"q*n*h 总触发动作数：{summary['natural']['qnh']}；"
                 f"全部自主动作数：{summary['natural']['actions']}。",
                 f"全部 Repair 步数：{summary['natural']['total_steps']}；"
                 f"唯一合法目标的确定性步数：{summary['natural']['deterministic_steps']}。",
                 "主统计分母为合法目标数大于 1 的决策步；全步辅助统计见 natural.all_steps。",
                 "覆盖、条件/零填充概率质量、大小分布、系数及实例分布见 summary.json。",
                 "精英/非精英新边的边长及近邻排名分布见 natural.new_edge_geometry，按实例和深度亦有汇总。",
                 "近邻排名基于全体其他客户，距离相同时按原客户 ID 排序，不是合法目标集合内排名。",
                 "自然好轨迹与精英命中的关联不证明强化这些动作会改善结果；系数不是梯度范数。", "",
                 "## 下一步与限制", "",
                 "资产和实现通过后，结合自然信号覆盖、集中程度及反事实证据，另行决定有限 A/B/C 训练比较。",
                 "不根据任意命中率阈值自动宣布训练可行；反事实不确定也不等于训练方法无效。",
                 "保留伤害证据；本实验不检验训练后泛化或完整 NLNS 搜索收益。", "",
                 "## 预算", "", f"已计费调用：{counts}。",
                 "完整 Repair 与部分补全分列；中断后重跑仍消耗预算，耗尽时停止，不追加额度。",
                 "初始化、Destroy、快照、统计属于额外开销；各实例耗时在原始记录中，语义检查单列。", ""])
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
    return summary


def aggregate_natural(summaries, include_auxiliary=True):
    count = sum(s["actions"] for s in summaries)
    out = dict(actions=count, states=len(summaries), states_with_qnh=sum(s["any_qnh"] for s in summaries))
    for k in ("n", "nh", "qnh"):
        value = sum(s[k] for s in summaries)
        out[k], out[k+"_per_action"] = value, value/count if count else None
    for k in ("Vstar", "Vnew"):
        groups = [s[k] for s in summaries]
        out[k] = dict(nonempty_rate=sum(g["nonempty"] for g in groups)/count if count else None,
                      size=distribution([x for g in groups for x in g["sizes"]]),
                      conditional_mass=distribution([x for g in groups for x in g["conditional_masses"]]),
                      zero_filled_mass=distribution([x for g in groups for x in g["zero_filled_masses"]]))
    coeff = [c for s in summaries for c in s["coefficients"]]
    nonzero = [abs(c) for c in coeff if c != 0]
    out["coefficient_abs"] = distribution(nonzero)
    out["coefficient_abs_scope"] = "conditional on nonzero coefficient"
    out["coefficient_abs_zero_filled"] = distribution([abs(c) for c in coeff])
    out["coefficient_nonzero_rate"] = len(nonzero)/count if count else None
    out["within_P_good_minus_other_nh"] = distribution([s["natural_good_vs_other_nh"] for s in summaries
                                                       if s["natural_good_vs_other_nh"] is not None])
    for key in ("natural_good_vs_other_nh_per_action", "natural_good_vs_other_nh_per_new"):
        out[key] = distribution([s[key] for s in summaries if s[key] is not None])
    out["trajectory_action_counts"] = distribution([t["actions"] for s in summaries for t in s["trajectories"]])
    out["trajectory_nh_per_action"] = distribution([t["nh_per_action"] for s in summaries for t in s["trajectories"]
                                                   if t["nh_per_action"] is not None])
    out["trajectory_nh_per_new"] = distribution([t["nh_per_new"] for s in summaries for t in s["trajectories"]
                                                if t["nh_per_new"] is not None])
    out["denominator_scope"] = "multiple_legal_targets" if include_auxiliary else "all_repair_steps"
    if include_auxiliary:
        out["statistics_schema"] = "natural-autonomous-v2"
        out["total_steps"] = sum(s["total_steps"] for s in summaries)
        out["deterministic_steps"] = sum(s["deterministic_steps"] for s in summaries)
        out["all_steps"] = aggregate_natural([s["all_steps"] for s in summaries], False)
        out["trajectory_total_step_counts"] = distribution([t["total_steps"] for s in summaries for t in s["trajectories"]])
        out["new_edge_geometry"] = dict(
            rank_scope="all_other_customers; unrounded Euclidean distance; ties by original customer ID",
            edge_scope="autonomously created, retained, novel undirected edges; deduplicated within trajectory")
        for label in ("elite", "nonelite"):
            groups = [s["new_edge_geometry"][label] for s in summaries]
            lengths = [v for g in groups for v in g["edge_lengths"]]
            ranks = [v for g in groups for v in g["nearest_neighbor_ranks"]]
            out["new_edge_geometry"][label] = dict(count=len(lengths), edge_length=distribution(lengths),
                                                   nearest_neighbor_rank=distribution(ranks))
    return out
