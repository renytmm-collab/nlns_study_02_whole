import time
from .common import distance, edge, Invalid, digest
from .rng import local, stream, seed
from .adapter import Session, routes


def complete_point(selected, actor, metadata, x, ident, depth, before_cost, diagnostic, ledger, task):
    if selected is None:
        return None
    snap, d = selected["snapshot"], selected["decision"]
    aE = local(ident, depth, "elite-target").choice(d["Vnew"])
    customers = [i for i in d["legal"] if d["mapping"][i] != 0]
    aN = min(customers, key=lambda i: (distance(x["original_locations"], d["origin"], d["mapping"][i]),
                                      x["customer_ids"][d["mapping"][i]]))
    actions = dict(E=aE, P=selected["policy_target"], N=aN)
    groups, aliases = {}, {}
    for branch, action in actions.items():
        representative = next((b for b, a in actions.items() if a == action), branch)
        aliases[branch] = representative
    for branch, action in actions.items():
        if aliases[branch] != branch:
            continue
        results = []
        for k in range(8):
            parts = (ident, depth, "counterfactual", branch, k)
            def complete():
                started = time.perf_counter()
                with stream(*parts):
                    session = Session.restore(snap, actor, metadata)
                    if digest(session.snapshot()) != selected["snapshot_hash"]:
                        raise Invalid("Branch snapshot differs")
                    forced = session.apply(d, action)
                    tail, cost = session.finish()
                return dict(replicate=k, seed=seed(*parts), cost=cost, feasible=True, status="completed",
                            elapsed_seconds=time.perf_counter()-started, completion_steps=1+len(tail),
                            forced_action=forced, actions=tail, routes=routes(session.instance))
            results.append(ledger.cached("partial", task, [depth, branch, k, action], complete))
        groups[branch] = results
    costs = {b: sum(r["cost"] for r in groups[aliases[b]])/8 for b in actions}
    for b in actions:
        if actions[b] == aE and costs[b] != costs["E"]:
            raise Invalid("Overlapping branch reuse failed")
    nedge = edge(d["origin"], d["mapping"][aN])
    return dict(point_id=digest([ident, depth, d["step"]]), instance_id=ident, depth=depth,
                eligible_count=selected["eligible_count"], snapshot=snap, snapshot_hash=selected["snapshot_hash"],
                decision=d, actions=actions, aliases=aliases,
                targets_original_ids={b: x["customer_ids"][d["mapping"][a]] for b, a in actions.items()},
                nearest_is_new=nedge not in diagnostic.previous, nearest_is_elite=nedge in diagnostic.elite,
                requested_completions=24, executed_completions=len(groups)*8, results=groups,
                mean_costs=costs, before_cost=before_cost, dEP=(costs["E"]-costs["P"])/before_cost,
                dEN=(costs["E"]-costs["N"])/before_cost)
