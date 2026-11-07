"""Training-only privileged diagnostics. The actor never receives this object."""
import statistics
from .common import edge, edges, distance, Invalid


class Diagnostic:
    def __init__(self, elite_routes, previous_routes, x):
        self.elite, self.previous = edges(elite_routes), edges(previous_routes)
        self.x = x

    def qualify(self, d):
        u = d["origin"]
        star = [i for i in d["legal"] if edge(u, d["mapping"][i]) in self.elite]
        new = [i for i in star if edge(u, d["mapping"][i]) not in self.previous]
        if not set(new) <= set(star) <= set(d["legal"]):
            raise Invalid("Vnew subset violation")
        return dict(Vstar=star, Vnew=new,
                    mass_star=sum(d["probabilities"][i] for i in star),
                    mass_new=sum(d["probabilities"][i] for i in new))

    def annotate(self, trajectories, before_cost):
        costs = [t["cost"] for t in trajectories]
        advantages = [(len(costs) * c - sum(costs)) / (len(costs) - 1) for c in costs]
        if abs(sum(advantages)) > 1e-8 * max(1, abs(before_cost)):
            raise Invalid("RLOO advantages do not sum to zero")
        epsilon = 1e-8 * max(1, abs(before_cost))
        for t, advantage in zip(trajectories, advantages):
            final, seen = edges(t["routes"]), set()
            q = int(advantage < -epsilon)
            t.update(advantage=advantage, q=q, epsilon=epsilon)
            for a in t["actions"]:
                autonomous = len(a["legal"]) > 1
                e = edge(a["origin"], a["target"])
                n = int(autonomous and e is not None and e in final and e not in self.previous and e not in seen)
                h = int(e in self.elite)
                if e:
                    seen.add(e)
                a.update(autonomous=autonomous, n=n, h=h, q=q, nh=n*h, qnh=q*n*h,
                         coefficient=0.5 * advantage * q * n * h)
                ids = self.x["customer_ids"]
                a["origin_original_id"], a["target_original_id"] = ids[a["origin"]], ids[a["target"]]
                a["legal_original_ids"] = [ids[a["mapping"][i]] for i in a["legal"]]
                if e:
                    u, v = a["origin"], a["target"]
                    xy = self.x["original_locations"]
                    order = sorted((w for w in range(1, len(ids)) if w != u),
                                   key=lambda w: (distance(xy, u, w), ids[w]))
                    a["edge_length"] = distance(xy, u, v)
                    a["nearest_neighbor_rank"] = order.index(v) + 1
                else:
                    a["edge_length"], a["nearest_neighbor_rank"] = None, None


def natural_summary(trajectories):
    """Primary denominator: multiple legal targets. Keep every raw policy step."""
    scoped = [dict(t, actions=[a for a in t["actions"] if len(a["legal"]) > 1]) for t in trajectories]
    summary = _action_summary(scoped)
    summary["all_steps"] = _action_summary(trajectories)
    summary["total_steps"] = sum(len(t["actions"]) for t in trajectories)
    summary["deterministic_steps"] = summary["total_steps"] - summary["actions"]
    for stats, original in zip(summary["trajectories"], trajectories):
        stats["total_steps"] = len(original["actions"])
        stats["deterministic_steps"] = stats["total_steps"] - stats["actions"]
    actions = [a for t in scoped for a in t["actions"]]
    summary["new_edge_geometry"] = {}
    for label, h in (("elite", 1), ("nonelite", 0)):
        subset = [a for a in actions if a["n"] == 1 and a["h"] == h]
        summary["new_edge_geometry"][label] = dict(
            edge_lengths=[a["edge_length"] for a in subset],
            nearest_neighbor_ranks=[a["nearest_neighbor_rank"] for a in subset])
    return summary


def _action_summary(trajectories):
    actions = [a for t in trajectories for a in t["actions"]]
    def collection(name, mass):
        subset = [a for a in actions if a[name]]
        return dict(nonempty=sum(bool(a[name]) for a in actions), denominator=len(actions),
                    sizes=[len(a[name]) for a in actions],
                    conditional_masses=[a[mass] for a in subset],
                    zero_filled_masses=[a[mass] for a in actions])
    good = [sum(a["nh"] for a in t["actions"]) for t in trajectories if t["q"]]
    other = [sum(a["nh"] for a in t["actions"]) for t in trajectories if not t["q"]]
    def difference(normalizer):
        groups = {0: [], 1: []}
        for t in trajectories:
            denominator = normalizer(t)
            if denominator:
                groups[t["q"]].append(sum(a["nh"] for a in t["actions"])/denominator)
        return statistics.mean(groups[1])-statistics.mean(groups[0]) if groups[1] and groups[0] else None
    return dict(actions=len(actions), Vstar=collection("Vstar", "mass_star"), Vnew=collection("Vnew", "mass_new"),
                n=sum(a["n"] for a in actions), nh=sum(a["nh"] for a in actions), qnh=sum(a["qnh"] for a in actions),
                any_qnh=any(a["qnh"] for a in actions),
                coefficients=[a["coefficient"] for a in actions],
                natural_good_vs_other_nh=(statistics.mean(good)-statistics.mean(other)) if good and other else None,
                natural_good_vs_other_nh_per_action=difference(lambda t: len(t["actions"])),
                natural_good_vs_other_nh_per_new=difference(lambda t: sum(a["n"] for a in t["actions"])),
                trajectories=[dict(q=t["q"], actions=len(t["actions"]),
                                   n=sum(a["n"] for a in t["actions"]), nh=sum(a["nh"] for a in t["actions"]),
                                   nh_per_action=(sum(a["nh"] for a in t["actions"])/len(t["actions"]))
                                   if t["actions"] else None,
                                   nh_per_new=(sum(a["nh"] for a in t["actions"])/sum(a["n"] for a in t["actions"]))
                                   if sum(a["n"] for a in t["actions"]) else None) for t in trajectories])
