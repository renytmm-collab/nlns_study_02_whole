from pathlib import Path
from .common import Blocked, read_json

PROTOCOL_VERSION = "frozen-repair-precheck-v2"
INITIALIZATION = dict(method="upstream VRPInstance.create_initial_solution",
                      ranking_formula="sum((locations[candidate] * locations[origin]) ** 2)",
                      limitation="Coordinate-product squared ranking, not Euclidean nearest-neighbor; conclusions apply to this initialization and its subsequent state chain")


def budget_limits(kind):
    n = 64 if kind == "collect" else 4
    return dict(evolution_full=n*20, natural_full=n*12, partial=n*3*3*8,
                semantic_full=64 if kind != "collect" else 0,
                semantic_partial=64 if kind != "collect" else 0)

FIXED = dict(instances=64, depths=[0, 5, 20], M=4, K=8, seed=20261105,
             bootstrap_seed=20261106, bootstrap_replicates=2000, delta=0.001, alpha=0.5)


def load(path):
    path = Path(path).resolve()
    c = read_json(path)
    allowed = {"checkpoint", "checkpoint_sha256", "checkpoint_metadata", "dataset", "device", "threads", "protocol", "protocol_version"}
    if set(c) - allowed:
        raise Blocked(f"Unknown config fields: {sorted(set(c) - allowed)}")
    if c.get("protocol", FIXED) != FIXED:
        raise Blocked("Fixed exploratory protocol changed")
    c["protocol"] = FIXED.copy()
    if c.get("protocol_version") != PROTOCOL_VERSION:
        raise Blocked(f"protocol_version must explicitly equal {PROTOCOL_VERSION}")
    if c.get("device", "cpu") != "cpu":
        raise Blocked("v1 supports CPU only; CUDA parity has not been verified")
    c["device"] = "cpu"
    c["threads"] = c.get("threads", 1)
    if type(c["threads"]) is not int or c["threads"] < 1:
        raise Blocked("threads must be a positive integer")
    for key in ("checkpoint", "checkpoint_metadata", "dataset"):
        if not c.get(key):
            raise Blocked(f"Missing config path: {key}")
        p = Path(c[key]).expanduser()
        c[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    if len(c.get("checkpoint_sha256", "")) != 64:
        raise Blocked("checkpoint_sha256 must be supplied independently")
    return c
