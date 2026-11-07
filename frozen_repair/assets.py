"""Explicit JSON contracts. Elite routes are not returned to the actor adapter."""
import math
from pathlib import Path
from .common import Blocked, Invalid, read_json, file_hash, digest, validate_routes, routes_cost
from .rng import local
from .identity import source_identity, environment, git_state


def inspect(config):
    c = config
    for key in ("checkpoint", "checkpoint_metadata", "dataset"):
        if not Path(c[key]).is_file():
            raise Blocked(f"Missing {key}: {c[key]}")
    ck_hash = file_hash(c["checkpoint"])
    if ck_hash != c["checkpoint_sha256"]:
        raise Blocked("Checkpoint SHA256 mismatch")
    meta = read_json(c["checkpoint_metadata"])
    required = ("checkpoint_sha256", "problem", "nb_customers", "distribution", "capacity",
                "coordinate_scale", "destroy_operation", "p_destruction", "round_distances", "evidence")
    if any(k not in meta or meta[k] is None for k in required):
        raise Blocked("Checkpoint metadata incomplete; do not guess training provenance")
    for field in ("distribution", "evidence"):
        value = meta[field]
        if not isinstance(value, str) or not value.strip() or value.strip().lower() in ("unknown", "未知") or "<" in value:
            raise Blocked(f"Unresolved checkpoint provenance: {field}")
    if meta["checkpoint_sha256"] != ck_hash or meta["problem"] != "CVRP":
        raise Blocked("Checkpoint identity/problem mismatch")
    if (meta["destroy_operation"] not in ("R", "P", "T") or
            not 0 < meta["p_destruction"] <= 1 or not meta["evidence"] or
            type(meta["round_distances"]) is not bool or type(meta["nb_customers"]) is not int or
            meta["nb_customers"] < 2 or int(meta["nb_customers"] * meta["p_destruction"]) < 1):
        raise Blocked("Invalid checkpoint/Destroy metadata")
    dataset = read_json(c["dataset"])
    if dataset.get("schema") != "frozen-repair-assets-v1":
        raise Blocked("Unsupported dataset schema")
    records = dataset.get("instances", [])
    ids, fingerprints, inputs, elites, assets = set(), set(), {}, {}, []
    base = Path(c["dataset"]).parent
    for entry in records:
        ident = entry.get("instance_id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise Blocked("Missing/duplicate instance ID")
        ids.add(ident)
        if entry.get("split") != "train" or not entry.get("split_evidence"):
            raise Blocked(f"No training membership evidence: {ident}")
        loaded, source_info = {}, {}
        for kind in ("instance", "elite"):
            p = (base / entry[kind]).resolve()
            h = file_hash(p)
            if h != entry[kind + "_sha256"]:
                raise Blocked(f"Asset hash mismatch: {ident}/{kind}")
            loaded[kind] = read_json(p)
            source_info[kind] = dict(path=str(p), sha256=h)
        x, e = loaded["instance"], loaded["elite"]
        n = meta["nb_customers"]
        if x.get("instance_id") != ident or e.get("instance_id") != ident:
            raise Blocked("Instance/elite ID mismatch")
        if x.get("distribution") != meta["distribution"] or x.get("capacity") != meta["capacity"]:
            raise Blocked(f"Distribution/capacity mismatch: {ident}")
        xy, demands, original_ids = x["original_locations"], x["demand"], x["customer_ids"]
        if (len(xy) != n + 1 or len(demands) != n + 1 or len(original_ids) != n + 1 or
                original_ids[0] != 0 or len(set(original_ids)) != n + 1 or
                any(type(i) is not int or i < 0 for i in original_ids)):
            raise Blocked("Invalid depot/customer mapping or problem size")
        if any(len(p) != 2 or any(not math.isfinite(v) for v in p) for p in xy):
            raise Blocked("Invalid coordinates")
        if (demands[0] != 0 or any(type(v) is not int or v <= 0 or v > x["capacity"] for v in demands[1:]) or
                x["capacity"] <= 0 or not math.isfinite(meta["coordinate_scale"]) or meta["coordinate_scale"] <= 0):
            raise Blocked("Invalid integer CVRP demand/capacity/scale")
        scale = meta["coordinate_scale"]
        if any(v < 0 or v / scale > 1 for p in xy for v in p):
            raise Blocked("Coordinates incompatible with declared NLNS input scale")
        if e.get("instance_sha256") != source_info["instance"]["sha256"] or not e.get("source"):
            raise Blocked("Elite must bind exact instance file and identify source")
        mapping = {v: i for i, v in enumerate(original_ids)}
        try:
            routes = [[mapping[v] for v in r] for r in e["routes"]]
            validate_routes(routes, demands, x["capacity"])
        except (KeyError, Invalid) as exc:
            raise Blocked(f"Elite infeasible/mapping failure: {ident}: {exc}") from exc
        fingerprint = digest(dict(depot=xy[0], customers=sorted([*p, d] for p, d in zip(xy[1:], demands[1:])),
                                  capacity=x["capacity"]))
        if fingerprint in fingerprints:
            raise Blocked("Duplicate instance content, even with different IDs")
        fingerprints.add(fingerprint)
        inputs[ident] = x
        elites[ident] = routes
        assets.append(dict(instance_id=ident, sources=source_info, fingerprint=fingerprint,
                           split="train", split_evidence=entry["split_evidence"], elite_source=e["source"],
                           elite_cost_recomputed=routes_cost(routes, xy, meta["round_distances"])))
    if len(ids) < 64:
        raise Blocked(f"Need 64 distinct compatible training instances; found {len(ids)}")
    selected = local("asset-selection").sample(sorted(ids), 64)
    env = environment()
    if not env["numpy"] or not env["torch"]:
        raise Blocked("NumPy/PyTorch unavailable; no dependency installation is automatic")
    from .config import PROTOCOL_VERSION, INITIALIZATION
    manifest = dict(protocol_version=PROTOCOL_VERSION, initialization=INITIALIZATION,
                    config=c, metadata=meta, checkpoint_sha256=ck_hash,
                    metadata_sha256=file_hash(c["checkpoint_metadata"]), dataset_sha256=file_hash(c["dataset"]),
                    compatible_universe=assets, selected_ids=selected, source_hashes=source_identity(),
                    environment=env, git=git_state(),
                    rng=dict(algorithm="sha256(canonical([master,*task_parts])) first 32 bits", master=20261105,
                             bootstrap=20261106),
                    scope="Frozen network; training-data diagnostics; no training or full-search benefit evaluation")
    return manifest, inputs, elites
