import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "8fd1e83faeb0ecff7986c4e5993f7398e6b6b6f8"


class Blocked(RuntimeError):
    pass


class BudgetExceeded(Blocked):
    pass


class Invalid(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def edge(u, v):
    return tuple(sorted((u, v))) if u != 0 and v != 0 and u != v else None


def edges(routes):
    return {e for route in routes for u, v in zip(route, route[1:]) if (e := edge(u, v)) is not None}


def distance(xy, u, v, rounded=False):
    d = math.hypot(xy[u][0] - xy[v][0], xy[u][1] - xy[v][1])
    return float(round(d)) if rounded else d


def routes_cost(routes, xy, rounded=False):
    return sum(distance(xy, u, v, rounded) for r in routes for u, v in zip(r, r[1:]))


def validate_routes(routes, demand, capacity):
    visited = []
    for r in routes:
        if not r or r[0] != 0 or r[-1] != 0:
            raise Invalid("Incomplete route")
        if any(type(v) is not int or not 0 <= v < len(demand) for v in r):
            raise Invalid("Invalid customer index")
        if 0 in r[1:-1]:
            raise Invalid("Interior depot")
        visited.extend(r[1:-1])
        if sum(demand[v] for v in r) > capacity:
            raise Invalid("Capacity exceeded")
    if sorted(visited) != list(range(1, len(demand))):
        raise Invalid("Customer coverage/duplicates")
