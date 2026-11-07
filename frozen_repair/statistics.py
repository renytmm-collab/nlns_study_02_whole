import random
import statistics


def mean(values):
    return statistics.mean(values) if values else None


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*p
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (index-lo)*(values[hi]-values[lo])


def distribution(values):
    return dict(n=len(values), mean=mean(values), min=min(values) if values else None,
                median=quantile(values, .5), p90=quantile(values, .9), p95=quantile(values, .95),
                max=max(values) if values else None)


def classify(interval, delta=.001):
    if interval is None:
        return "NO_ELIGIBLE_POINTS"
    low, high = interval
    if low > 0:
        return "HARM_SUPPORTED"
    if high < -delta:
        return "MEANINGFUL_LOCAL_ADVANTAGE_SUPPORTED"
    if low > -delta:
        return "NO_MEANINGFUL_ADVANTAGE_DETECTED"
    return "INCONCLUSIVE"


def clustered(points, key, total_instances=64, divergence=False):
    groups = {}
    comparison = "P" if key == "dEP" else "N"
    for p in points:
        if divergence and p["actions"]["E"] == p["actions"][comparison]:
            continue
        groups.setdefault(p["instance_id"], []).append(p[key])
    values = [statistics.mean(v) for _, v in sorted(groups.items())]
    interval = None
    if values:
        rng = random.Random(20261106)
        samples = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)]
        interval = [quantile(samples, .025), quantile(samples, .975)]
    label = classify(interval)
    if values and len(values) < 32:
        label = "INCONCLUSIVE"
    return dict(estimate=mean(values), ci95=interval, comparison_label=label,
                descriptive_only=len(values) < 32, available_instances=len(values),
                total_instances=total_instances, points=sum(map(len, groups.values())),
                instance_means={i: statistics.mean(v) for i, v in sorted(groups.items())},
                scope="divergent actions only" if divergence else "all sampled eligible points, including identical actions")
