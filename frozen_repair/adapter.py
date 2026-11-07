"""Upstream actor/environment bridge. This module has no elite input."""
import copy
import hashlib
import math
import sys
import types
from pathlib import Path
from .common import ROOT, Blocked, Invalid, digest, validate_routes, routes_cost


def upstream():
    path = str(ROOT / "vendor" / "NLNS")
    if path not in sys.path:
        sys.path.insert(0, path)
    import numpy as np
    import torch
    from actor import VrpActorModel
    from vrp.vrp_problem import VRPInstance, get_mask
    return np, torch, VrpActorModel, VRPInstance, get_mask


def parameter_hash(actor):
    h = hashlib.sha256()
    for name, p in sorted(actor.state_dict().items()):
        h.update(name.encode())
        h.update(str(p.dtype).encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_actor(config, metadata):
    np, torch, Actor, _, _ = upstream()
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    try:
        checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Blocked(f"Safe checkpoint loading failed ({type(exc).__name__}); no unsafe pickle fallback") from exc
    if not isinstance(checkpoint, dict) or "parameters" not in checkpoint:
        raise Blocked("Expected official NLNS checkpoint dictionary with parameters")
    for key in ("destroy_operation", "p_destruction"):
        if checkpoint.get(key) != metadata[key]:
            raise Blocked(f"Checkpoint disagrees with declared {key}")
    params = checkpoint["parameters"]
    try:
        hidden = params["all_embed.embed.weight"].shape[0]
        actor = Actor(torch.device("cpu"), hidden_size=hidden)
        actor.load_state_dict(params, strict=True)
    except Exception as exc:
        raise Blocked("Checkpoint is not compatible with the official actor") from exc
    # Original Pointer.forward uses squeeze() on [B,H,1] and [B,1,H].
    # Only remove the intended singleton axes, leaving B=1 intact.
    def pointer_forward(self, all_hidden, origin_hidden):
        attn = self.encoder_attn(all_hidden, origin_hidden.transpose(2, 1).squeeze(1))
        context = attn.bmm(all_hidden.permute(0, 2, 1))
        z = torch.cat((origin_hidden.squeeze(2), context.squeeze(1)), dim=1)
        z = torch.nn.functional.relu(self.fc1(z))
        z = torch.nn.functional.relu(self.fc2(z)).unsqueeze(2).expand_as(all_hidden)
        return torch.bmm(self.v.expand(all_hidden.size(0), -1, -1), torch.tanh(all_hidden + z)).squeeze(1)
    actor.pointer.forward = types.MethodType(pointer_forward, actor.pointer)
    actor.eval()
    actor.requires_grad_(False)
    return actor, dict(hidden_size=hidden, parameter_sha256=parameter_hash(actor),
                       checkpoint_keys=sorted(checkpoint), decoding="categorical", network_mode="eval",
                       compatibility=["Pointer.squeeze removes only known singleton axes",
                                      "Reference repair np.long replaced by np.int64 in memory"])


def new_instance(x, metadata):
    np, _, _, Instance, _ = upstream()
    original = np.array(x["original_locations"], dtype=float)
    return Instance(len(x["demand"]) - 1, original / metadata["coordinate_scale"], original,
                    np.array(x["demand"], dtype=int), x["capacity"], use_cost_memory=False)


def routes(instance):
    return [[int(e[0]) for e in t] for t in instance.solution]


def check(instance, rounded):
    rr = routes(instance)
    validate_routes(rr, instance.demand.tolist(), instance.capacity)
    for tour in instance.solution:
        for v, served, _ in tour:
            if served != (0 if v == 0 else instance.demand[v]):
                raise Invalid("Served demand mismatch")
    cost = float(instance.get_costs(rounded))
    independent = routes_cost(rr, instance.original_locations.tolist(), rounded)
    if not math.isfinite(cost) or cost <= 0 or not math.isclose(cost, independent, rel_tol=1e-10, abs_tol=1e-9):
        raise Invalid("Final cost failed independent recomputation")
    return cost


def destroy(instance, metadata):
    methods = dict(R="destroy_random", P="destroy_point_based", T="destroy_tour_based")
    getattr(instance, methods[metadata["destroy_operation"]])(metadata["p_destruction"])
    if not instance.incomplete_tours:
        raise Invalid("Configured Destroy produced no residual problem")


def instance_state(instance):
    """JSON graph encoding preserves aliases; upstream __deepcopy__ drops these."""
    pool, indexes = [], {}
    def idx(t):
        if id(t) not in indexes:
            indexes[id(t)] = len(pool)
            pool.append([[int(a), int(b), None if c is None else int(c)] for a, b, c in t])
        return indexes[id(t)]
    solution = [idx(t) for t in instance.solution]
    incomplete = None if instance.incomplete_tours is None else [idx(t) for t in instance.incomplete_tours]
    mapping = None if instance.nn_input_idx_to_tour is None else [
        None if m is None else [idx(m[0]), int(m[1])] for m in instance.nn_input_idx_to_tour]
    return dict(nb_customers=instance.nb_customers, locations=instance.locations.tolist(),
                original_locations=instance.original_locations.tolist(), demand=instance.demand.tolist(),
                capacity=instance.capacity, pool=pool, solution=solution, incomplete=incomplete,
                mapping=mapping, open=None if instance.open_nn_input_idx is None else list(instance.open_nn_input_idx))


def restore_instance(state):
    np, _, _, Instance, _ = upstream()
    s = copy.deepcopy(state)
    inst = Instance(s["nb_customers"], np.array(s["locations"]), np.array(s["original_locations"]),
                    np.array(s["demand"], dtype=int), s["capacity"], use_cost_memory=False)
    inst.solution = [s["pool"][i] for i in s["solution"]]
    inst.incomplete_tours = None if s["incomplete"] is None else [s["pool"][i] for i in s["incomplete"]]
    inst.nn_input_idx_to_tour = None if s["mapping"] is None else [
        None if m is None else [s["pool"][m[0]], m[1]] for m in s["mapping"]]
    inst.open_nn_input_idx = s["open"]
    return inst


class Session:
    def __init__(self, instance, actor, metadata):
        np, torch, _, _, _ = upstream()
        self.instance, self.actor, self.metadata = instance, actor, metadata
        count = instance.get_max_nb_input_points()
        static, dynamic = instance.get_network_input(count)
        self.static = torch.from_numpy(static[None]).float()
        self.dynamic = torch.from_numpy(dynamic[None].astype(np.int64)).long()
        self.origin, self.step = 0, 0

    @property
    def done(self):
        return not self.instance.open_nn_input_idx

    def distribution(self):
        np, torch, _, _, get_mask = upstream()
        if self.done:
            raise Invalid("Decision requested after termination")
        if self.origin == 0:
            self.origin = int(np.random.choice(self.instance.open_nn_input_idx, 1).item())
        mask = get_mask(np.array([self.origin]), self.dynamic, [self.instance],
                        types.SimpleNamespace(split_delivery=False, device="cpu"), self.instance.capacity).float()
        dynamic = self.dynamic.float()
        dynamic[:, :, 0] /= float(self.instance.capacity)
        with torch.no_grad():
            logits = self.actor(self.static, dynamic, self.static[:, self.origin], dynamic[:, self.origin], mask)
            probs = torch.softmax(logits + mask.log(), dim=1)[0]
        if not torch.isfinite(probs).all() or abs(float(probs.sum()) - 1) > 1e-6:
            raise Invalid("Nonfinite/unnormalized policy")
        mapping = []
        for i, m in enumerate(self.instance.nn_input_idx_to_tour):
            mapping.append(0 if i == 0 else (None if m is None else int(m[0][m[1]][0])))
        return dict(step=self.step, origin_index=self.origin, origin=mapping[self.origin],
                    mapping=mapping, legal=[i for i, v in enumerate(mask[0].tolist()) if v],
                    probabilities=probs.tolist())

    def apply(self, decision, target):
        np, torch, *_ = upstream()
        if decision["origin_index"] != self.origin or target not in decision["legal"]:
            raise Invalid("Illegal/stale forced action")
        p = decision["probabilities"][target]
        if not math.isfinite(p) or p < 0:
            raise Invalid("Invalid action probability")
        # Record endpoint identity BEFORE do_action mutates/reverses tours.
        categorical = torch.distributions.Categorical(torch.tensor(decision["probabilities"], dtype=torch.float32))
        logp = float(categorical.log_prob(torch.tensor(target)))
        result = dict(decision, target_index=target, target=decision["mapping"][target],
                      action_probability=p, log_probability=logp)
        updates, following = self.instance.do_action(self.origin, target)
        if updates:
            arr = np.array(updates, dtype=np.int64)
            self.dynamic[0, arr[:, 0]] = torch.from_numpy(arr[:, 1:]).long()
        self.origin = 0 if self.done else int(following)
        self.step += 1
        if self.step > 4 * self.instance.nb_customers + 10:
            raise Invalid("Repair exceeded semantic step bound")
        result["terminal"] = self.done
        return result

    def sample(self, decision):
        _, torch, *_ = upstream()
        return int(torch.distributions.Categorical(torch.tensor(decision["probabilities"], dtype=torch.float32)).sample())

    def finish(self):
        actions = []
        while not self.done:
            d = self.distribution()
            actions.append(self.apply(d, self.sample(d)))
        return actions, check(self.instance, self.metadata["round_distances"])

    def snapshot(self):
        return dict(instance=instance_state(self.instance), static=self.static.tolist(),
                    dynamic=self.dynamic.tolist(), origin=self.origin, step=self.step)

    @classmethod
    def restore(cls, snap, actor, metadata):
        _, torch, *_ = upstream()
        obj = cls.__new__(cls)
        obj.instance = restore_instance(snap["instance"])
        obj.actor, obj.metadata = actor, metadata
        obj.static = torch.tensor(snap["static"], dtype=torch.float32)
        obj.dynamic = torch.tensor(snap["dynamic"], dtype=torch.int64)
        obj.origin, obj.step = snap["origin"], snap["step"]
        if digest(obj.snapshot()) != digest(snap):
            raise Invalid("Snapshot roundtrip mismatch")
        return obj


def reference_repair(instance, actor):
    """Execute official loop, changing only removed NumPy alias in memory."""
    module = types.ModuleType("nlns_reference_repair")
    source = (ROOT / "vendor/NLNS/repair.py").read_text(encoding="utf-8")
    exec(compile(source.replace("dtype=np.long", "dtype=np.int64"), "upstream/repair.py", "exec"), module.__dict__)
    actor.train()  # Upstream selects categorical based on this flag; actor has no dropout/BN.
    _, torch, *_ = upstream()
    try:
        with torch.no_grad():
            return module.repair([instance], actor, types.SimpleNamespace(device="cpu", split_delivery=False))
    finally:
        actor.eval()
