"""Real-checkpoint semantic checks. No optimizer or elite access."""
import types
from .common import Invalid, digest
from .rng import stream
from .adapter import (upstream, new_instance, destroy, Session, instance_state, restore_instance,
                      reference_repair, parameter_hash, check, routes)


def verify_actor(actor, manifest, inputs, ledger):
    np, torch, Actor, *_ = upstream()
    metadata = manifest["metadata"]
    before = parameter_hash(actor)
    # Validate the singleton-axis patch against untouched upstream for B>=2.
    original = Actor(torch.device("cpu"), hidden_size=actor.all_embed.embed.weight.shape[0])
    original.load_state_dict(actor.state_dict())
    with stream("verify", "pointer-shapes"), torch.no_grad():
        hidden = actor.all_embed.embed.weight.shape[0]
        all_hidden, origin_hidden = torch.rand(2, hidden, 7), torch.rand(2, hidden, 1)
        if not torch.equal(original.pointer(all_hidden, origin_hidden), actor.pointer(all_hidden, origin_hidden)):
            raise Invalid("Singleton patch changed multi-instance pointer output")
    records = []
    for ident in manifest["selected_ids"][:4]:
        call_ids = {}
        instance = new_instance(inputs[ident], metadata)
        instance.create_initial_solution()
        with stream(ident, "verify-destroy"):
            destroy(instance, metadata)
        residual = instance_state(instance)
        reference = restore_instance(residual)
        captured = []
        action = reference.do_action
        def capture(this, origin, target):
            m = this.nn_input_idx_to_tour
            captured.append((int(m[origin][0][m[origin][1]][0]),
                             0 if target == 0 else int(m[target][0][m[target][1]][0])))
            return action(origin, target)
        reference.do_action = types.MethodType(capture, reference)
        call_ids["reference"] = ledger.charge("semantic_full", "verify", [ident, "reference"])
        with stream(ident, "verify-parity"):
            ptr, logp, _ = reference_repair(reference, actor)
        call_ids["adapted"] = ledger.charge("semantic_full", "verify", [ident, "adapted"])
        with stream(ident, "verify-parity"):
            session = Session(restore_instance(residual), actor, metadata)
            actions, cost = session.finish()
        if (captured != [(a["origin"], a["target"]) for a in actions] or
                ptr[0].tolist() != [a["target_index"] for a in actions] or
                routes(reference) != routes(session.instance) or
                check(reference, metadata["round_distances"]) != cost):
            raise Invalid(f"Original/adapted action or solution mismatch: {ident}")
        if not actions[-1]["terminal"] or not all(a["log_probability"] <= 1e-6 for a in actions):
            raise Invalid("Terminal autonomous action missing")
        if any(abs(float(logp[0, i])-a["log_probability"]) > 2e-5 for i, a in enumerate(actions[:-1])):
            raise Invalid("Nonterminal log probabilities differ from original")
        # Pick a middle decision without consulting elites or outcomes.
        with stream(ident, "verify-parity"):
            session = Session(restore_instance(residual), actor, metadata)
            for i in range(max(1, len(actions)//2)):
                d = session.distribution()
                if i == max(1, len(actions)//2)-1:
                    snapshot, decision = session.snapshot(), d
                    break
                session.apply(d, session.sample(d))
        before_snapshot = digest(snapshot)
        replays = []
        for replicate in range(2):
            role = f"replay-{replicate}"
            call_ids[role] = ledger.charge("semantic_partial", "verify", [ident, role])
            with stream(ident, "verify-replay"):
                restored = Session.restore(snapshot, actor, metadata)
                forced = restored.apply(decision, restored.sample(decision))
                tail, final_cost = restored.finish()
                replays.append(dict(forced=forced, tail=tail, cost=final_cost,
                                    final_state=restored.snapshot()))
        if digest(replays[0]) != digest(replays[1]) or digest(snapshot) != before_snapshot:
            raise Invalid("Snapshot replay or branch isolation failed")
        records.append(dict(instance_id=ident, reference_parity=True, snapshot_roundtrip=True,
                            replay=True, branch_isolation=True, terminal_record=True,
                            actions=len(actions), cost=cost, call_ids=call_ids))
    after = parameter_hash(actor)
    if before != after or any(p.requires_grad or p.grad is not None for p in actor.parameters()):
        raise Invalid("Actor parameters/gradient state changed")
    from .evidence import CHECKS_SCHEMA_VERSION
    return dict(checks_schema_version=CHECKS_SCHEMA_VERSION, identity_sha256=digest(manifest),
                instance_ids=manifest["selected_ids"][:4], passed=True, records=records,
                parameter_before_sha256=before, parameter_after_sha256=after, parameters_frozen=True,
                pointer_patch_batch2_parity=True, cpu_only=True)
