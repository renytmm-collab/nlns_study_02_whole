"""Synthetic evidence records only; these fixtures are never solver results."""
from frozen_repair.common import atomic_json, digest
from frozen_repair.config import PROTOCOL_VERSION, INITIALIZATION
from frozen_repair.evidence import CHECK_FIELDS, ROLES
from frozen_repair.ledger import Ledger


def create_run(path, finalized_tasks=4, checks=True):
    (path / "raw" / "calls").mkdir(parents=True)
    ids = ["i0", "i1", "i2", "i3"]
    tasks = ["t0", "t1", "t2", "t3"]
    identity = dict(scope="synthetic reliability fixture, no solver evidence", selected_ids=ids,
                    config=dict(protocol_version=PROTOCOL_VERSION), protocol_version=PROTOCOL_VERSION,
                    initialization=INITIALIZATION, model=dict(parameter_sha256="a"*64))
    limits = dict(evolution_full=80, natural_full=48, partial=288, semantic_full=64, semantic_partial=64)
    manifest = dict(kind="smoke", protocol_version=PROTOCOL_VERSION, ledger_schema_version=2,
                    identity=identity, identity_sha256=digest(identity), task_ids=tasks, instance_ids=ids,
                    limits=limits, smoke_proof=None)
    atomic_json(path / "manifest.json", manifest)
    ledger = Ledger(path, limits, expected_tasks=tasks)
    for i in range(finalized_tasks):
        ledger.complete(tasks[i], dict(instance_id=ids[i], states=[]))
    records = []
    if checks:
        for ident in ids:
            calls = {role: ledger.charge(kind, "verify", [ident, role]) for role, kind in ROLES}
            records.append(dict(instance_id=ident, call_ids=calls, **{k: True for k in CHECK_FIELDS}))
        atomic_json(path / "checks.json", dict(checks_schema_version=1, identity_sha256=digest(identity),
                    instance_ids=ids, records=records, parameter_before_sha256="a"*64,
                    parameter_after_sha256="a"*64, passed=True, pointer_patch_batch2_parity=True,
                    cpu_only=True, parameters_frozen=True))
        ledger.commit_checks()
    ledger.close()
    return manifest
