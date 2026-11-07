"""One read-only validation path for resume, smoke reuse, finalization and reports."""
from pathlib import Path
from .common import Invalid, read_json, file_hash, digest, atomic_json
from .config import PROTOCOL_VERSION, budget_limits
from .ledger import Ledger

CHECKS_SCHEMA_VERSION = 1
CHECK_FIELDS = ("reference_parity", "snapshot_roundtrip", "replay", "branch_isolation", "terminal_record")
ROLES = (("reference", "semantic_full"), ("adapted", "semantic_full"),
         ("replay-0", "semantic_partial"), ("replay-1", "semantic_partial"))


def validate_checks(directory, identity, ledger):
    directory = Path(directory)
    committed = ledger.db.execute("SELECT hash FROM proofs WHERE name='checks'").fetchone()
    path = directory / "checks.json"
    if not committed or not path.is_file() or file_hash(path) != committed[0]:
        raise Invalid("Missing, uncommitted or damaged checks evidence; never overwrite/re-run it implicitly")
    c = read_json(path)
    expected = identity["selected_ids"][:4]
    parameter = identity["model"]["parameter_sha256"]
    if (c.get("checks_schema_version") != CHECKS_SCHEMA_VERSION or c.get("identity_sha256") != digest(identity) or
            c.get("instance_ids") != expected or len(expected) != 4 or len(set(expected)) != 4 or
            c.get("parameter_before_sha256") != parameter or c.get("parameter_after_sha256") != parameter or
            any(c.get(k) is not True for k in ("passed", "pointer_patch_batch2_parity", "cpu_only", "parameters_frozen"))):
        raise Invalid("Checks header, identity, instances or parameter evidence invalid")
    records = c.get("records", [])
    if len(records) != 4 or [r.get("instance_id") for r in records] != expected:
        raise Invalid("Checks must cover exactly the designated four instances")
    used = set()
    for r in records:
        if any(r.get(k) is not True for k in CHECK_FIELDS):
            raise Invalid("Incomplete/failed semantic checks")
        if set(r.get("call_ids", {})) != {role for role, _ in ROLES}:
            raise Invalid("Missing semantic check charged calls")
        for role, kind in ROLES:
            call_id = r["call_ids"][role]
            call = ledger.db.execute("SELECT kind,task,detail FROM calls WHERE id=?", (call_id,)).fetchone()
            import json
            if (type(call_id) is not int or call_id in used or call is None or call[:2] != (kind, "verify") or
                    json.loads(call[2]) != [r["instance_id"], role]):
                raise Invalid("Checks do not match unique charged semantic calls")
            used.add(call_id)
    return committed[0]


def validate_manifest(m):
    identity = m["identity"]
    if (m.get("protocol_version") != PROTOCOL_VERSION or identity.get("protocol_version") != PROTOCOL_VERSION or
            identity.get("config", {}).get("protocol_version") != PROTOCOL_VERSION or
            m.get("ledger_schema_version") != 2 or digest(identity) != m["identity_sha256"] or
            m["limits"] != budget_limits(m["kind"])):
        raise Invalid("Manifest identity/protocol/schema mismatch; no old-run migration")
    tasks, instances = m["task_ids"], m["instance_ids"]
    if (len(tasks) != len(set(tasks)) or len(instances) != len(set(instances)) or
            m["kind"] not in ("collect", "smoke", "verify") or
            (m["kind"] != "verify" and len(tasks) != len(instances)) or
            (m["kind"] == "verify" and tasks)):
        raise Invalid("Invalid manifest task/instance list")


def audit(directory):
    directory = Path(directory).resolve()
    m = read_json(directory / "manifest.json")
    validate_manifest(m)
    ledger = Ledger(directory, m["limits"], existing=True, expected_tasks=m["task_ids"], readonly=True)
    try:
        tasks = ledger.db.execute("SELECT id,status,hash,error FROM tasks ORDER BY id").fetchall()
        actual = {r[0] for r in tasks}
        expected = set(m["task_ids"])
        if actual - expected:
            raise Invalid("Unexpected ledger task IDs")
        if (directory / "failure.json").exists() or any(r[1] == "failed" for r in tasks):
            raise Invalid("Run contains a failure record")
        hashes = {r[0]: r[2] for r in tasks if r[1] == "completed"}
        expected_instances = dict(zip(m["task_ids"], m["instance_ids"]))
        for task in hashes:
            if read_json(directory / "raw" / (task+".json")).get("instance_id") != expected_instances[task]:
                raise Invalid("Raw instance ID does not match manifest task")
        proofs = ledger.db.execute("SELECT name,hash FROM proofs ORDER BY name").fetchall()
        if any(name != "checks" for name, _ in proofs):
            raise Invalid("Unexpected evidence record")
        checks_hash = None
        if m["kind"] in ("smoke", "verify"):
            if proofs or (directory / "checks.json").exists():
                checks_hash = validate_checks(directory, m["identity"], ledger)
        elif proofs or (directory / "checks.json").exists():
            raise Invalid("Collect must reference smoke evidence, not create local checks")
        smoke = None
        if m["kind"] == "collect":
            proof = m["smoke_proof"]
            smoke = smoke_proof(proof["path"], m["identity"])
            if smoke != proof:
                raise Invalid("Referenced smoke evidence changed")
        calls = ledger.db.execute("SELECT id,kind,task,started,detail FROM calls ORDER BY id").fetchall()
        cache = ledger.db.execute("SELECT id,hash,call_id FROM cache ORDER BY id").fetchall()
        bindings = dict(completion_schema_version=1, protocol_version=PROTOCOL_VERSION,
                        manifest_sha256=file_hash(directory / "manifest.json"), identity_sha256=m["identity_sha256"],
                        task_hashes=hashes, charged_calls=ledger.counts(), checks_sha256=checks_hash,
                        smoke_proof=smoke, parameter_sha256=m["identity"]["model"]["parameter_sha256"],
                        ledger_records_sha256=digest(dict(tasks=tasks, calls=calls, cache=cache, proofs=proofs)))
        complete_tasks = actual == expected and all(r[1] == "completed" for r in tasks)
        path = directory / "completion.json"
        if path.exists():
            done = read_json(path)
            if not complete_tasks or done != dict(status="COMPLETE", **bindings):
                raise Invalid("Completion credential or bound artifacts mismatch")
            if m["kind"] != "collect" and checks_hash is None:
                raise Invalid("Finalized run lacks semantic checks")
            state = "COMPLETE"
        else:
            state = "UNFINALIZED" if complete_tasks else "INCOMPLETE"
        return dict(status=state, bindings=bindings, manifest=m)
    finally:
        ledger.close()


def finalize(directory, parameter_sha256):
    a = audit(directory)
    if a["status"] == "INCOMPLETE":
        raise Invalid("Cannot finalize incomplete tasks")
    if parameter_sha256 != a["bindings"]["parameter_sha256"]:
        raise Invalid("Final model parameter hash mismatch")
    if a["manifest"]["kind"] != "collect" and a["bindings"]["checks_sha256"] is None:
        raise Invalid("Cannot finalize without committed checks")
    if a["status"] != "COMPLETE":
        atomic_json(Path(directory) / "completion.json", dict(status="COMPLETE", **a["bindings"]))
    return audit(directory)


def smoke_proof(directory, identity):
    path = Path(directory).resolve()
    a = audit(path)
    if (a["status"] != "COMPLETE" or a["manifest"]["kind"] != "smoke" or
            a["manifest"]["identity_sha256"] != digest(identity)):
        raise Invalid("Matching finalized smoke required")
    return dict(path=str(path), manifest_sha256=a["bindings"]["manifest_sha256"],
                checks_sha256=a["bindings"]["checks_sha256"], completion_sha256=file_hash(path / "completion.json"))
