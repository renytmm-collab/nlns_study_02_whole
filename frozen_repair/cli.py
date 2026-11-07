import argparse
import json
import sys
import time
import sqlite3
from pathlib import Path
from .common import Blocked, BudgetExceeded, Invalid, read_json, atomic_json, digest, file_hash
from .config import load
from .config import PROTOCOL_VERSION, budget_limits
from .assets import inspect
from .ledger import Ledger, run_lock, status
from .evidence import smoke_proof, validate_checks, validate_manifest, audit, finalize


def output(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


def inspect_full(config):
    from .adapter import load_actor
    identity, inputs, elites = inspect(config)
    actor, model = load_actor(config, identity["metadata"])
    identity["model"] = model
    return identity, inputs, elites, actor


def limits(kind):
    return budget_limits(kind)


def execute(kind, config, directory, resume=False, smoke_run=None):
    from .adapter import parameter_hash
    from .verification import verify_actor
    from .experiment import run_instance
    from .reporting import generate
    identity, inputs, elites, actor = inspect_full(config)
    identity_hash = digest(identity)
    directory = Path(directory).resolve()
    if resume:
        manifest = read_json(directory / "manifest.json")
        if manifest["identity_sha256"] != identity_hash or digest(manifest["identity"]) != identity_hash:
            raise Blocked("Code/config/assets/environment changed; refuse joining old run")
        kind = manifest["kind"]
        validate_manifest(manifest)
        if (directory / "failure.json").exists():
            raise Invalid("INVALID run is terminal; fix implementation and use a new run directory")
    else:
        if directory.exists():
            raise Blocked("New run directory must not exist; use resume for existing runs")
        proof = smoke_proof(smoke_run, identity) if kind == "collect" else None
        selected = identity["selected_ids"] if kind == "collect" else identity["selected_ids"][:4]
        tasks = [] if kind == "verify" else [digest([i])[:24] for i in selected]
        manifest = dict(protocol_version=PROTOCOL_VERSION, ledger_schema_version=2,
                        identity=identity, identity_sha256=identity_hash, kind=kind, task_ids=tasks,
                        instance_ids=selected, limits=limits(kind), smoke_proof=proof)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "raw").mkdir()
        (directory / "raw" / "calls").mkdir()
        atomic_json(directory / "manifest.json", manifest)
    with run_lock(directory):
        ledger = Ledger(directory, manifest["limits"], existing=resume, expected_tasks=manifest["task_ids"])
        current = "verify"
        try:
            actual = {r[0] for r in ledger.db.execute("SELECT id FROM tasks")}
            if actual != set(manifest["task_ids"]):
                raise Invalid("Ledger task list differs from manifest; do not recreate missing rows")
            if resume:
                audit(directory)
            if kind != "collect":
                proof_path = directory / "checks.json"
                proof_row = ledger.db.execute("SELECT hash FROM proofs WHERE name='checks'").fetchone()
                if not proof_path.exists() and not proof_row:
                    if ledger.db.execute("SELECT 1 FROM calls WHERE task='verify' LIMIT 1").fetchone():
                        raise Invalid("Interrupted uncommitted checks; use a new run, do not silently re-run evidence")
                    output(dict(stage="semantic-verification", instances=4))
                    checks = verify_actor(actor, identity, inputs, ledger)
                    atomic_json(proof_path, checks)
                    ledger.commit_checks()
                validate_checks(directory, identity, ledger)
            else:
                proof = manifest["smoke_proof"]
                if smoke_proof(proof["path"], identity) != proof:
                    raise Invalid("Smoke evidence changed")
            for ident, task in zip(manifest["instance_ids"], manifest["task_ids"]):
                if ledger.completed(task):
                    continue
                current = task
                ledger.begin(task)
                output(dict(stage=kind, instance_id=ident, task_id=task, charged_calls=ledger.counts()))
                result = run_instance(ident, inputs[ident], elites[ident], actor, identity["metadata"], ledger, task)
                if parameter_hash(actor) != identity["model"]["parameter_sha256"]:
                    raise Invalid("Frozen actor parameters changed")
                ledger.complete(task, result)
                output(dict(stage="instance-completed", instance_id=ident, charged_calls=ledger.counts()))
            audited = audit(directory)
            if audited["status"] == "INCOMPLETE":
                raise Invalid("Cannot finalize incomplete task set")
            # Detect assets/source edits during the run, including mutable input files.
            fresh_identity, _, _, _ = inspect_full(config)
            if digest(fresh_identity) != identity_hash:
                raise Invalid("Inputs or implementation changed during run")
            finalize(directory, parameter_hash(actor))
        except (KeyboardInterrupt, Blocked) as exc:
            atomic_json(directory / "interruption.json", dict(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt)
                        else "BUDGET_OR_EXTERNAL_BLOCK", task=current, reason=str(exc), charged_calls=ledger.counts()))
            raise
        except Exception as exc:
            if current != "verify":
                ledger.fail(current, str(exc))
            atomic_json(directory / "failure.json", dict(status="INVALID", task=current,
                        error_type=type(exc).__name__, reason=str(exc), charged_calls=ledger.counts()))
            raise
        finally:
            ledger.close()
        # Reporting is a derived artifact. Its failure must not invalidate a finalized run.
        if kind != "verify":
            try:
                result = generate(directory)
            except Exception as exc:
                output(dict(status="COMPLETE", report_status="FAILED", reason=str(exc),
                            recovery="Run report again; completion evidence is preserved", run_dir=str(directory)))
                return False
            if result["status"] != "COMPLETE":
                output(dict(status=result["status"], run_dir=str(directory), report_status="EVIDENCE_REJECTED"))
                return False
        output(dict(status="COMPLETE", run_dir=str(directory), kind=kind))
        return True


def main():
    p = argparse.ArgumentParser(description="Frozen NLNS Repair training-data precheck (CPU, no training)")
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("inspect", "verify", "smoke", "collect"):
        q = sub.add_parser(cmd)
        q.add_argument("--config", required=True)
        if cmd != "inspect":
            q.add_argument("--run-dir", required=True)
        if cmd == "collect":
            q.add_argument("--smoke-run", required=True)
    for cmd in ("resume", "status", "report"):
        q = sub.add_parser(cmd)
        q.add_argument("--run-dir", required=True)
    a = p.parse_args()
    try:
        if a.command == "inspect":
            identity, _, _, _ = inspect_full(load(a.config))
            output(dict(status="ASSETS_COMPATIBLE", identity=identity))
        elif a.command in ("verify", "smoke", "collect"):
            if execute(a.command, load(a.config), a.run_dir, smoke_run=getattr(a, "smoke_run", None)) is False:
                sys.exit(1)
        elif a.command == "resume":
            m = read_json(Path(a.run_dir) / "manifest.json")
            if execute(m["kind"], m["identity"]["config"], a.run_dir, resume=True) is False:
                sys.exit(1)
        elif a.command == "status":
            output(status(a.run_dir))
        else:
            from .reporting import generate
            with run_lock(a.run_dir):
                result = generate(a.run_dir)
            output(dict(status=result["status"], run_dir=a.run_dir))
            if result["status"] == "INVALID":
                sys.exit(1)
    except Blocked as exc:
        output(dict(status="BLOCKED_BUDGET" if isinstance(exc, BudgetExceeded) else "BLOCKED_ASSET", reason=str(exc)))
        sys.exit(2)
    except KeyboardInterrupt:
        output(dict(status="INTERRUPTED", resume="python -B -m frozen_repair resume --run-dir <same-directory>"))
        sys.exit(130)
    except Exception as exc:
        output(dict(status="INVALID", error_type=type(exc).__name__, reason=str(exc)))
        sys.exit(1)
