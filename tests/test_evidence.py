import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from frozen_repair.common import ROOT, Invalid, atomic_json, read_json, file_hash
from frozen_repair.ledger import Ledger
from frozen_repair.evidence import audit, finalize, smoke_proof, validate_checks
from frozen_repair.reporting import generate
from evidence_fixtures import create_run


class Reliability(unittest.TestCase):
    def setUp(self):
        (ROOT / "test_artifacts").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "test_artifacts")
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def reopen(self, m):
        return Ledger(self.path, m["limits"], existing=True, expected_tasks=m["task_ids"])

    def mutate(self, sql):
        with closing(sqlite3.connect(self.path / "ledger.sqlite")) as db:
            db.execute(sql)
            db.commit()

    def test_missing_table_or_field_rejected_without_repair(self):
        m = create_run(self.path)
        self.mutate("DROP TABLE cache")
        with self.assertRaises(Invalid):
            self.reopen(m)
        with closing(sqlite3.connect(self.path / "ledger.sqlite")) as db:
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='cache'").fetchone())
        self.mutate("CREATE TABLE cache (id TEXT PRIMARY KEY, hash TEXT)")
        with self.assertRaises(Invalid):
            self.reopen(m)

    def test_missing_database_and_old_schema_are_not_recreated(self):
        with self.assertRaises(Invalid):
            Ledger(self.path, {}, existing=True, expected_tasks=[])
        self.assertFalse((self.path / "ledger.sqlite").exists())
        m = create_run(self.path)
        self.mutate("PRAGMA user_version=1")
        with self.assertRaises(Invalid):
            self.reopen(m)

    def test_cache_cannot_bind_an_unrelated_charged_call(self):
        m = create_run(self.path)
        ledger = self.reopen(m)
        try:
            ledger.cached("natural_full", "t0", [0, 1], lambda: dict(cost=7))
        finally:
            ledger.close()
        self.mutate("UPDATE cache SET call_id=1")
        with self.assertRaises(Invalid):
            self.reopen(m)

    def test_orphan_cache_call_rejected(self):
        m = create_run(self.path)
        self.mutate("INSERT INTO cache VALUES ('bad', 'bad', 99999)")
        with self.assertRaises(Invalid):
            self.reopen(m)

    def test_unknown_task_call_kind_or_excess_budget_rejected(self):
        m = create_run(self.path)
        self.mutate("INSERT INTO tasks VALUES ('unexpected','pending',NULL,NULL)")
        with self.assertRaises(Invalid):
            self.reopen(m)
        self.mutate("DELETE FROM tasks WHERE id='unexpected'")
        self.mutate("UPDATE calls SET kind='unbudgeted' WHERE id=1")
        with self.assertRaises(Invalid):
            self.reopen(m)
        self.mutate("UPDATE calls SET kind='semantic_full' WHERE id=1")
        m["limits"]["semantic_full"] = 0
        with self.assertRaises(Invalid):
            self.reopen(m)

    def test_normal_resume_reuses_cache_without_charge(self):
        m = create_run(self.path)
        ledger = self.reopen(m)
        try:
            first = ledger.cached("natural_full", "t0", [0, 1], lambda: dict(cost=7))
            before = ledger.counts()
        finally:
            ledger.close()
        ledger = self.reopen(m)
        try:
            self.assertEqual(ledger.cached("natural_full", "t0", [0, 1], lambda: self.fail("reran cached call")), first)
            self.assertEqual(ledger.counts(), before)
        finally:
            ledger.close()

    def test_last_task_commit_without_finalization_is_unfinalized(self):
        m = create_run(self.path)
        self.assertEqual(generate(self.path)["status"], "UNFINALIZED")
        self.assertEqual(generate(self.path)["comparisons"]["dEP"]["comparison_label"], "UNFINALIZED")
        from frozen_repair.cli import execute
        actor = object()
        with patch("frozen_repair.cli.inspect_full", return_value=(m["identity"], {}, {}, actor)), \
                patch("frozen_repair.adapter.parameter_hash", return_value="a"*64), patch("frozen_repair.cli.output"):
            execute("smoke", m["identity"]["config"], self.path, resume=True)
        self.assertEqual(generate(self.path)["status"], "COMPLETE")
        self.assertEqual(smoke_proof(self.path, m["identity"])["completion_sha256"], file_hash(self.path / "completion.json"))

    def test_completion_binds_actual_task_set_hashes_and_calls(self):
        m = create_run(self.path)
        finalize(self.path, "a"*64)
        self.mutate("UPDATE tasks SET id='replacement' WHERE id='t3'")
        self.assertEqual(generate(self.path)["status"], "INVALID")

    def test_checks_structure_and_parameter_tampering_rejected(self):
        m = create_run(self.path)
        original = read_json(self.path / "checks.json")
        variants = [dict(passed=True), dict(original, records=original["records"][:-1]),
                    dict(original, parameter_after_sha256="b"*64)]
        for changed in variants:
            with self.subTest(changed=list(changed)):
                atomic_json(self.path / "checks.json", changed)
                # Even rebinding the file hash cannot make malformed content valid.
                with closing(sqlite3.connect(self.path / "ledger.sqlite")) as db:
                    db.execute("UPDATE proofs SET hash=? WHERE name='checks'", (file_hash(self.path / "checks.json"),))
                    db.commit()
                with self.assertRaises(Invalid):
                    smoke_proof(self.path, m["identity"])

    def test_missing_or_uncommitted_checks_not_overwritten(self):
        m = create_run(self.path)
        self.mutate("DELETE FROM proofs")
        with self.assertRaises(Invalid):
            audit(self.path)
        before = (self.path / "checks.json").read_bytes()
        from frozen_repair.cli import execute
        with patch("frozen_repair.cli.inspect_full", return_value=(m["identity"], {}, {}, object())):
            with self.assertRaises(Invalid):
                execute("smoke", m["identity"]["config"], self.path, resume=True)
        self.assertEqual((self.path / "checks.json").read_bytes(), before)

    def test_report_failure_does_not_invalidate_finalized_run(self):
        m = create_run(self.path)
        from frozen_repair.cli import execute
        with patch("frozen_repair.cli.inspect_full", return_value=(m["identity"], {}, {}, object())), \
                patch("frozen_repair.adapter.parameter_hash", return_value="a"*64), \
                patch("frozen_repair.cli.output"), \
                patch("frozen_repair.reporting.generate", side_effect=OSError("simulated report failure")):
            execute("smoke", m["identity"]["config"], self.path, resume=True)
        self.assertFalse((self.path / "failure.json").exists())
        self.assertEqual(audit(self.path)["status"], "COMPLETE")
        self.assertEqual(generate(self.path)["status"], "COMPLETE")

    def test_damaged_final_credential_is_invalid(self):
        create_run(self.path)
        finalize(self.path, "a"*64)
        done = read_json(self.path / "completion.json")
        done["parameter_sha256"] = "b"*64
        atomic_json(self.path / "completion.json", done)
        self.assertEqual(generate(self.path)["status"], "INVALID")

    def test_deleted_checks_file_and_changed_calls_rejected(self):
        m = create_run(self.path)
        finalize(self.path, "a"*64)
        original = (self.path / "checks.json").read_bytes()
        (self.path / "checks.json").unlink()
        with self.assertRaises(Invalid):
            smoke_proof(self.path, m["identity"])
        (self.path / "checks.json").write_bytes(original)
        self.mutate("INSERT INTO calls(kind,task,started,detail) VALUES ('natural_full','t0',0,'[0,1]')")
        with self.assertRaises(Invalid):
            audit(self.path)

    def test_protocol_change_cannot_reuse_completion(self):
        create_run(self.path)
        finalize(self.path, "a"*64)
        manifest = read_json(self.path / "manifest.json")
        manifest["protocol_version"] = "old-protocol"
        atomic_json(self.path / "manifest.json", manifest)
        self.assertEqual(generate(self.path)["status"], "INVALID")
