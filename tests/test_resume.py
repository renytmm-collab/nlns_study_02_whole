import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from frozen_repair.common import ROOT, Invalid, Blocked, atomic_json
from frozen_repair.ledger import Ledger, status, run_lock


class Resume(unittest.TestCase):
    def setUp(self):
        root = ROOT / "test_artifacts"
        root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.path = Path(self.tmp.name)
        (self.path / "raw" / "calls").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reuse_checksum_and_budget(self):
        calls = []
        ledger = Ledger(self.path, dict(full=2))
        try:
            ledger.add("task")
            ledger.begin("task")
            def compute():
                calls.append(1)
                return dict(cost=123)
            self.assertEqual(ledger.cached("full", "task", [0], compute), dict(cost=123))
            self.assertEqual(ledger.cached("full", "task", [0], compute), dict(cost=123))
            self.assertEqual(len(calls), 1)
            ledger.charge("full", "interrupted")
            with self.assertRaises(Blocked):
                ledger.charge("full", "retry")
            ledger.complete("task", dict(result=1))
            self.assertTrue(ledger.completed("task"))
            atomic_json(self.path / "raw" / "task.json", dict(result=2))
            with self.assertRaises(Invalid):
                ledger.completed("task")
            self.assertEqual(status(self.path)["calls"], dict(full=2))
        finally:
            ledger.close()

    def test_single_writer_lock(self):
        with run_lock(self.path):
            with self.assertRaises(Blocked):
                with run_lock(self.path):
                    pass

    def test_failed_execution_not_cached_or_refunded(self):
        ledger = Ledger(self.path, dict(full=2))
        def fail():
            raise RuntimeError("simulated interruption")
        try:
            with self.assertRaises(RuntimeError):
                ledger.cached("full", "task", [0], fail)
            self.assertEqual(ledger.counts(), dict(full=1))
            self.assertEqual(ledger.cached("full", "task", [0], lambda: dict(cost=1)), dict(cost=1))
            self.assertEqual(ledger.counts(), dict(full=2))
        finally:
            ledger.close()

    def test_resume_rejects_changed_identity_before_ledger(self):
        from frozen_repair.cli import execute
        from frozen_repair.common import digest
        old = dict(source_hashes={"adapter.py": "old"})
        new = dict(source_hashes={"adapter.py": "new"})
        atomic_json(self.path / "manifest.json", dict(identity=old, identity_sha256=digest(old), kind="smoke"))
        with patch("frozen_repair.cli.inspect_full", return_value=(new, {}, {}, None)):
            with self.assertRaises(Blocked):
                execute("smoke", {}, self.path, resume=True)
        self.assertFalse((self.path / "ledger.sqlite").exists())
