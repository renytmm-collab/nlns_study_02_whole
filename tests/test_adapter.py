"""Opt-in checks with actual server assets, never silently substitutes a model."""
import os
import tempfile
import unittest
from pathlib import Path
from frozen_repair.common import ROOT


@unittest.skipUnless(os.environ.get("PRECHECK_CONFIG"), "Requires PRECHECK_CONFIG and real compatible server assets")
class RealAdapter(unittest.TestCase):
    def test_official_parity_and_snapshot_replay(self):
        from frozen_repair.cli import inspect_full
        from frozen_repair.config import load
        from frozen_repair.verification import verify_actor
        from frozen_repair.ledger import Ledger
        identity, inputs, _, actor = inspect_full(load(os.environ["PRECHECK_CONFIG"]))
        root = ROOT / "test_artifacts"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            ledger = Ledger(Path(directory), dict(semantic_full=8, semantic_partial=8))
            try:
                result = verify_actor(actor, identity, inputs, ledger)
                self.assertTrue(result["passed"])
                self.assertEqual(len(result["records"]), 4)
            finally:
                ledger.close()
