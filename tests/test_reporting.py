import tempfile
import unittest
from pathlib import Path
from frozen_repair.common import ROOT, atomic_json, digest, read_json
from frozen_repair.ledger import Ledger
from frozen_repair.reporting import generate
from frozen_repair.config import PROTOCOL_VERSION, INITIALIZATION, budget_limits


class Reports(unittest.TestCase):
    def test_recompute_costs_and_mark_partial_invalid(self):
        root = ROOT / "test_artifacts"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            d = Path(directory)
            (d / "raw").mkdir()
            identity = dict(scope="synthetic contract fixture, not solver evidence", protocol_version=PROTOCOL_VERSION,
                            config=dict(protocol_version=PROTOCOL_VERSION), initialization=INITIALIZATION,
                            selected_ids=["i", "b", "c", "d"], model=dict(parameter_sha256="a"*64))
            atomic_json(d / "manifest.json", dict(identity=identity, identity_sha256=digest(identity),
                        protocol_version=PROTOCOL_VERSION, ledger_schema_version=2,
                        kind="smoke", task_ids=["t", "missing"], instance_ids=["i", "missing"], limits=budget_limits("smoke")))
            x = dict(customer_ids=[0, 1, 2], original_locations=[[0, 0], [1, 0], [2, 0]])
            def trajectory(c):
                return dict(cost=c, routes=[[0, 1, 2, 0]], actions=[
                    dict(origin=1, target=2, legal=[0, 2], mapping=[0, 1, 2],
                         Vstar=[2], Vnew=[2], mass_star=.4, mass_new=.4),
                    dict(origin=2, target=0, legal=[0], mapping=[0, 1, 2],
                         Vstar=[], Vnew=[], mass_star=0, mass_new=0, terminal=True)])
            trajectories = [trajectory(c) for c in (9, 10, 11, 12)]
            point = dict(point_id="p", instance_id="i", depth=0, actions=dict(E=1, P=2, N=1),
                         aliases=dict(E="E", P="P", N="E"),
                         results=dict(E=[dict(cost=9)]*8, P=[dict(cost=10)]*8),
                         requested_completions=24, executed_completions=16, before_cost=10,
                         targets_original_ids=dict(E=1, P=2, N=1), dEP=999, dEN=999)
            state = dict(depth=0, before_cost=10, complete_state=dict(pool=[[[0, 0, 0], [1, 1, None],
                          [0, 0, 0]], [[0, 0, 0], [2, 1, None], [0, 0, 0]]], solution=[0, 1]),
                         trajectories=trajectories, point=point)
            ledger = Ledger(d, {})
            try:
                ledger.add("t")
                ledger.complete("t", dict(instance_id="i", instance=x, elite_routes=[[0, 1, 2, 0]], states=[state]))
            finally:
                ledger.close()
            result = generate(d)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["comparisons"]["dEP"]["estimate"], -.1)
            self.assertEqual(result["comparisons"]["dEN"]["estimate"], 0)
            self.assertEqual(result["divergent_exploratory"]["dEP"]["comparison_label"], "INCOMPLETE")
            self.assertEqual(result["natural"]["actions"], 4)
            self.assertEqual(result["natural"]["total_steps"], 8)
            self.assertEqual(result["natural_by_instance"]["i"]["new_edge_geometry"]["elite"]["count"], 4)
            self.assertEqual(result["natural_by_depth"]["0"]["all_steps"]["actions"], 8)
            atomic_json(d / "failure.json", dict(status="INVALID"))
            self.assertEqual(generate(d)["status"], "INVALID")
