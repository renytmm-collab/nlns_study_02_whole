import unittest
from frozen_repair.common import edges, validate_routes, routes_cost, Invalid
from frozen_repair.diagnostics import Diagnostic, natural_summary
from frozen_repair.statistics import clustered, classify
from frozen_repair.rng import seed
from frozen_repair.reporting import aggregate_natural


class Semantics(unittest.TestCase):
    def test_edges_flip_depot_dedup(self):
        self.assertEqual(edges([[0, 1, 2, 3, 0], [0]]), {(1, 2), (2, 3)})
        self.assertEqual(edges([[0, 3, 2, 1, 0], [0, 1, 2, 0]]), {(1, 2), (2, 3)})

    def test_feasibility_independent_cost(self):
        validate_routes([[0], [0, 1, 2, 0]], [0, 1, 1], 2)
        self.assertEqual(routes_cost([[0, 1, 2, 0]], [[0, 0], [3, 0], [3, 4]]), 12)
        for rr in ([[0, 1, 0]], [[0, 1, 1, 2, 0]], [[1, 2, 0]]):
            with self.assertRaises(Invalid):
                validate_routes(rr, [0, 1, 1], 2)

    def test_probability_denominator_and_recovered_edge(self):
        diag = Diagnostic([[0, 1, 2, 3, 0]], [[0, 1, 2, 0], [0, 3, 0]], {})
        d = dict(origin=2, legal=[0, 1, 3], mapping=[0, 1, 2, 3], probabilities=[.5, .2, 0, .3])
        q = diag.qualify(d)
        self.assertEqual(q["Vstar"], [1, 3])
        self.assertEqual(q["Vnew"], [3])
        self.assertAlmostEqual(q["mass_new"], .3)  # depot stays in probability denominator

    def test_rloo_ties_terminal_dedup(self):
        x = dict(customer_ids=[0, 10, 20, 30], original_locations=[[0, 0], [1, 0], [2, 0], [3, 0]])
        diag = Diagnostic([[0, 1, 2, 3, 0]], [[0, 1, 0], [0, 2, 0], [0, 3, 0]], x)
        def trajectory(c):
            return dict(cost=c, routes=[[0, 1, 2, 3, 0]], actions=[
                dict(origin=1, target=2, mapping=[0, 1, 2], legal=[0, 2], Vstar=[2], Vnew=[2],
                     mass_star=.4, mass_new=.4, terminal=True),
                dict(origin=2, target=1, mapping=[0, 1, 2], legal=[0, 1], Vstar=[1], Vnew=[1],
                     mass_star=.4, mass_new=.4, terminal=True)])
        tr = [trajectory(c) for c in (10, 10, 10, 10)]
        diag.annotate(tr, 10)
        self.assertTrue(all(t["q"] == 0 for t in tr))
        self.assertTrue(all(a["coefficient"] == 0 for t in tr for a in t["actions"]))
        self.assertEqual([a["n"] for a in tr[0]["actions"]], [1, 0])
        tr = [trajectory(c) for c in (9, 10, 11, 12)]
        diag.annotate(tr, 10)
        self.assertAlmostEqual(sum(t["advantage"] for t in tr), 0)
        self.assertLess(tr[0]["actions"][0]["coefficient"], 0)

    def test_instance_weighting_and_identical_zero(self):
        points = [dict(instance_id="a", dEP=-.01, actions=dict(E=1, P=2)),
                  dict(instance_id="b", dEP=0, actions=dict(E=1, P=1)),
                  dict(instance_id="b", dEP=0, actions=dict(E=1, P=1))]
        c = clustered(points, "dEP")
        self.assertEqual(c["estimate"], -.005)
        self.assertEqual(c["comparison_label"], "INCONCLUSIVE")
        self.assertEqual(clustered(points, "dEP", divergence=True)["points"], 1)
        self.assertEqual(classify([.001, .002]), "HARM_SUPPORTED")
        self.assertEqual(classify([-.003, -.002]), "MEANINGFUL_LOCAL_ADVANTAGE_SUPPORTED")

    def test_rng_key_is_order_independent(self):
        expected = seed("customer-id", 5, "natural", 1)
        seed("unrelated-task")
        self.assertEqual(seed("customer-id", 5, "natural", 1), expected)
        self.assertNotEqual(seed("customer-id", 5, "natural", 2), expected)

    def test_autonomous_denominators_and_new_edge_geometry(self):
        x = dict(customer_ids=[0, 10, 20, 30], original_locations=[[0, 0], [1, 0], [2, 0], [3, 0]])
        diag = Diagnostic([[0, 1, 2, 0], [0, 3, 0]], [[0, 1, 0], [0, 2, 0], [0, 3, 0]], x)
        trajectories = []
        for cost in (9, 10, 11, 12):
            actions = []
            for u, v, legal in ((1, 2, [0, 2]), (2, 3, [0, 3]), (3, 0, [0])):
                a = dict(origin=u, target=v, mapping=[0, 1, 2, 3], legal=legal,
                         probabilities=([.6, 0, .4, 0] if u == 1 else
                                        ([.6, 0, 0, .4] if u == 2 else [1, 0, 0, 0])), terminal=v == 0)
                a.update(diag.qualify(a))
                actions.append(a)
            trajectories.append(dict(cost=cost, routes=[[0, 1, 2, 3, 0]], actions=actions))
        diag.annotate(trajectories, 10)
        summary = natural_summary(trajectories)
        self.assertEqual(len(trajectories[0]["actions"]), 3)  # raw terminal step retained
        self.assertEqual([a["autonomous"] for a in trajectories[0]["actions"]], [True, True, False])
        self.assertEqual((summary["actions"], summary["total_steps"], summary["deterministic_steps"]), (8, 12, 4))
        result = aggregate_natural([summary])
        self.assertEqual(result["Vnew"]["nonempty_rate"], .5)
        self.assertAlmostEqual(result["all_steps"]["Vnew"]["nonempty_rate"], 1/3)
        self.assertAlmostEqual(result["Vnew"]["zero_filled_mass"]["mean"], .2)
        self.assertAlmostEqual(result["all_steps"]["Vnew"]["zero_filled_mass"]["mean"], .4/3)
        self.assertEqual(result["qnh_per_action"], .25)
        self.assertAlmostEqual(result["all_steps"]["qnh_per_action"], 1/6)
        geometry = result["new_edge_geometry"]
        self.assertEqual(geometry["elite"]["count"], 4)
        self.assertEqual(geometry["nonelite"]["count"], 4)
        self.assertEqual(geometry["elite"]["edge_length"]["mean"], 1)
        self.assertEqual(geometry["elite"]["nearest_neighbor_rank"]["mean"], 1)
        self.assertEqual(geometry["nonelite"]["nearest_neighbor_rank"]["mean"], 2)

    def test_no_autonomous_actions_have_undefined_ratios(self):
        x = dict(customer_ids=[0, 7], original_locations=[[0, 0], [1, 0]])
        diag = Diagnostic([[0, 1, 0]], [[0, 1, 0]], x)
        trajectories = []
        for cost in (1, 1, 1, 1):
            a = dict(origin=1, target=0, mapping=[0, 1], legal=[0], probabilities=[1, 0], terminal=True)
            a.update(diag.qualify(a))
            trajectories.append(dict(cost=cost, routes=[[0, 1, 0]], actions=[a]))
        diag.annotate(trajectories, 1)
        s = natural_summary(trajectories)
        self.assertTrue(all(t["nh_per_action"] is None for t in s["trajectories"]))
        result = aggregate_natural([s])
        self.assertEqual(result["actions"], 0)
        self.assertEqual(result["total_steps"], 4)
        self.assertIsNone(result["qnh_per_action"])
        self.assertIsNone(result["Vnew"]["nonempty_rate"])
        self.assertIsNone(result["Vnew"]["zero_filled_mass"]["mean"])
        self.assertEqual(result["trajectory_nh_per_action"]["n"], 0)
        self.assertEqual(result["all_steps"]["qnh_per_action"], 0)
        self.assertEqual(result["new_edge_geometry"]["elite"]["count"], 0)
