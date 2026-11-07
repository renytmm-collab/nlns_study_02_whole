import copy
import unittest
from types import SimpleNamespace
from frozen_repair.adapter import instance_state
from frozen_repair.common import digest


class ArrayView:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return copy.deepcopy(self.values)


class SnapshotIsolation(unittest.TestCase):
    def test_snapshot_detaches_live_environment_lists(self):
        depot, customer = [[0, 0, 0]], [[1, 1, 1]]
        instance = SimpleNamespace(nb_customers=1, capacity=1,
            locations=ArrayView([[0, 0], [1, 0]]), original_locations=ArrayView([[0, 0], [1, 0]]),
            demand=ArrayView([0, 1]), solution=[depot, customer], incomplete_tours=[customer],
            nn_input_idx_to_tour=[[depot, 0], [customer, 0]], open_nn_input_idx=[1])
        snapshot = instance_state(instance)
        before = digest(snapshot)
        self.assertIsNot(snapshot["open"], instance.open_nn_input_idx)
        # A natural trajectory keeps mutating its source after a point was selected.
        instance.open_nn_input_idx.clear()
        customer[0][2] = 0
        customer.append([0, 0, 0])
        self.assertEqual(snapshot["open"], [1])
        self.assertEqual(digest(snapshot), before)
