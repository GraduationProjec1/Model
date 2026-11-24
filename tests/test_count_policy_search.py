import numpy as np

from tools.count_policy_search import Scorer, add_sku_thresholds, sign_test


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 5) == 1.0
    assert sign_test(0, 10) < 0.003


class _Fake(Scorer):
    """Three images, one label, scores 0.9 / 0.3 / 0.25 on a single detection each; GT has 1, 0, 0 objects."""

    def __init__(self):
        from autocheckout.predictions import Predictions  # noqa: F401
        self.ids, self.labels = [0, 1, 2], [0]
        self.gt = np.array([[1], [0], [0]])
        self._scores = np.array([0.9, 0.3, 0.25])

    def right(self, vector):
        return (self._scores >= vector[0]).astype(int) == self.gt[:, 0]


def test_sku_threshold_needs_min_gain():
    fake, vec = _Fake(), np.array([0.2])
    assert add_sku_thresholds(fake, vec, [0], [0.2, 0.5], min_gain=2)[1] == {0: 0.5}
    assert add_sku_thresholds(fake, vec, [0], [0.2, 0.5], min_gain=3)[1] == {}


class _FakeRecall(_Fake):
    """Image 0 has the only object (score 0.3); images 1 and 2 hold boxes on no object (0.35, 0.25)."""

    def __init__(self):
        super().__init__()
        self._scores = np.array([0.3, 0.35, 0.25])

    def recall(self, vector, label):
        return float(self._scores[0] >= vector[0])


def test_sku_threshold_rejected_when_recall_drops():
    fake, vec = _FakeRecall(), np.array([0.2])
    assert add_sku_thresholds(fake, vec, [0], [0.2, 0.4], min_gain=1)[1] == {0: 0.4}
    assert add_sku_thresholds(fake, vec, [0], [0.2, 0.4], min_gain=1, max_recall_drop=0.02)[1] == {}


def test_scorer_recall_counts_objects_kept():
    s = Scorer.__new__(Scorer)
    s.ids, s.labels = [0, 1], [3]
    s.gt = np.array([[2], [1]])
    s.counts = lambda vector: np.array([[1], [1]])
    assert s.recall(np.zeros(4), 3) == 2 / 3
