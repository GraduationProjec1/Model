"""Round 3 (docs/design/2026-11-07-vong-3, Task 7): tools/fit_check.py splits Val-A into trained / untrained pool /
calib and scores each with the same thresholds."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_check

PREDS = SimpleNamespace(image_id=np.arange(1, 7))

SPLIT = {'calib': [1, 2], 'pool': [3, 4, 5, 6]}


def fake_count_on(seen):
    def count_on(val_full, cfg, preds, ids, thresholds):
        seen.append((tuple(ids), tuple(thresholds)))
        return {'cAcc_m2': len(ids) / 10}
    return count_on


def test_fit_scores_splits_val_a(monkeypatch):
    seen = []
    monkeypatch.setattr(fit_check, 'count_on', fake_count_on(seen))
    r = fit_check.fit_scores({}, None, PREDS, SPLIT, [3, 4, 5, 99], thresholds=[0.1] * 5)
    assert [ids for ids, _ in seen] == [(3, 4, 5), (6,), (1, 2)]
    assert {t for _, t in seen} == {(0.1,) * 5}
    assert (r['images_trained'], r['images_pool_untrained'], r['images_calib']) == (3, 1, 2)
    assert r['cAcc_trained'] == pytest.approx(0.3)


def test_fit_scores_thresholds_from_calib(monkeypatch):
    seen = []
    monkeypatch.setattr(fit_check, 'count_on', fake_count_on(seen))
    monkeypatch.setattr(fit_check, 'score', lambda v, c, p, calib, target: {
        'thresholds': {f'task_{t}': t / 10 for t in range(1, 6)}} if calib == target == [1, 2] else None)
    r = fit_check.fit_scores({}, None, PREDS, SPLIT, [3])
    assert r['fit_thresholds'] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_fit_scores_refuses_a_selection_on_calib():
    with pytest.raises(ValueError, match='calib'):
        fit_check.fit_scores({}, None, PREDS, SPLIT, [1, 3], thresholds=[0.1] * 5)


def test_fit_scores_needs_predictions_on_val_a():
    with pytest.raises(ValueError, match='Val-A'):
        fit_check.fit_scores({}, None, SimpleNamespace(image_id=np.arange(1, 3)), SPLIT, [3], thresholds=[0.1] * 5)
