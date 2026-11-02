"""tools/forgetting_report.py: ACC, BWT and forgetting of an accuracy matrix (B1)."""

import pytest

from tools.forgetting_report import summarize


def test_no_forgetting():
    s = summarize({'1': {'1': 0.9}, '2': {'1': 0.9, '2': 0.8}})
    assert s['forgetting'] == pytest.approx(0) and s['BWT'] == pytest.approx(0)
    assert s['ACC'] == {1: pytest.approx(0.9), 2: pytest.approx(0.85)}


def test_forgetting_and_bwt():
    s = summarize({'1': {'1': 0.9}, '2': {'1': 0.6, '2': 0.8}, '3': {'1': 0.3, '2': 0.7, '3': 0.9}})
    assert s['forgetting'] == pytest.approx(((0.9 - 0.3) + (0.8 - 0.7)) / 2)
    assert s['BWT'] == pytest.approx(((0.3 - 0.9) + (0.7 - 0.8)) / 2)
    assert s['ACC_final'] == pytest.approx((0.3 + 0.7 + 0.9) / 3)
