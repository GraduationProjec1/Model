import json

from tools.export_count_policy import export

CFG = 'configs/relearn/task_config_relearn.json'


def test_group_thresholds_to_label_vector_and_review_table(tmp_path):
    scored = tmp_path / 'scored.json'
    scored.write_text(json.dumps({'name': 'r', 'thresholds': {f'task_{t}': 0.1 * t for t in range(1, 6)},
                                  'nms_iou': 0.45}))
    search = tmp_path / 'search.json'
    search.write_text(json.dumps({'policies': {'nms=0.45+sku': {'nms': 0.45, 'sku_thresholds': {'141': 0.53}}}}))
    out = export(scored, CFG, review_from=search, review_policy='nms=0.45+sku')
    v = out['label_thresholds']
    assert len(v) == 200 and v[0] == 0.1 and v[199] == 0.5  # label 0 thuộc task 1, label 199 thuộc task 5
    assert out['nms_iou'] == 0.45 and out['review_score_by_label'] == {'141': 0.53}


def test_review_policy_with_other_nms_is_refused(tmp_path):
    import pytest
    scored = tmp_path / 'scored.json'
    scored.write_text(json.dumps({'thresholds': {f'task_{t}': 0.2 for t in range(1, 6)}, 'nms_iou': 0.45}))
    search = tmp_path / 'search.json'
    search.write_text(json.dumps({'policies': {'nms=0.5+sku': {'nms': 0.5, 'sku_thresholds': {'141': 0.4}}}}))
    with pytest.raises(ValueError):
        export(scored, CFG, review_from=search, review_policy='nms=0.5+sku')
