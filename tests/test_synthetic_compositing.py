"""Synthetic v2, T2.2: cutout transforms and scene rendering with visible-mask labels."""

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from autocheckout.compositing import SceneError, render_scene, transform_cutout
from autocheckout.synthetic import fit_scales

LAYOUT = {"center_mean": [0.5, 0.5], "center_std": {"easy": 0.2}, "min_visible": 0.7,
          "position_tries": 60, "object_tries": 4, "scene_retries": 6}
SCALE = {"jitter_log_sigma": 0.05, "max_side_frac": 0.55}
PHOTO = {"object_gain": {"0": [0.9, 1.0], "2": [0.9, 1.0]}, "object_contrast": 0.05, "object_color": 0.02,
         "shadow_blur": 0.06, "shadow_strength": [0.15, 0.3], "shadow_offset": [0.01, 0.03], "edge_blur": 3,
         "global_gain": [0.98, 1.02], "global_offset": 3, "noise_sigma": [0.5, 1.0], "blur_prob": 0.0,
         "blur_sigma": [0.4, 0.8], "jpeg_quality": [75, 75]}
CONFIG = {"layout": LAYOUT, "scale": SCALE, "photometric": PHOTO}


def l_shape():
    rgba = np.zeros((40, 30, 4), np.uint8)
    rgba[:, :8] = (200, 30, 30, 255)     # vertical bar on the left
    rgba[32:, :] = (30, 30, 200, 255)    # foot at the bottom
    return rgba


def test_transform_without_rotation_keeps_orientation():
    out = transform_cutout(l_shape(), 0.0, 1.0)
    assert out.shape == (40, 30, 4) and np.array_equal(out[..., 3] > 127, l_shape()[..., 3] > 127)


def test_quarter_turn_is_a_rotation_not_a_mirror():
    out = transform_cutout(l_shape(), 90.0, 1.0)[..., 3] > 127
    expected = np.rot90(l_shape()[..., 3] > 127)          # counter-clockwise
    mirrored = np.fliplr(expected)
    assert abs(out.shape[0] - expected.shape[0]) <= 1 and abs(out.shape[1] - expected.shape[1]) <= 1
    h, w = min(out.shape[0], expected.shape[0]), min(out.shape[1], expected.shape[1])
    out, expected, mirrored = out[:h, :w], expected[:h, :w], mirrored[:h, :w]
    assert (out == expected).mean() > 0.95 and (out == mirrored).mean() < 0.9


@pytest.fixture
def assets(tmp_path):
    Image.new("RGB", (240, 240), (140, 142, 141)).save(tmp_path / "bg.jpg")
    cutouts = {}
    rng = np.random.default_rng(0)
    for k in range(4):
        rgba = np.zeros((50, 36, 4), np.uint8)
        rgba[4:46, 3:33] = (*rng.integers(0, 255, 3), 255)
        rgba[4:14, 3:12, 3] = 0            # notch so the shape is asymmetric
        Image.fromarray(rgba, "RGBA").save(tmp_path / f"c{k}.png")
        cutouts[f"x:{k}"] = {"cutout_id": f"x:{k}", "file": str(tmp_path / f"c{k}.png"), "rpc_category_id": k + 1,
                             "camera": 2, "size": [36, 50], "bbox_source": [0, 0, 36, 50]}
    scales = {f"{k + 1}:2": {"k": 1.0} for k in range(4)}
    return tmp_path, cutouts, scales


def recipe(n=3, seed=5):
    return {"generation_task": 1, "image_id": 1, "level": "easy", "seed": seed, "background": "bg.jpg",
            "objects": [{"instance_id": i + 1, "rpc_category_id": i + 1, "camera": 2, "cutout_id": f"x:{i}",
                         "labeled": True} for i in range(n)]}


def render(assets, rec, config=CONFIG):
    root, cutouts, scales = assets
    return render_scene(rec, cutouts=cutouts, scales=scales, config=config, backgrounds_dir=root)


def test_labels_come_from_the_final_owner_mask(assets):
    out = render(assets, recipe(4))
    owner, anns = out["owner"], out["annotations"]
    assert out["image"].shape == (240, 240, 3) and owner.dtype == np.uint16
    assert sorted(a["instance_id"] for a in anns) == [1, 2, 3, 4]
    for a in anns:
        visible = owner == a["instance_id"]
        decoded = mask_utils.decode(a["segmentation"]).astype(bool)
        assert np.array_equal(decoded, visible)
        ys, xs = np.nonzero(visible)
        assert a["bbox"] == [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        assert a["area"] == int(visible.sum())
        assert a["visible_fraction"] >= 0.7


def test_rgb_shows_each_object(assets):
    out = render(assets, recipe(3))
    background = np.array([140, 142, 141], float)
    for a in out["annotations"]:
        pixels = out["image"][out["owner"] == a["instance_id"]].astype(float)
        assert np.abs(pixels.mean(axis=0) - background).max() > 15


def test_render_is_deterministic(assets):
    a, b = render(assets, recipe(4)), render(assets, recipe(4))
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["owner"], b["owner"])
    assert a["resolved"] == b["resolved"]
    assert not np.array_equal(a["image"], render(assets, recipe(4, seed=6))["image"])


def test_impossible_layout_raises(assets):
    root, cutouts, _ = assets
    huge = {f"{k + 1}:2": {"k": 6.0} for k in range(4)}
    config = {**CONFIG, "scale": {**SCALE, "max_side_frac": 0.9}}
    with pytest.raises(SceneError):
        render_scene(recipe(4), cutouts=cutouts, scales=huge, config=config, backgrounds_dir=root)


def test_fit_scales_matches_val_bbox_area(tmp_path):
    rgba = np.zeros((20, 10, 4), np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    # PNG stored at half the source resolution: 20x10 png for a 40x20 source bbox
    cutouts = [{"cutout_id": "x:1", "file": str(tmp_path / "a.png"), "rpc_category_id": 1, "camera": 2,
                "size": [10, 20], "bbox_source": [0, 0, 20, 40]}]
    skus = [{"rpc_category_id": 1, "camera_weights": {"2": 1.0}}]
    val = {"annotations": [{"category_id": 1, "bbox": [0, 0, 60, 120]}] * 3}
    scales = fit_scales(cutouts, skus, val)
    # source 20x40 -> val 60x120 is a factor 3 on the long side; rotated bbox areas are compared, so
    # the fitted factor lies a little below 3
    assert 2.0 < scales["1:2"]["k"] < 3.0 and scales["1:2"]["method"] == "rotated_area"


def test_occluded_share_of_scenes_uses_the_lower_visibility(assets):
    layout = {**LAYOUT, "min_visible": 0.95, "occluded_scene_prob": 0.5, "occluded_min_visible": 0.75}
    config = {**CONFIG, "layout": layout}
    outs = [render(assets, recipe(4, seed=s), config) for s in range(12)]
    flags = [o["resolved"]["occluded"] for o in outs]
    assert any(flags) and not all(flags)
    for o in outs:
        floor = 0.75 if o["resolved"]["occluded"] else 0.95
        assert o["resolved"]["min_visible"] == floor
        assert all(a["visible_fraction"] >= floor for a in o["annotations"])


def test_basket_without_layout_replays_retries_at_occluded_visibility(assets):
    # four large packs cannot fit at 95% visible in any regular retry, nor on its last-resort retries
    root, cutouts, _ = assets
    large = {f"{k + 1}:2": {"k": 3.2} for k in range(4)}
    strict = {**LAYOUT, "min_visible": 0.95}
    config = {**CONFIG, "scale": {**SCALE, "max_side_frac": 0.9}}
    with pytest.raises(SceneError):
        render_scene(recipe(4), cutouts=cutouts, scales=large, config={**config, "layout": strict}, backgrounds_dir=root)
    layout = {**strict, "occluded_scene_prob": 0.0, "occluded_min_visible": 0.6}
    out = render_scene(recipe(4), cutouts=cutouts, scales=large, config={**config, "layout": layout},
                       backgrounds_dir=root)
    assert out["resolved"]["retry"] >= LAYOUT["scene_retries"] and out["resolved"]["min_visible"] == 0.6
    assert all(a["visible_fraction"] >= 0.6 for a in out["annotations"])


def test_basket_fuller_than_the_tray_replays_once_more_one_jitter_sigma_smaller(assets):
    # four packs too large to fit even at the occluded share's visibility fit once each is exp(-0.1) smaller
    root, cutouts, _ = assets
    large = {f"{k + 1}:2": {"k": 3.6} for k in range(4)}
    layout = {**LAYOUT, "min_visible": 0.95, "occluded_scene_prob": 0.0, "occluded_min_visible": 0.6}
    config = {**CONFIG, "scale": {**SCALE, "max_side_frac": 0.9, "jitter_log_sigma": 0.1}, "layout": layout}
    out = render_scene(recipe(4), cutouts=cutouts, scales=large, config=config, backgrounds_dir=root)
    assert out["resolved"]["retry"] >= 2 * LAYOUT["scene_retries"] and out["resolved"]["min_visible"] == 0.6
    assert all(a["visible_fraction"] >= 0.6 for a in out["annotations"])


def _saturation(rgb):
    import cv2
    return cv2.cvtColor(rgb.astype(np.uint8)[None], cv2.COLOR_RGB2HSV)[0][:, 1].astype(float)


def test_object_saturation_default_keeps_old_images(assets):
    """v2c (AN/cyclegan/so_sanh_xu_ly): a missing object_saturation renders exactly as before (factor 1.0)."""
    old = render(assets, recipe(4))
    same = render(assets, recipe(4), {**CONFIG, "photometric": {**PHOTO, "object_saturation": 1.0}})
    assert np.array_equal(old["image"], same["image"]) and old["resolved"] == same["resolved"]


def test_object_saturation_scales_object_colour_only(assets):
    base = render(assets, recipe(4))
    more = render(assets, recipe(4), {**CONFIG, "photometric": {**PHOTO, "object_saturation": 1.3}})
    assert np.array_equal(base["owner"], more["owner"])            # same layout: no extra random draws
    obj = base["owner"] > 0
    s0, s1 = _saturation(base["image"][obj]), _saturation(more["image"][obj])
    room = (s0 > 40) & (s0 < 150)                                   # pixels that do not clip at full saturation
    assert 1.2 < np.median(s1[room] / s0[room]) < 1.4
    far = ~obj
    for a in base["annotations"]:                                    # tray pixels away from every object
        x, y, w, h = a["bbox"]
        far[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12] = False
    assert np.abs(more["image"][far].astype(int) - base["image"][far].astype(int)).max() <= 1


def _upright_fixture(tmp_path):
    rgba = np.zeros((40, 10, 4), np.uint8)                 # lying bottle: long and thin
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "side.png")
    cap = np.zeros((12, 12, 4), np.uint8)                  # standing bottle seen from the top: the cap
    cap[..., 3] = 255
    Image.fromarray(cap, "RGBA").save(tmp_path / "top.png")
    cutouts = [{"cutout_id": f"x:{c}", "file": str(tmp_path / ("top.png" if c == 2 else "side.png")),
                "rpc_category_id": 1, "camera": c, "size": [12, 12] if c == 2 else [10, 40],
                "bbox_source": [0, 0, 12, 12] if c == 2 else [0, 0, 10, 40]} for c in range(4)]
    val = {"annotations": [{"category_id": 1, "bbox": [0, 0, 30, 120]}] * 3}
    return cutouts, val


def test_fit_scales_standing_cameras_with_four_camera_weights(tmp_path):
    """A2: with weights over 4 cameras no camera of an upright SKU may reach 0.5; ``standing_cameras`` names the
    standing pose (camera 2), every other camera is a main (lying) pose."""
    cutouts, val = _upright_fixture(tmp_path)
    skus = [{"rpc_category_id": 1, "shape_group": "upright", "standing_cameras": [2],
             "camera_weights": {"0": 0.40, "1": 0.20, "2": 0.03, "3": 0.37}}]
    scales = fit_scales(cutouts, skus, val)
    assert {scales[f"1:{c}"]["method"] for c in (0, 1, 3)} == {"rotated_area"}
    assert scales["1:2"]["method"] == "standing_short_side"
    assert scales["1:2"]["k"] == pytest.approx(30 / 12)    # cap diameter = median val short side


def test_fit_scales_without_standing_cameras_keeps_the_old_rule(tmp_path):
    cutouts, val = _upright_fixture(tmp_path)
    skus = [{"rpc_category_id": 1, "shape_group": "upright", "camera_weights": {"0": 0.99, "2": 0.01}}]
    scales = fit_scales([c for c in cutouts if c["camera"] in (0, 2)], skus, val)
    assert scales["1:0"]["method"] == "rotated_area" and scales["1:2"]["method"] == "standing_short_side"


def _rect_cutout(tmp_path, name, w, h):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / name)
    return str(tmp_path / name)


def test_fit_scales_per_camera_from_matched_val_objects(tmp_path):
    """Cach 1 (AN/cyclegan/ra_soat/README.md, section 5): with ``matches`` every main camera of a SKU gets its own
    k from the Val-A objects that look most like that camera; a camera with fewer than ``min_matches`` takes the
    shape group's median ratio to the SKU's best-matched camera."""
    big, small = _rect_cutout(tmp_path, "big.png", 20, 20), _rect_cutout(tmp_path, "small.png", 10, 10)
    cutouts, skus, matches = [], [], {}
    for sid in (1, 2, 3):
        for cam, f, side in ((2, big, 20), (0, small, 10)):
            cutouts.append({"cutout_id": f"{sid}:{cam}", "file": f, "rpc_category_id": sid, "camera": cam,
                            "size": [side, side], "bbox_source": [0, 0, side, side]})
        skus.append({"rpc_category_id": sid, "shape_group": "flat", "camera_weights": {"0": 0.3, "2": 0.7}})
    # SKUs 1 and 2: camera-2 look-alikes are 40x40 boxes, camera-0 look-alikes 30x30; SKU 3 has only 3 camera-0 ones
    for sid, n0 in ((1, 12), (2, 12), (3, 3)):
        matches[sid] = [(40 * 40, 2)] * 20 + [(30 * 30, 0)] * n0
    val = {"annotations": [{"category_id": s, "bbox": [0, 0, 40, 40]} for s in (1, 2, 3)]}
    scales = fit_scales(cutouts, skus, val, matches=matches, min_matches=10)
    k2 = scales["1:2"]["k"]
    k0 = scales["1:0"]["k"]
    assert scales["1:2"]["method"] == scales["1:0"]["method"] == "matched_area"
    assert k0 / k2 == pytest.approx((30 / 10) / (40 / 20), rel=0.05)      # each camera fits its own look-alikes
    assert scales["3:0"]["method"] == "group_ratio"
    assert scales["3:0"]["k"] / scales["3:2"]["k"] == pytest.approx(k0 / k2, rel=0.01)


def test_fit_scales_without_matches_is_unchanged(tmp_path):
    big = _rect_cutout(tmp_path, "big.png", 20, 20)
    cutouts = [{"cutout_id": "1:2", "file": big, "rpc_category_id": 1, "camera": 2, "size": [20, 20],
                "bbox_source": [0, 0, 20, 20]}]
    skus = [{"rpc_category_id": 1, "camera_weights": {"2": 1.0}}]
    val = {"annotations": [{"category_id": 1, "bbox": [0, 0, 40, 40]}] * 3}
    assert fit_scales(cutouts, skus, val) == fit_scales(cutouts, skus, val, matches=None)
    assert fit_scales(cutouts, skus, val)["1:2"]["method"] == "rotated_area"


def test_fit_scales_without_a_group_ratio_uses_the_reference_camera_k(tmp_path):
    big, small = _rect_cutout(tmp_path, "big.png", 20, 20), _rect_cutout(tmp_path, "small.png", 10, 10)
    cutouts = [{"cutout_id": f"1:{cam}", "file": f, "rpc_category_id": 1, "camera": cam, "size": [side, side],
                "bbox_source": [0, 0, side, side]} for cam, f, side in ((2, big, 20), (0, small, 10))]
    skus = [{"rpc_category_id": 1, "shape_group": "flat", "camera_weights": {"0": 0.3, "2": 0.7}}]
    val = {"annotations": [{"category_id": 1, "bbox": [0, 0, 40, 40]}]}
    scales = fit_scales(cutouts, skus, val, matches={1: [(1600, 2)] * 20 + [(900, 0)] * 3}, min_matches=10)
    assert scales["1:0"]["method"] == "ref_camera_k" and scales["1:0"]["k"] == scales["1:2"]["k"]
