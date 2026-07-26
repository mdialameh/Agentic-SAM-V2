"""Unit tests for the pure-logic parts of core/ (no GPU, no network)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.imaging import (
    SegmentationResult,
    bbox_from_mask,
    box_around_point,
    box_text,
    keep_smallest_box_candidate,
    measure_mask,
    parse_box,
    points_to_box,
    render_overlay,
    to_numpy_mask,
)
from core.procedure import RFA_PHASES, ProcedureLog
from core.video import is_lfs_pointer_stub


class TestBoxes:
    def test_points_to_box_orders_corners(self):
        assert points_to_box([[100, 50], [10, 200]]) == [10, 50, 100, 200]

    def test_points_to_box_needs_two_points(self):
        assert points_to_box([[10, 20]]) is None

    def test_parse_box_roundtrip(self):
        assert parse_box("1, 2, 3.5, 4") == [1.0, 2.0, 3.5, 4.0]
        assert parse_box([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("bad", ["1,2,3", "3,2,1,4", "1,4,3,2"])
    def test_parse_box_rejects_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_box(bad)

    def test_box_around_point_clamps(self):
        image = Image.new("RGB", (200, 100))
        assert box_around_point(0, 0, image, 96)[:2] == [0, 0]
        assert box_around_point(199, 99, image, 96)[2:] == [199, 99]

    def test_box_text(self):
        assert box_text([1.4, 2.6, 3.0, 4.0]) == "1,3,3,4"
        assert box_text(None) == ""


class TestMasks:
    def _mask(self):
        mask = np.zeros((100, 100), bool)
        mask[20:40, 30:60] = True
        return mask

    def test_bbox_and_area(self):
        mask = self._mask()
        assert bbox_from_mask(mask) == [30.0, 20.0, 59.0, 39.0]
        assert measure_mask(mask).area_px == 600

    def test_measure_with_spacing(self):
        result = measure_mask(self._mask(), pixel_spacing_mm=0.1)
        assert result.area_mm2 == pytest.approx(6.0)
        assert "mm" in result.describe()

    def test_to_numpy_mask_squeezes(self):
        assert to_numpy_mask(np.ones((1, 1, 5, 5))).shape == (5, 5)

    def test_render_overlay_shapes(self):
        image = Image.new("RGB", (100, 100))
        overlay = render_overlay(image, [self._mask()], [[30, 20, 59, 39]])
        assert overlay.size == (100, 100)

    def test_smallest_candidate_rule(self):
        mask = self._mask()
        result = SegmentationResult(
            source="t", query="PTMC", masks=[mask, mask],
            boxes=[[0, 0, 50, 50], [0, 0, 10, 10]], scores=[0.9, 0.8],
        )
        result = keep_smallest_box_candidate(result)
        assert result.num_masks == 1
        assert result.boxes == [[0, 0, 10, 10]]


class TestProcedureLog:
    def test_phases_and_events(self, tmp_path, monkeypatch):
        import core.procedure as procedure

        monkeypatch.setattr(procedure, "SESSION_DIR", tmp_path)
        log = ProcedureLog("t1")
        log.set_phase(RFA_PHASES[1])
        log.set_phase(RFA_PHASES[1])  # no duplicate event
        kinds = [event.kind for event in log.events]
        assert kinds == ["note", "phase_change"]

    def test_area_trend_flags_drop(self, tmp_path, monkeypatch):
        import core.procedure as procedure

        monkeypatch.setattr(procedure, "SESSION_DIR", tmp_path)
        log = ProcedureLog("t2")
        log.record_area(1, 100.0)
        summary = log.record_area(2, 80.0)
        assert summary["estimated_area_not_captured_percent"] == 20
        assert summary["size_drop_flag"] is True
        summary = log.record_area(3, 95.0)
        assert summary["size_drop_flag"] is False

    def test_snapshot_saves_image(self, tmp_path, monkeypatch):
        import core.procedure as procedure

        monkeypatch.setattr(procedure, "SESSION_DIR", tmp_path)
        log = ProcedureLog("t3")
        event = log.add("snapshot", "snap", image=Image.new("RGB", (8, 8)))
        assert event.image_path and not is_lfs_pointer_stub(event.image_path)
        assert log.snapshots() == [event]

    def test_agent_context_mentions_phase(self, tmp_path, monkeypatch):
        import core.procedure as procedure

        monkeypatch.setattr(procedure, "SESSION_DIR", tmp_path)
        log = ProcedureLog("t4")
        log.set_phase("Ablation")
        assert "Ablation" in log.context_for_agent()


class TestAblationTracker:
    def _setup(self):
        from core.ablation import AblationTracker

        baseline = np.zeros((100, 100), bool)
        baseline[20:60, 20:60] = True  # 40x40 PTMC = 1600 px
        tracker = AblationTracker(pixel_spacing_mm=0.1)
        tracker.set_baseline(10, baseline, Image.new("RGB", (100, 100), (60, 60, 60)))
        return tracker, baseline

    def test_baseline(self):
        tracker, _ = self._setup()
        assert tracker.active and tracker.baseline_area_px == 1600
        assert tracker.coverage_percent == 0.0 and tracker.residual_percent == 100.0

    def test_progressive_coverage(self):
        tracker, baseline = self._setup()
        # sample 1: top half of the PTMC no longer captured -> 50% treated
        half = baseline.copy(); half[20:40, :] = False
        s1 = tracker.add_sample(20, 5.0, half, Image.new("RGB", (100, 100), (90, 90, 90)))
        assert s1.captured_ratio == 0.5
        assert tracker.coverage_percent == 50.0
        # sample 2: full mask captured again -> coverage stays (cumulative)
        tracker.add_sample(30, 10.0, baseline, Image.new("RGB", (100, 100), (120, 120, 120)))
        assert tracker.coverage_percent == 50.0
        # sample 3: everything lost -> full coverage, residual 0
        empty = np.zeros_like(baseline)
        tracker.add_sample(40, 15.0, empty, Image.new("RGB", (100, 100), (150, 150, 150)))
        assert tracker.coverage_percent == 100.0
        assert tracker.residual_percent == 0.0

    def test_summary_and_timeline(self):
        tracker, baseline = self._setup()
        half = baseline.copy(); half[20:40, :] = False
        tracker.add_sample(20, 5.0, half, Image.new("RGB", (100, 100), (90, 90, 90)))
        summary = tracker.summary()
        assert summary["active"] and summary["coverage_percent"] == 50.0
        assert summary["baseline_area_mm2"] == 16.0
        assert summary["residual_bbox"] is not None
        assert len(tracker.timeline()) == 1
        # brightness increases across samples are reported
        tracker.add_sample(30, 10.0, half, Image.new("RGB", (100, 100), (150, 150, 150)))
        assert tracker.summary()["brightness_change"] > 0

    def test_coverage_map_renders(self):
        tracker, baseline = self._setup()
        half = baseline.copy(); half[20:40, :] = False
        tracker.add_sample(20, 5.0, half, Image.new("RGB", (100, 100)))
        cov = tracker.coverage_map()
        assert cov is not None and cov.size == (100, 100)

    def test_fallback_ablation_report(self):
        from core.agent.report import _fallback_ablation_report

        tracker, baseline = self._setup()
        half = baseline.copy(); half[20:40, :] = False
        tracker.add_sample(20, 5.0, half, Image.new("RGB", (100, 100)))
        report = _fallback_ablation_report(tracker)
        assert "50.0%" in report and "Coverage" in report


class TestDeterministicRouting:
    """Safety-critical: these questions must never be answered from model guesswork."""

    @pytest.mark.parametrize(
        "question",
        [
            "How much of the PTMC is ablated so far?",
            "Did I cover the whole PTMC?",
            "Is there any residual tumor left?",
            "Any part of the PTMC untreated?",
            "What is the final ablation pattern?",
            "How much is remaining?",
        ],
    )
    def test_ablation_questions_route_to_tracker(self, question):
        from core.agent.graph import _should_force_ablation_status

        assert _should_force_ablation_status(question)

    @pytest.mark.parametrize(
        "question",
        ["Describe this ultrasound image.", "Is the tracking stable?", "What is RFA?"],
    )
    def test_non_ablation_questions_use_the_graph(self, question):
        from core.agent.graph import _should_force_ablation_status

        assert not _should_force_ablation_status(question)

    @pytest.mark.parametrize(
        "question",
        [
            "Where is the PTMC in this ultrasound image?",
            "Segment the nodule please",
            "Show the tumor boundaries",
        ],
    )
    def test_box_segmentation_questions_force_medsam2(self, question):
        from core.agent.graph import _should_force_box_segmentation

        assert _should_force_box_segmentation(question)


class TestVideo:
    def test_lfs_stub_detection(self, tmp_path):
        stub = tmp_path / "stub.mp4"
        stub.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 5\n")
        real = tmp_path / "real.mp4"
        real.write_bytes(b"\x00" * 2000)
        assert is_lfs_pointer_stub(stub) is True
        assert is_lfs_pointer_stub(real) is False
