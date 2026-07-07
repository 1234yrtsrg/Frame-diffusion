from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from syntheticdata.json_dataset import ARKIT_DIM, VideoSequence, write_video_json
from syntheticdata.keyframe_annotation import (
    annotate_dataset_root,
    annotate_sequence_file,
    compute_energy,
    pick_keyframes,
)


def _two_peak_signal(num_frames: int = 41) -> np.ndarray:
    """Synthetic ARKit clip with two strong peaks at t=10 and t=30."""
    blendshapes = np.zeros((num_frames, ARKIT_DIM), dtype=np.float32)
    timeline = np.arange(num_frames)
    blendshapes[:, 0] = np.exp(-((timeline - 10) ** 2) / 6.0)  # peak 1
    blendshapes[:, 1] = np.exp(-((timeline - 30) ** 2) / 6.0)  # peak 2
    return blendshapes


class PickKeyframesTests(unittest.TestCase):
    def test_endpoints_always_included(self) -> None:
        blendshapes = _two_peak_signal()
        annotation = pick_keyframes(blendshapes)
        self.assertEqual(annotation.indices[0], 0)
        self.assertEqual(annotation.indices[-1], blendshapes.shape[0] - 1)

    def test_detects_both_peaks(self) -> None:
        blendshapes = _two_peak_signal()
        annotation = pick_keyframes(blendshapes)
        peak_set = set(annotation.peak_indices.tolist())
        # Peaks should land within 1 frame of the synthesized centers.
        self.assertTrue(any(abs(index - 10) <= 1 for index in peak_set))
        self.assertTrue(any(abs(index - 30) <= 1 for index in peak_set))

    def test_single_frame_returns_anchor_only(self) -> None:
        blendshapes = np.zeros((1, ARKIT_DIM), dtype=np.float32)
        annotation = pick_keyframes(blendshapes)
        self.assertEqual(annotation.indices.tolist(), [0])
        self.assertEqual(annotation.method, "anchor_only")

    def test_two_frame_keeps_both_anchors(self) -> None:
        blendshapes = np.zeros((2, ARKIT_DIM), dtype=np.float32)
        annotation = pick_keyframes(blendshapes)
        self.assertEqual(annotation.indices.tolist(), [0, 1])

    def test_neutral_clip_returns_anchors_only(self) -> None:
        blendshapes = np.zeros((30, ARKIT_DIM), dtype=np.float32)
        annotation = pick_keyframes(blendshapes)
        self.assertEqual(annotation.indices.tolist(), [0, 29])
        self.assertEqual(annotation.method, "anchor_only")
        self.assertEqual(annotation.peak_indices.size, 0)

    def test_indices_sorted_and_unique(self) -> None:
        blendshapes = _two_peak_signal(num_frames=60)
        annotation = pick_keyframes(blendshapes)
        indices = annotation.indices.tolist()
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(indices), len(set(indices)))

    def test_max_keyframes_cap_respected(self) -> None:
        # Build a signal with many evenly-spaced peaks.
        blendshapes = np.zeros((101, ARKIT_DIM), dtype=np.float32)
        for center in range(5, 100, 8):
            window = np.arange(101)
            blendshapes[:, 0] += np.exp(-((window - center) ** 2) / 1.5)
        annotation = pick_keyframes(blendshapes, max_keyframes=6)
        self.assertLessEqual(annotation.indices.size, 6)

    def test_peaks_near_boundary_dropped(self) -> None:
        # Peak at frame 1 should be suppressed by the boundary band.
        blendshapes = np.zeros((20, ARKIT_DIM), dtype=np.float32)
        blendshapes[1, 0] = 1.0
        blendshapes[10, 1] = 1.0
        annotation = pick_keyframes(blendshapes, method="peak", min_distance=3)
        self.assertNotIn(1, annotation.peak_indices.tolist())

    def test_energy_modes_finite(self) -> None:
        blendshapes = _two_peak_signal()
        for mode in ("l1", "l2", "linf"):
            energy = compute_energy(blendshapes, mode)
            self.assertEqual(energy.shape, (blendshapes.shape[0],))
            self.assertTrue(np.all(np.isfinite(energy)))


class RdpKeyframeTests(unittest.TestCase):
    def test_pure_linear_ramp_has_no_interior_keyframe(self) -> None:
        # A frame sequence that is *already* a straight line in blendshape
        # space is perfectly reconstructed by linear interpolation, so RDP
        # should keep only the two endpoints.
        ramp = np.linspace(0.0, 1.0, 25, dtype=np.float32)[:, None] * np.ones(
            (1, ARKIT_DIM), dtype=np.float32
        )
        annotation = pick_keyframes(ramp, method="rdp", epsilon=0.02)
        self.assertEqual(annotation.indices.tolist(), [0, 24])
        self.assertEqual(annotation.method, "anchor_only")
        self.assertLessEqual(annotation.max_error, 0.02)

    def test_corner_is_selected(self) -> None:
        # Triangle wave: rises to a peak at frame 12 then falls. The corner is
        # NOT an energy local-max artifact -- it is the one frame a single
        # linear segment from 0..24 cannot reproduce. RDP must keep it.
        seq = np.zeros((25, ARKIT_DIM), dtype=np.float32)
        seq[:13, 0] = np.linspace(0.0, 1.0, 13)
        seq[12:, 0] = np.linspace(1.0, 0.0, 13)
        annotation = pick_keyframes(seq, method="rdp", epsilon=0.02)
        self.assertIn(12, annotation.indices.tolist())

    def test_smaller_epsilon_keeps_more_keyframes(self) -> None:
        blendshapes = _two_peak_signal(num_frames=60)
        coarse = pick_keyframes(blendshapes, method="rdp", epsilon=0.20)
        fine = pick_keyframes(blendshapes, method="rdp", epsilon=0.01)
        self.assertGreaterEqual(fine.indices.size, coarse.indices.size)

    def test_max_keyframes_cap_respected(self) -> None:
        blendshapes = np.zeros((101, ARKIT_DIM), dtype=np.float32)
        for center in range(5, 100, 8):
            window = np.arange(101)
            blendshapes[:, 0] += np.exp(-((window - center) ** 2) / 1.5)
        annotation = pick_keyframes(
            blendshapes, method="rdp", epsilon=0.001, max_keyframes=6
        )
        self.assertLessEqual(annotation.indices.size, 6)

    def test_reconstruction_within_epsilon_when_uncapped(self) -> None:
        blendshapes = _two_peak_signal(num_frames=80)
        eps = 0.03
        annotation = pick_keyframes(
            blendshapes, method="rdp", epsilon=eps, max_keyframes=80
        )
        self.assertLessEqual(annotation.max_error, eps + 1e-6)

    def test_endpoints_always_included(self) -> None:
        blendshapes = _two_peak_signal()
        annotation = pick_keyframes(blendshapes, method="rdp")
        self.assertEqual(annotation.indices[0], 0)
        self.assertEqual(annotation.indices[-1], blendshapes.shape[0] - 1)


class AnnotateSequenceFileTests(unittest.TestCase):
    def _write_demo(self, tmp: Path, num_frames: int = 41) -> Path:
        blendshapes = _two_peak_signal(num_frames)
        seq = VideoSequence(
            video_id="demo_two_peaks",
            dataset="demo",
            source_representation="arkit52",
            native_fps=30.0,
            target_fps=10.0,
            blendshapes=blendshapes,
            timestamps_ms=(np.arange(num_frames) * 100).astype(np.int64),
        )
        path = tmp / "demo.json"
        write_video_json(seq, path)
        return path

    def test_writes_keyframe_indices_into_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_demo(Path(temp_dir))
            entry = annotate_sequence_file(path)

            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("keyframe_indices", document)
            self.assertIn("keyframe_annotation", document)
            self.assertEqual(document["keyframe_annotation"]["energy_mode"], "l2")
            self.assertGreaterEqual(entry["num_keyframes"], 4)
            self.assertEqual(document["keyframe_indices"][0], 0)
            self.assertEqual(
                document["keyframe_indices"][-1], document["num_frames"] - 1
            )

    def test_overwrite_false_keeps_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_demo(Path(temp_dir))
            annotate_sequence_file(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["keyframe_indices"] = [0, 5, 10]
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

            entry = annotate_sequence_file(path, overwrite=False)
            self.assertTrue(entry.get("skipped"))
            document_after = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document_after["keyframe_indices"], [0, 5, 10])


class AnnotateDatasetRootTests(unittest.TestCase):
    def test_summary_counts_videos_per_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(3):
                seq = VideoSequence(
                    video_id=f"demo_{index:04d}",
                    dataset="demo",
                    source_representation="arkit52",
                    native_fps=30.0,
                    target_fps=10.0,
                    blendshapes=_two_peak_signal(),
                    timestamps_ms=(np.arange(41) * 100).astype(np.int64),
                )
                write_video_json(seq, root / "demo" / f"demo_{index:04d}.json")

            summary = annotate_dataset_root(root)

            self.assertEqual(summary["total_videos_annotated"], 3)
            self.assertEqual(summary["datasets"]["demo"]["videos"], 3)
            self.assertGreaterEqual(
                summary["datasets"]["demo"]["peak_plus_anchor"], 3
            )


if __name__ == "__main__":
    unittest.main()
