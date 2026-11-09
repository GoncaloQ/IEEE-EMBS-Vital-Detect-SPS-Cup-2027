from dataclasses import replace

import numpy as np
import pytest

from vital_detect.config import Config
from vital_detect.datasets.ubfc import discover, read_ground_truth


def test_discovery_counts_missing_files(dataset):
    (dataset / "subject999").write_text("unrelated file")
    recordings = discover(dataset)
    assert [r.subject_id for r in recordings] == ["subject1", "subject2", "subject3"]
    recordings[0].validate()
    with pytest.raises(FileNotFoundError):
        recordings[2].validate()
    assert discover(dataset, ["subject2"])[0].recording_id == "subject2/vid.avi"
    with pytest.raises(ValueError):
        discover(dataset, ["subject1", "subject1"])
    with pytest.raises(ValueError):
        discover(dataset, ["../subject1"])
    with pytest.raises(ValueError):
        replace(recordings[0], reference_path=recordings[1].reference_path).validate()
    with pytest.raises(ValueError, match="Recording ID"):
        replace(recordings[0], recording_id="subject2/vid.avi").validate()


def test_symlink_reference_cannot_mix_subjects(dataset):
    first, second, _ = discover(dataset)
    first.reference_path.unlink()
    first.reference_path.symlink_to(second.reference_path)
    with pytest.raises(ValueError, match="outside"):
        first.validate()
    (dataset / "subject4").symlink_to(dataset / "subject2", target_is_directory=True)
    with pytest.raises(ValueError, match="aliases"):
        discover(dataset, ["subject4"])[0].validate()


def test_verified_format_preserves_units_and_time(tmp_path):
    path = tmp_path / "ground_truth.txt"
    path.write_text("-2 0 2 0\n60 70 80 90\n2e0 2.1e0 2.7e0 3e0\n")
    gt = read_ground_truth(path)
    np.testing.assert_array_equal(gt.ppg, [-2, 0, 2, 0])
    assert gt.timestamps_s[0] == 2

    expected = 0.1 * 65 + 0.6 * 75 + 0.3 * 85
    assert gt.mean_interval(2, 3)[0] == pytest.approx(expected)
    assert gt.mean_interval(2.1, 2.7)[0] == pytest.approx(75)
    with pytest.raises(ValueError, match="coverage"):
        gt.mean_interval(0, 1)
    with pytest.raises(ValueError, match="gap"):
        gt.mean_interval(2, 3, max_gap_s=0.2)
    aligned = gt.aligned_times(0, Config(sync_policy="recording-start"))
    assert aligned[0] == 0
    assert gt.timestamps_s[0] == 2
    assert gt.aligned_times(0, Config())[0] == 2


def test_duplicate_instants_preserved_and_coalesced_only_for_integration(tmp_path):
    path = tmp_path / "ground_truth.txt"
    path.write_text("0 1 2 3\n60 70 90 100\n0 .5 .5 1\n")
    gt = read_ground_truth(path)
    assert gt.duplicate_timestamp_count == 1
    assert len(gt.timestamps_s) == 4
    assert gt.mean_interval(0, 1)[0] == pytest.approx(80)


def test_reference_whitespace_does_not_change_format_or_units(tmp_path):
    path = tmp_path / "ground_truth.txt"
    path.write_text("\ufeff\n0 1 0\n60 70 80\n0 .5 1\n\n  \n")
    result = read_ground_truth(path)
    assert result.mean_interval(0, 1)[0] == pytest.approx(70)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "0 1\n60 70",
        "0\n60\n0\n",
        "0 bad\n60 70\n0 .1\n",
        "0 nan\n60 70\n0 .1\n",
        "0 1\n0 70\n0 .1\n",
        "0 1\n60 70\n.1 0\n",
        "0 1\n60 70\n0 0\n",
        "0 1\n60 70\n0 33\n",
        "0 1\n.1 .2\n60 70\n",
        "0 1\n60\n0 .1\n",
        "0 1\n60 inf\n0 .1\n",
    ],
)
def test_malformed_reference(tmp_path, content):
    path = tmp_path / "ground_truth.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_ground_truth(path)
