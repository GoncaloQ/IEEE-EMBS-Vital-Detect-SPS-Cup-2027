import math

import pytest

from vital_detect.evaluation.metrics import calculate_metrics


def test_known_metrics():
    m = calculate_metrics([61, 72, 83], [60, 70, 80])
    assert m["mae_bpm"] == 2
    assert m["rmse_bpm"] == pytest.approx(math.sqrt(14 / 3))
    assert m["pearson_r"] == pytest.approx(1)
    ba = m["bland_altman"]
    assert ba["bias_bpm"] == 2
    assert ba["sample_sd_bpm"] == 1
    assert ba["lower_loa_bpm"] == pytest.approx(0.04)
    assert ba["upper_loa_bpm"] == pytest.approx(3.96)
    assert calculate_metrics([1, 2, 3], [1, 3, 2])["pearson_r"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "e,r", [([], []), ([70], [65]), ([70, 70], [60, 80]), ([60, 80], [70, 70])]
)
def test_undefined_correlation(e, r):
    result = calculate_metrics(e, r)
    assert result["pearson_r"] is None
    assert "pearson_r" in result["undefined_reasons"]
    if len(e) < 2:
        assert result["bland_altman"]["lower_loa_bpm"] is None


@pytest.mark.parametrize("e,r", [([math.nan], [70]), ([70], [math.inf]), ([1], [1, 2])])
def test_invalid_pairs_are_never_silently_dropped(e, r):
    with pytest.raises(ValueError):
        calculate_metrics(e, r)


def test_finite_large_values_do_not_overflow_rmse():
    result = calculate_metrics([1e200, 2e200, 3e200], [0, 0, 0])
    assert result["rmse_bpm"] == pytest.approx(math.sqrt(14 / 3) * 1e200)
    assert math.isfinite(result["bland_altman"]["sample_sd_bpm"])


def test_negative_bias_sample_sd_and_pearson_independent_values():
    result = calculate_metrics([60, 64, 66], [65, 66, 64])
    ba = result["bland_altman"]
    assert ba["bias_bpm"] == pytest.approx(-5 / 3)
    expected_sd = math.sqrt(37 / 3)
    assert ba["sample_sd_bpm"] == pytest.approx(expected_sd)
    assert ba["lower_loa_bpm"] == pytest.approx(-5 / 3 - 1.96 * expected_sd)
    assert ba["upper_loa_bpm"] == pytest.approx(-5 / 3 + 1.96 * expected_sd)
    assert calculate_metrics([4, 1, 3], [2, 3, 1])["pearson_r"] == pytest.approx(-math.sqrt(3 / 7))


def test_unrepresentable_difference_is_explicit():
    with pytest.raises(ValueError, match="numeric range"):
        calculate_metrics([1e308], [-1e308])
