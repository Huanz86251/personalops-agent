from scripts.appworld_batch_stats import distribution, percentile, wilson


def test_percentile_uses_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.8499999999999996


def test_distribution_keeps_empty_values_unknown():
    assert distribution([]) == {
        "count": 0, "mean": None, "p50": None,
        "p95": None, "min": None, "max": None,
    }


def test_wilson_interval_contains_observed_rate():
    low, high = wilson(67, 84)
    assert low < 67 / 84 < high
