import copy
import numpy as np
import pytest
from spf_ssp126.data import calendar_rows, groups, validate_metadata, VARIABLES, UNITS
from spf_ssp126.prepare import combine_moments, verify_source_scenario


def norm():
    return dict(mean=[280, 284, 276, 0.00003], std=[10, 10, 10, 0.00005],
                scenarios=["ssp126"], fit_scope="SSP126 train only",
                variables=list(VARIABLES), units=list(UNITS))


def test_complete_calendar_and_all_epochs():
    rows = calendar_rows()
    assert len(rows) == 1031 and sum(row["days"] for row in rows) == 31359
    assert (rows[-1]["year"], rows[-1]["month"]) == (2100, 11)
    for epoch in (1, 2, 467, 534):
        batches = groups(rows, epoch)
        assert len(batches) == 45
        assert sorted(len(g) for g in batches if len(g) < 24) == [4, 8, 14, 21]
        assert sorted(i for batch in batches for i in batch) == list(range(1031))
        for batch in batches:
            assert len({rows[i]["days"] for i in batch}) == 1
            assert all(batch[rank::4] for rank in range(4))


@pytest.mark.parametrize("scenarios", [["ssp245"], ["ssp126", "ssp245"], []])
def test_other_scenario_normalization_is_rejected(scenarios):
    value = norm()
    value["scenarios"] = scenarios
    with pytest.raises(ValueError):
        validate_metadata({"status": "complete", "training_scenarios": ["ssp126"]},
                          value, calendar_rows())


def test_metadata_only_ssp126():
    validate_metadata({"status": "complete", "training_scenarios": ["ssp126"]},
                      norm(), calendar_rows())
    rows = copy.deepcopy(calendar_rows())
    rows[0]["scenario"] = "ssp245"
    with pytest.raises(ValueError):
        groups(rows, 1)
    with pytest.raises(ValueError):
        validate_metadata({"status": "complete", "training_scenarios": ["ssp126"]}, norm(), rows)


def test_streaming_train_statistics_exact():
    values = np.random.default_rng(8).normal(size=(12, 4, 8, 6))
    n, mean, m2 = 0, np.zeros(4), np.zeros(4)
    for part in (values[:5], values[5:9], values[9:]):
        n, mean, m2 = combine_moments(n, mean, m2, part)
    full = np.moveaxis(values, 1, 0).reshape(4, -1)
    np.testing.assert_allclose(mean, full.mean(1), atol=1e-14)
    np.testing.assert_allclose(np.sqrt(m2 / n), full.std(1), atol=1e-14)


def test_source_scenario_guard():
    verify_source_scenario("ssp126", "anything.nc")
    verify_source_scenario("", "model_train_ssp126.nc")
    for experiment, name in (("ssp245", "model_train_ssp126.nc"),
                             ("ssp585", "model.nc"), ("", "model_train_ssp245.nc")):
        with pytest.raises(ValueError):
            verify_source_scenario(experiment, name)
