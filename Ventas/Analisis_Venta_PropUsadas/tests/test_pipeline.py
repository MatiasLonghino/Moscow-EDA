from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_data
from cleaning import fit_cleaning, apply_cleaning
from encoding import fit_encoding, apply_encoding
from features import build_features
from ols import fit_ols, predict_ols, evaluate_ols, diagnose_ols
from selection import SELECTORS
from pipeline import run_pipeline, predict_pipeline


@pytest.fixture
def datasets(tmp_path):
    rng = np.random.default_rng(73)
    n = 90
    area = rng.normal(65, 12, n)
    area[5] = np.nan
    sale = pd.DataFrame({"id": [f"S{i}" for i in range(n)],
                         "date_posted": pd.date_range("2020-02-01", periods=n),
                         "district": "D", "okrug": "O", "metro_station": "M", "metro_line": "L",
                         "total_area": area, "rooms": rng.integers(1, 5, n),
                         "renovation": np.where(np.arange(n) % 2, "yes", "no"),
                         "price_rub": 1000000 + np.nan_to_num(area, nan=65) * 100000 + rng.normal(0, 1000, n),
                         "price_per_sqm": 999999})
    monthly = pd.DataFrame({"year_month": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"],
                            "district": "D", "okrug": "O", "secondary_price_per_sqm": [1, 2, 3, 4, 9999]})
    metro = pd.DataFrame({"station_name": ["M"], "line": ["L"], "lat": [55.7], "lon": [37.6], "year_opened": [1980]})
    paths = [tmp_path / name for name in ("sales.csv", "monthly.csv", "metro.csv")]
    for frame, path in zip((sale, monthly, metro), paths):
        frame.to_csv(path, index=False)
    return paths


def run(paths, **kwargs):
    return run_pipeline(*paths, train_end="2020-03-01", validation_end="2020-04-01",
                        include=["total_area", "rooms", "renovation"], **kwargs)


def test_join_integrity_and_lag(datasets):
    frame = load_data(*datasets, require_matches=True)
    assert len(frame) == 90
    assert frame.iloc[0].monthly_secondary_price_per_sqm == 1
    assert frame.attrs["join_audit"]["metro_unmatched"] == 0
    assert frame.id.is_unique


@pytest.mark.parametrize("table", [0, 1, 2])
def test_duplicate_keys_rejected(datasets, table):
    frame = pd.read_csv(datasets[table])
    pd.concat([frame, frame.iloc[[0]]]).to_csv(datasets[table], index=False)
    with pytest.raises(ValueError, match="duplicad"):
        load_data(*datasets)


def test_unmatched_rows_retained(datasets):
    frame = pd.read_csv(datasets[2])
    frame["station_name"] = "unknown"
    frame.to_csv(datasets[2], index=False)
    result = load_data(*datasets)
    assert len(result) == 90
    assert result.attrs["join_audit"]["metro_unmatched"] == 90
    with pytest.raises(ValueError, match="incompletos"):
        load_data(*datasets, require_matches=True)


def test_imputation_train_only():
    train = pd.DataFrame({"a": [1., 3., np.nan], "empty": [np.nan] * 3, "cat": ["a", None, "b"]})
    state = fit_cleaning(train)
    future = pd.DataFrame({"a": [np.nan, 10000], "empty": [np.nan, np.nan], "cat": [None, "z"]})
    result = apply_cleaning(future, state)
    assert result.a.tolist() == [2, 10000]
    assert result.empty is False
    assert result["empty"].tolist() == [0, 0]
    assert not result.isna().any().any()
    assert state["fill"]["a"] == 2


def test_encoding_schema_unknown_and_groups():
    train = pd.DataFrame({"a": [1., 2., 3.], "cat": ["a", "b", "c"]})
    state = fit_encoding(train)
    result = apply_encoding(pd.DataFrame({"a": [4.], "cat": ["future"]}), state)
    assert list(result) == list(apply_encoding(train, state))
    assert len(state["groups"]["cat"]) == 2
    assert result[state["groups"]["cat"]].to_numpy().sum() == 0
    with pytest.raises(ValueError, match="desconocidas"):
        apply_encoding(pd.DataFrame({"a": [4.], "cat": ["future"]}), state, "error")


def test_forbidden_features(datasets):
    raw = load_data(*datasets)
    assert not {"id", "price_per_sqm", "price_rub", "date_posted", "year_month"} & set(build_features(raw))
    with pytest.raises(ValueError, match="admisibles"):
        build_features(raw, include=["price_per_sqm"])
    with pytest.raises(ValueError, match=">= 1"):
        load_data(*datasets, monthly_lag_months=0)


@pytest.fixture
def regression():
    rng = np.random.default_rng(91)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["signal", "noise", "other"])
    y = 5 + 100 * X.signal + rng.normal(scale=0.1, size=100)
    return X, y


@pytest.mark.parametrize("method", ["none", "forward", "backward", "stepwise", "best_subset"])
@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_selectors(regression, method, criterion):
    X, y = regression
    result = SELECTORS[method](X, y, criterion=criterion)
    assert "signal" in result["selected_features"]
    assert set(result["selected_features"]) <= set(X)
    assert result == SELECTORS[method](X, y, criterion=criterion)


@pytest.mark.parametrize("method", ["forward", "backward", "stepwise"])
def test_group_significance(regression, method):
    X, y = regression
    groups = {"signal_block": ["signal", "other"], "noise": ["noise"]}
    result = SELECTORS[method](X, y, groups=groups, criterion="pvalue")
    assert {"signal", "other"} <= set(result["selected_features"])


def test_best_subset_limits(regression):
    X, y = regression
    with pytest.raises(ValueError, match="límites"):
        SELECTORS["best_subset"](X, y, max_models=2)


def test_fit_prediction_and_metrics(regression):
    X, y = regression
    model = fit_ols(X, y)
    assert abs(model.params["signal"] - 100) < 0.1
    assert len(predict_ols(model, X)) == 100
    metrics = evaluate_ols(model, X, y, "train")
    assert metrics["train_r2"] > 0.99
    assert "validation_aic" not in evaluate_ols(model, X, y)
    assert not diagnose_ols(model, X)["rank_deficient"]
    intercept = fit_ols(X[[]], y)
    np.testing.assert_allclose(predict_ols(intercept, X), y.mean())


def test_pipeline_temporal_no_leakage_reproducible(datasets):
    original = run(datasets, selection_method="forward", return_frames=True)
    again = run(datasets, selection_method="forward")
    assert original["metrics"] == again["metrics"]
    assert original["selected_features"] == again["selected_features"]
    raw = pd.read_csv(datasets[0])
    future = raw.date_posted >= "2020-03-01"
    raw.loc[future, "price_rub"] *= 100
    raw.loc[future, "total_area"] = 1e6
    raw.loc[future, "renovation"] = "unseen"
    raw.to_csv(datasets[0], index=False)
    changed = run(datasets, selection_method="forward", return_frames=True)
    assert original["preprocessor"] == changed["preprocessor"]
    assert original["selection"] == changed["selection"]
    np.testing.assert_allclose(original["model"].params, changed["model"].params)
    assert original["metrics"]["validation_rmse"] != changed["metrics"]["validation_rmse"]
    assert original["frames"]["prepared"]["train"].index.equals(original["frames"]["encoded"]["train"].index)
    assert "frames" not in again
    prediction = predict_pipeline(original, load_data(*datasets))
    assert len(prediction) == 90


def test_empty_partition_rejected(datasets):
    with pytest.raises(ValueError, match="particiones"):
        run_pipeline(*datasets, train_end="2030-01-01", validation_end="2031-01-01")


def test_bad_target_rejected(datasets):
    raw = pd.read_csv(datasets[0])
    raw.loc[0, "price_rub"] = np.nan
    raw.to_csv(datasets[0], index=False)
    with pytest.raises(ValueError, match="positivos"):
        run(datasets)


def test_future_monthly_values_not_used(datasets):
    original = run_pipeline(*datasets, train_end="2020-03-01", validation_end="2020-04-01",
                            include=["total_area", "monthly_secondary_price_per_sqm"], return_frames=True)
    monthly = pd.read_csv(datasets[1])
    monthly.loc[monthly.year_month >= "2020-03", "secondary_price_per_sqm"] = 1e9
    monthly.to_csv(datasets[1], index=False)
    changed = run_pipeline(*datasets, train_end="2020-03-01", validation_end="2020-04-01",
                           include=["total_area", "monthly_secondary_price_per_sqm"], return_frames=True)
    pd.testing.assert_frame_equal(original["frames"]["encoded"]["train"], changed["frames"]["encoded"]["train"])
    pd.testing.assert_frame_equal(original["frames"]["encoded"]["validation"], changed["frames"]["encoded"]["validation"])
    np.testing.assert_allclose(original["model"].params, changed["model"].params)


def test_collinearity_report(regression):
    X, y = regression
    X = X.copy()
    X["duplicate"] = 2 * X.signal
    model = fit_ols(X, y)
    diagnostic = diagnose_ols(model, X)
    assert diagnostic["rank_deficient"]
    assert np.isinf(diagnostic["vif"]["signal"])


@pytest.mark.parametrize("uri", [None, "", "file:///tmp/mlruns", "./mlruns", "https://user:password@example.test", "https://example.test?token=secret"])
def test_tracking_requires_external_server(monkeypatch, uri):
    from pipeline import get_tracking_client
    if uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(ValueError):
        get_tracking_client()


def test_tracking_connection_error_redacts_secrets(monkeypatch):
    import mlflow.tracking
    from pipeline import get_tracking_client
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.test")
    def broken(**kwargs):
        raise OSError("secret-token-must-not-appear")
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", broken)
    with pytest.raises(RuntimeError, match="No se pudo conectar") as error:
        get_tracking_client()
    assert "secret-token" not in str(error.value)


def test_tracking_client_checks_connection(monkeypatch):
    import mlflow.tracking
    from pipeline import get_tracking_client
    from unittest.mock import Mock
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.test")
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", factory)
    assert get_tracking_client() is client
    factory.assert_called_once_with(tracking_uri="https://example.test")
    client.search_experiments.assert_called_once_with(max_results=1)


def test_tracking_disabled_never_connects(datasets, monkeypatch):
    import pipeline
    def forbidden():
        raise AssertionError("No debe conectar")
    monkeypatch.setattr(pipeline, "get_tracking_client", forbidden)
    assert run(datasets)["run_id"] is None


def test_tracking_missing_uri_fails_before_training(monkeypatch):
    import pipeline
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("No debe cargar ni entrenar sin tracking configurado")
    monkeypatch.setattr(pipeline, "load_data", forbidden)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        pipeline.run_pipeline(tracking=True)


def test_tracking_artifacts_and_failed_run(datasets, monkeypatch, tmp_path):
    import json
    import pickle
    import tempfile
    import pipeline
    from types import SimpleNamespace
    from unittest.mock import Mock
    monkeypatch.chdir(tmp_path)
    actual_temporary = tempfile.TemporaryDirectory
    monkeypatch.setattr(pipeline.tempfile, "TemporaryDirectory",
                        lambda **kwargs: actual_temporary(dir=tmp_path, **kwargs))
    client = Mock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="E1")
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="R1"))
    captured = {}
    def capture(run_id, folder):
        root = Path(folder)
        captured["config"] = json.loads((root / "config.json").read_text(encoding="utf-8"))
        captured["features"] = json.loads((root / "features.json").read_text(encoding="utf-8"))
        captured["model"] = pickle.loads((root / "model.pkl").read_bytes())
        assert (root / "diagnostics.png").read_bytes().startswith(b"\x89PNG")
        assert (root / "source" / "pipeline.py").is_file()
        assert (root / "source_manifest.json").is_file()
    client.log_artifacts.side_effect = capture
    monkeypatch.setattr(pipeline, "get_tracking_client", lambda: client)
    result = run(datasets, tracking=True)
    assert result["run_id"] == "R1"
    assert captured["features"]["selected"] == result["selected_features"]
    assert captured["config"]["train_end"].startswith("2020-03-01")
    restored = predict_pipeline(captured["model"], load_data(*datasets))
    np.testing.assert_allclose(restored, predict_pipeline(result, load_data(*datasets)))
    client.set_terminated.assert_called_with("R1", status="FINISHED")
    assert not (tmp_path / "mlruns").exists()
    client.log_artifacts.side_effect = OSError("secret-token")
    with pytest.raises(RuntimeError, match="Tracking MLflow") as error:
        pipeline.track_experiment(result)
    assert "secret-token" not in str(error.value)
    assert error.value.result is result
    client.set_terminated.assert_called_with("R1", status="FAILED")


def test_compare_experiments(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import scripts.compare_experiments as comparison
    client = Mock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="E1")
    client.search_runs.return_value = [SimpleNamespace(info=SimpleNamespace(run_id="R1", status="FINISHED"),
        data=SimpleNamespace(params={"selection_method": "forward"}, metrics={"validation_rmse": 100}, tags={}))]
    monkeypatch.setattr(comparison, "get_tracking_client", lambda: client)
    result = comparison.compare_experiments()
    assert result.iloc[0]["metric.validation_rmse"] == 100
    assert result.iloc[0]["param.selection_method"] == "forward"
    assert client.search_runs.call_args.kwargs["order_by"] == ["metrics.validation_rmse ASC"]
