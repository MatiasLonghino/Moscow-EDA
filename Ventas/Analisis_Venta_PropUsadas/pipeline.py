"""Coordinación funcional: entrenamiento temporal, evaluación y tracking optativo."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import tempfile
from urllib.parse import urlparse

import pandas as pd

if __package__:
    from .data import load_data, DEFAULT_DATA_DIR
    from .cleaning import validate_target, fit_cleaning, apply_cleaning
    from .features import build_features
    from .encoding import fit_encoding, apply_encoding
    from .ols import fit_ols, evaluate_ols, diagnose_ols, predict_ols, diagnostic_figure
    from .selection import SELECTORS
else:
    from data import load_data, DEFAULT_DATA_DIR
    from cleaning import validate_target, fit_cleaning, apply_cleaning
    from features import build_features
    from encoding import fit_encoding, apply_encoding
    from ols import fit_ols, evaluate_ols, diagnose_ols, predict_ols, diagnostic_figure
    from selection import SELECTORS


def run_pipeline(secondary_path=None, monthly_path=None, metro_path=None, *,
                 train_end="2024-01-01", validation_end="2025-01-01", test_end=None,
                 include=None, exclude=None, selection_method="none", criterion="aic",
                 selection_params=None, preprocessing=None, tracking=False,
                 experiment_name="Moscow_OLS", run_name=None, return_frames=False):
    """Intervalos [inicio, train_end), [train_end, validation_end), [validation_end, test_end).

    El modelo final permanece ajustado sólo en train. Validación permite comparar
    experimentos; test debe consultarse una vez tomada la decisión de modelado.
    """
    preprocessing = dict(preprocessing or {})
    allowed = {"monthly_lag_months", "require_matches", "numeric_strategy", "categorical_fill", "handle_unknown", "derive"}
    if set(preprocessing) - allowed:
        raise ValueError(f"Opciones desconocidas: {sorted(set(preprocessing) - allowed)}")
    if selection_method not in SELECTORS:
        raise ValueError(f"Método desconocido: {selection_method}")
    tracking_client = get_tracking_client() if tracking else None
    start, end = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    stop = pd.Timestamp(test_end) if test_end else None
    if pd.isna(start) or pd.isna(end) or not start < end or (stop is not None and not end < stop):
        raise ValueError("Se requiere train_end < validation_end < test_end")
    raw = load_data(secondary_path, monthly_path, metro_path,
                    monthly_lag_months=preprocessing.get("monthly_lag_months", 1),
                    require_matches=preprocessing.get("require_matches", False))
    audit = raw.attrs["join_audit"]
    raw = raw.sort_values("date_posted", kind="stable")
    date = raw["date_posted"]
    masks = {"train": date < start, "validation": (date >= start) & (date < end),
             "test": (date >= end) & ((date < stop) if stop is not None else True)}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Las tres particiones temporales deben contener filas")
    y = validate_target(raw)
    features = build_features(raw, include, exclude, derive=preprocessing.get("derive", True))
    cleaning = fit_cleaning(features.loc[masks["train"]],
                            numeric_strategy=preprocessing.get("numeric_strategy", "median"),
                            categorical_fill=preprocessing.get("categorical_fill", "__MISSING__"))
    prepared = {name: apply_cleaning(features.loc[mask], cleaning) for name, mask in masks.items()}
    encoding = fit_encoding(prepared["train"])
    encoded = {name: apply_encoding(frame, encoding, preprocessing.get("handle_unknown", "ignore"))
               for name, frame in prepared.items()}
    # Constantes decididas sólo a partir de train. Se conserva el estado completo
    # de encoding para aplicar exactamente la misma transformación en inferencia.
    usable = list(encoded["train"].columns[encoded["train"].nunique() > 1])
    dropped = [c for c in encoded["train"] if c not in usable]
    groups = {g: [c for c in cols if c in usable] for g, cols in encoding["groups"].items()}
    groups = {g: cols for g, cols in groups.items() if cols}
    X_train = encoded["train"][usable]
    selection = SELECTORS[selection_method](X_train, y.loc[masks["train"]], groups=groups,
                                           criterion=criterion, **(selection_params or {}))
    selected = selection["selected_features"]
    model = fit_ols(X_train[selected], y.loc[masks["train"]])
    metrics, predictions = {}, {}
    for name in masks:
        metrics.update(evaluate_ols(model, encoded[name][selected], y.loc[masks[name]], name))
        predictions[name] = predict_ols(model, encoded[name][selected])
    paths = {"secondary": str(Path(secondary_path or DEFAULT_DATA_DIR / "secondary_market.csv").resolve()),
             "monthly": str(Path(monthly_path or DEFAULT_DATA_DIR / "district_prices_monthly.csv").resolve()),
             "metro": str(Path(metro_path or DEFAULT_DATA_DIR / "metro_stations.csv").resolve())}
    result = {"model": model, "selected_features": selected, "selected_groups": selection["selected_groups"],
              "candidate_features": usable, "candidate_groups": groups, "selection": selection,
              "metrics": metrics, "predictions": predictions, "join_audit": audit,
              "diagnostics": diagnose_ols(model, X_train[selected]),
              "preprocessor": {"cleaning": cleaning, "encoding": encoding, "include": include,
                               "exclude": exclude, "derive": preprocessing.get("derive", True),
                               "handle_unknown": preprocessing.get("handle_unknown", "ignore")},
              "dropped_constant_columns": dropped, "run_id": None,
              "config": {"paths": paths, "train_end": str(start), "validation_end": str(end),
                         "test_end": str(stop) if stop is not None else None,
                         "include": include, "exclude": exclude, "selection_method": selection_method,
                         "criterion": criterion, "selection_params": dict(selection_params or {}),
                         "preprocessing": preprocessing, "experiment_name": experiment_name}}
    if return_frames:
        result["frames"] = {"original": raw, "features": features, "prepared": prepared, "encoded": encoded,
                            "targets": {name: y.loc[mask] for name, mask in masks.items()}}
    if tracking:
        result["run_id"] = track_experiment(result, experiment_name, run_name, client=tracking_client)
    return result


def predict_pipeline(result, integrated_frame):
    """Predice sobre un frame previamente integrado con load_data y el mismo rezago."""
    state = result["preprocessor"]
    features = build_features(integrated_frame, state["include"], state["exclude"], state["derive"])
    cleaned = apply_cleaning(features, state["cleaning"])
    encoded = apply_encoding(cleaned, state["encoding"], state["handle_unknown"])
    return predict_ols(result["model"], encoded)


def get_tracking_client():
    """Nunca recurre al almacén local por defecto. Credenciales vía entorno MLflow."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Definir MLFLOW_TRACKING_URI con la URL HTTP(S) del servidor externo")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Usar variables de autenticación de MLflow; la URI no debe incluir credenciales ni parámetros")
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise RuntimeError("Instalar mlflow-skinny para activar tracking") from None
    try:
        client = MlflowClient(tracking_uri=uri)
        client.search_experiments(max_results=1)
        return client
    except Exception as error:
        # No propagar mensajes de terceros que podrían contener tokens o URI.
        raise RuntimeError(f"No se pudo conectar a MLflow ({type(error).__name__}); verificar servidor, red y autenticación") from None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def track_experiment(result, experiment_name="Moscow_OLS", run_name=None, client=None):
    """Registra artefactos temporales en el servidor, sin persistir DataFrames.

    Los errores se propagan: nunca se comunica una ejecución sin tracking como
    registrada. El resultado calculado queda accesible en error.result.
    """
    run_id = None
    try:
        client = client if client is not None else get_tracking_client()
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
        run = client.create_run(experiment_id, tags={"mlflow.runName": run_name or "OLS",
                                                    "target": "price_rub", "split_type": "temporal"})
        run_id = run.info.run_id
        config = result["config"]
        params = {"selection_method": config["selection_method"], "criterion": config["criterion"],
                  "train_end": config["train_end"], "validation_end": config["validation_end"],
                  "test_end": config["test_end"], "n_candidates": len(result["candidate_features"]),
                  "n_selected": len(result["selected_features"]), "target": "price_rub"}
        params.update({f"selection.{k}": v for k, v in config["selection_params"].items()})
        params.update({f"preprocessing.{k}": v for k, v in config["preprocessing"].items()})
        for key, value in params.items():
            client.log_param(run_id, key, str(value)[:500])
        for key, value in result["metrics"].items():
            if math.isfinite(float(value)):
                client.log_metric(run_id, key, float(value))
        datasets = {name: {"sha256": _sha256(path), "bytes": Path(path).stat().st_size}
                    for name, path in config["paths"].items()}
        for name, info in datasets.items():
            client.set_tag(run_id, f"dataset.{name}.sha256", info["sha256"])
        client.set_tag(run_id, "rank_deficient", str(result["diagnostics"]["rank_deficient"]))
        client.set_tag(run_id, "selection_converged", str(result["selection"]["converged"]))
        versions = {"python": platform.python_version(), "platform": platform.platform()}
        for package in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "mlflow-skinny"):
            versions[package] = importlib.metadata.version(package)
        # Directorio efímero: se elimina tanto si hay éxito como si hay error.
        with tempfile.TemporaryDirectory(prefix="moscow_ols_") as temporary:
            folder = Path(temporary)
            artifacts = {"config.json": config, "datasets.json": datasets, "environment.json": versions,
                         "features.json": {"candidates": result["candidate_features"],
                                           "candidate_groups": result["candidate_groups"],
                                           "selected": result["selected_features"],
                                           "selected_groups": result["selected_groups"],
                                           "dropped_constant_columns": result["dropped_constant_columns"]},
                         "selection.json": result["selection"], "diagnostics.json": result["diagnostics"],
                         "join_audit.json": result["join_audit"], "metrics.json": result["metrics"],
                         "run.json": {"run_id": run_id, "experiment_id": experiment_id}}
            for name, content in artifacts.items():
                (folder / name).write_text(json.dumps(_json_safe(content), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            (folder / "ols_summary.txt").write_text(result["model"].summary().as_text(), encoding="utf-8")
            result["model"].params.to_csv(folder / "coefficients.csv", header=["coefficient"])
            with (folder / "model.pkl").open("wb") as stream:
                pickle.dump({"model": result["model"], "preprocessor": result["preprocessor"],
                             "selected_features": result["selected_features"], "config": config}, stream)
            import matplotlib.pyplot as plt
            fig = diagnostic_figure(result["model"])
            try:
                fig.savefig(folder / "diagnostics.png", dpi=120)
            finally:
                plt.close(fig)
            # Código exacto utilizado, incluido CLI; permite reconstruir el experimento.
            import shutil
            source = folder / "source"
            source.mkdir()
            root = Path(__file__).resolve().parent
            files = list(root.glob("*.py")) + list((root / "scripts").glob("*.py")) + [root / "requirements.txt"]
            manifest = {}
            for path in files:
                relative = path.relative_to(root)
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                manifest[str(relative)] = _sha256(path)
            (folder / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            client.log_artifacts(run_id, str(folder))
        client.set_terminated(run_id, status="FINISHED")
        return run_id
    except Exception as error:
        if run_id is not None:
            try:
                client.set_tag(run_id, "tracking_error_type", type(error).__name__)
                client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass
        failure = RuntimeError(f"Tracking MLflow falló ({type(error).__name__}, run_id={run_id}); verificar configuración, conexión y permisos de artefactos. El resultado está en error.result.")
        failure.result = result
        raise failure from None
