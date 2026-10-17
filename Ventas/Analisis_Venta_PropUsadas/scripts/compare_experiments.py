"""Consulta ejecuciones remotas; ordena por validación y nunca por test."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import get_tracking_client


def compare_experiments(experiment_name="Moscow_OLS", filter_string="", max_results=100,
                        order_by="metrics.validation_rmse ASC"):
    if max_results < 1:
        raise ValueError("max_results debe ser positivo")
    client = get_tracking_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento inexistente: {experiment_name}")
    rows, token = [], None
    while len(rows) < max_results:
        runs = client.search_runs([experiment.experiment_id], filter_string=filter_string,
                                  max_results=min(1000, max_results - len(rows)),
                                  order_by=[order_by], page_token=token)
        for run in runs:
            rows.append({"run_id": run.info.run_id, "status": run.info.status,
                         "run_name": run.data.tags.get("mlflow.runName"),
                         **{f"param.{k}": v for k, v in run.data.params.items()},
                         **{f"metric.{k}": v for k, v in run.data.metrics.items()},
                         **{f"tag.{k}": v for k, v in run.data.tags.items() if k.startswith("dataset.")}})
        token = getattr(runs, "token", None)
        if not token:
            break
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="Moscow_OLS")
    parser.add_argument("--filter-string", default="attributes.status = 'FINISHED'")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--output", help="CSV de comparación, sólo si se solicita explícitamente")
    args = parser.parse_args()
    frame = compare_experiments(args.experiment_name, args.filter_string, args.max_results)
    keys = ["run_id", "param.selection_method", "param.criterion", "param.n_selected",
            "metric.train_r2", "metric.validation_r2", "metric.validation_rmse", "metric.validation_mae"]
    print(frame.reindex(columns=keys).to_string(index=False))
    if args.output:
        frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
