"""CLI del pipeline; ejecutar desde cualquier directorio."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import run_pipeline, _json_safe
from selection import SELECTORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("secondary-path", "monthly-path", "metro-path", "test-end", "run-name"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--validation-end", default="2025-01-01")
    parser.add_argument("--include", nargs="+")
    parser.add_argument("--exclude", nargs="+")
    parser.add_argument("--selection-method", choices=SELECTORS, default="none")
    parser.add_argument("--criterion", choices=["aic", "bic", "pvalue"], default="aic")
    parser.add_argument("--selection-params", type=json.loads, default={}, help='Objeto JSON, p.ej. {"max_steps": 10}')
    parser.add_argument("--preprocessing", type=json.loads, default={}, help="Objeto JSON de opciones")
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--experiment-name", default="Moscow_OLS")
    result = run_pipeline(**vars(parser.parse_args()))
    print(json.dumps(_json_safe({"metrics": result["metrics"], "selected_groups": result["selected_groups"],
                      "run_id": result["run_id"], "rank_deficient": result["diagnostics"]["rank_deficient"]}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
