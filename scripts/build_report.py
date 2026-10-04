"""Reporte estadístico completo y reproducible del modelo de riesgo crediticio.

Corre el mismo flujo que el job de entrenamiento (benchmark, selección estadística en
validación, tuning, reporte en test) más el análisis que justifica la elección, y
escribe:

- `reports/results.json`: todas las cifras (fuente única de verdad).
- `docs/figures/*.png`: gráficas.
- La sección de resultados del README, entre los marcadores RESULTADOS:INICIO/FIN.

Uso (CSV de Lending Club completo, o un subconjunto con las mismas columnas):

    python scripts/build_report.py --data data/accepted_2007_to_2018Q4.csv
    python scripts/build_report.py --data data/lc_2007_2015_subset.csv.gz --optuna-trials 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data.lending_club import parse_raw  # noqa: E402
from credit_risk.data.quality import clean  # noqa: E402
from credit_risk.reporting import figures, readme  # noqa: E402
from credit_risk.reporting.pipeline import build_results, json_safe  # noqa: E402

COLUMNS = [
    "addr_state", "annual_inc", "application_type", "delinq_2yrs", "dti", "earliest_cr_line", "emp_length",
    "fico_range_high", "fico_range_low", "grade", "home_ownership", "id", "initial_list_status", "inq_last_6mths",
    "installment", "int_rate", "issue_d", "loan_amnt", "loan_status", "mort_acc", "open_acc", "pub_rec",
    "pub_rec_bankruptcies", "purpose", "revol_bal", "revol_util", "sub_grade", "term", "total_acc",
    "verification_status",
]  # fmt: skip


def load_raw(path: Path) -> pd.DataFrame:
    """Lee solo las columnas que usa el parser (el CSV completo pesa 1.7 GB)."""
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in COLUMNS if c in header]
    return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="CSV (o .csv.gz) con el formato de Lending Club")
    source.add_argument(
        "--from-results",
        type=Path,
        help="results.json ya calculado (p. ej. descargado de MLflow: report/results.json): solo re-renderiza",
    )
    parser.add_argument("--optuna-trials", type=int, default=None, help="por defecto, el de config/platform.yaml")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--figures", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--no-readme", action="store_true", help="no reescribir la sección del README")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.from_results:
        results = json.loads(args.from_results.read_text(encoding="utf-8"))
    else:
        canonical = clean(parse_raw(load_raw(args.data)))
        results = build_results(canonical, optuna_trials=args.optuna_trials)
        results["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        results["data_file"] = f"{args.data.name} ({len(canonical):,} préstamos)"
    if args.from_results is None or args.from_results.resolve() != args.results.resolve():
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(json_safe(results), ensure_ascii=False, indent=1), encoding="utf-8")
    files = figures.render_all(json_safe(results), args.figures)
    logging.info("Resultados en %s; %d figuras en %s", args.results, len(files), args.figures)

    if not args.no_readme:
        rel = args.figures.resolve().relative_to(args.readme.resolve().parent).as_posix()
        section = readme.render(json_safe(results), files, fig_dir=rel)
        text = args.readme.read_text(encoding="utf-8")
        args.readme.write_text(readme.update_readme(text, section), encoding="utf-8")
        logging.info("Sección de resultados actualizada en %s", args.readme)


if __name__ == "__main__":
    main()
