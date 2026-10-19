"""Exporta el modelo de riesgo crediticio a formato ONNX para inferencia edge/web.

Genera:
1. `credit_risk_model.onnx`: Grafo optimizado compatible con ONNX Runtime WebAssembly (WASM).
2. `credit_risk_meta.json`: Metadatos de preprocesamiento, orden de columnas, medianas y vocabularios.

Uso:

    # Entrenar y exportar con los datos canónicos locales
    python scripts/export_onnx.py --data data/lc_2007_2015_subset.csv.gz --output dist/onnx

    # Exportar un modelo CreditRiskModel ya serializado en joblib
    python scripts/export_onnx.py --model ruta/credit_model.joblib --output dist/onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

logger = logging.getLogger("export_onnx")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportar Credit Risk Model a ONNX")
    parser.add_argument("--model", type=Path, default=None, help="Ruta a credit_model.joblib (opcional)")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "lc_2007_2015_subset.csv.gz",
        help="Ruta al dataset comprimido de Lending Club",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "onnx",
        help="Directorio de destino para los artefactos ONNX",
    )
    parser.add_argument("--rows", type=int, default=120000, help="Número de filas de entrenamiento si no hay joblib")
    parser.add_argument("--opset", type=int, default=14, help="Opset objetivo de ONNX")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from credit_risk.features.engineering import Preprocessor, build_features

    if args.model and args.model.exists():
        import joblib

        logger.info("Cargando modelo existente desde %s...", args.model)
        credit_model = joblib.load(args.model)
        pre = credit_model.preprocessor
        estimator = credit_model.estimator
        algorithm = credit_model.algorithm
        feature_columns = list(pre.columns_)
    else:
        logger.info("Entrenando modelo de calibración sobre %d préstamos de %s...", args.rows, args.data)
        from credit_risk.data.lending_club import label_from_status, parse_raw

        df_raw = pd.read_csv(args.data, nrows=args.rows)
        parsed = parse_raw(df_raw)
        parsed["target"] = label_from_status(df_raw["loan_status"])
        labeled = parsed.dropna(subset=["target"]).copy()
        labeled["target"] = labeled["target"].astype(int)

        train_n = int(len(labeled) * 0.75)
        train_df = labeled.iloc[:train_n]

        pre = Preprocessor(min_category_share=0.002, min_missing_share=0.001)
        x_train = pre.fit_transform(build_features(train_df))
        feature_columns = list(x_train.columns)

        import lightgbm as lgb

        algorithm = "lightgbm"
        estimator = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=16,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=150,
            reg_lambda=5.0,
            random_state=42,
            verbose=-1,
        )
        estimator.fit(x_train.to_numpy(), train_df["target"].to_numpy())
        logger.info("Entrenamiento finalizado. Matriz de features: %d columnas", len(feature_columns))

    # Conversión a ONNX
    logger.info("Convirtiendo estimador (%s) a formato ONNX con opset %d...", algorithm, args.opset)
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, len(feature_columns)]))]

    if algorithm == "lightgbm":
        onnx_model = onnxmltools.convert_lightgbm(
            estimator,
            initial_types=initial_type,
            zipmap=False,
            target_opset=args.opset,
        )
    elif algorithm == "xgboost":
        onnx_model = onnxmltools.convert_xgboost(
            estimator,
            initial_types=initial_type,
            target_opset=args.opset,
        )
    else:
        from skl2onnx import convert_sklearn

        onnx_model = convert_sklearn(
            estimator,
            initial_types=initial_type,
            options={"zipmap": False},
            target_opset=args.opset,
        )

    onnx_file = args.output / "credit_risk_model.onnx"
    with open(onnx_file, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("Modelo ONNX guardado exitosamente en %s (%.1f KB)", onnx_file, onnx_file.stat().st_size / 1024)

    # Serializar metadatos para la inferencia cliente
    meta = {
        "model_name": f"LendingClub-{algorithm.capitalize()}-Champion",
        "version": "1.0.0",
        "threshold": 0.18,
        "feature_columns": feature_columns,
        "numeric_columns": list(pre.numeric_columns),
        "medians": {k: float(v) for k, v in pre.medians_.items()},
        "missing_indicators": list(getattr(pre, "missing_indicators_", [])),
        "vocab": {k: list(v) for k, v in pre.vocab_.items()},
        "examples": {
            "conservative": {
                "name": "Perfil conservador (Bajo riesgo)",
                "values": {
                    "loan_amnt": 10000, "term": 36, "int_rate": 7.5, "grade": "A", "sub_grade": "A3",
                    "purpose": "credit_card", "annual_inc": 95000, "emp": 8, "home": "MORTGAGE",
                    "verif": "Verified", "dti": 9.0, "state": "TX", "fico": 760, "revol_bal": 6000,
                    "revol_util": 22.0, "open_acc": 9, "total_acc": 28, "history": 260
                }
            },
            "typical": {
                "name": "Perfil promedio (Riesgo moderado)",
                "values": {
                    "loan_amnt": 12000, "term": 36, "int_rate": 13.5, "grade": "C", "sub_grade": "C2",
                    "purpose": "debt_consolidation", "annual_inc": 65000, "emp": 5, "home": "RENT",
                    "verif": "Source Verified", "dti": 18.5, "state": "CA", "fico": 702, "revol_bal": 14500,
                    "revol_util": 55.0, "open_acc": 10, "total_acc": 24, "history": 180
                }
            },
            "risky": {
                "name": "Perfil riesgoso (Alto riesgo)",
                "values": {
                    "loan_amnt": 30000, "term": 60, "int_rate": 24.0, "grade": "F", "sub_grade": "F2",
                    "purpose": "small_business", "annual_inc": 38000, "emp": 1, "home": "RENT",
                    "verif": "Not Verified", "dti": 34.0, "state": "NV", "fico": 662, "revol_bal": 21000,
                    "revol_util": 92.0, "open_acc": 15, "total_acc": 18, "history": 60
                }
            }
        }
    }

    meta_file = args.output / "credit_risk_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Metadatos serializados en %s", meta_file)

    # Smoke test con onnxruntime
    logger.info("Ejecutando smoke test con onnxruntime...")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_file))
    dummy_input = np.zeros((1, len(feature_columns)), dtype=np.float32)
    res = sess.run(None, {"float_input": dummy_input})
    logger.info("Smoke test OK: salida de probabilidades shape %s", str(res[1].shape))


if __name__ == "__main__":
    main()
