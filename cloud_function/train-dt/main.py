import io
import json
import logging
import os
import re
import traceback
import zipfile
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured-v2/datasets/listings_master.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "structured-v2/modeling")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
PERMUTATION_REPEATS = int(os.getenv("PERMUTATION_REPEATS", "10") or 10)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42") or 42)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
FUEL_ALIASES = {
    "gasoline": "gas",
    "petrol": "gas",
    "ev": "electric",
    "plug in hybrid": "plug-in hybrid",
    "phev": "plug-in hybrid",
    "flex fuel": "flex-fuel",
}


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    client.bucket(bucket).blob(key).upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _write_json_to_gcs(client: storage.Client, bucket: str, key: str, payload: dict):
    client.bucket(bucket).blob(key).upload_from_string(
        json.dumps(payload, indent=2, sort_keys=True),
        content_type="application/json",
    )


def _write_bytes_to_gcs(client: storage.Client, bucket: str, key: str, payload: bytes, content_type: str):
    client.bucket(bucket).blob(key).upload_from_string(payload, content_type=content_type)


def _read_text_from_gcs(client: storage.Client, bucket: str, key: str) -> str:
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return blob.download_as_text()


def _read_bytes_from_gcs(client: storage.Client, bucket: str, key: str) -> bytes:
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return blob.download_as_bytes()


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^\d.\-]+", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return cleaned.astype(object).where(cleaned.notna(), np.nan)


def _clean_state(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    normalized = cleaned.map(
        lambda value: value.upper() if pd.notna(value) and len(value) <= 3 else value.title() if pd.notna(value) else value
    )
    return normalized.astype(object).where(pd.notna(normalized), np.nan)


def _clean_zip(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    normalized = cleaned.map(lambda value: ZIP_RE.search(value).group(0) if pd.notna(value) and ZIP_RE.search(value) else value)
    return normalized.astype(object).where(pd.notna(normalized), np.nan)


def _clean_transmission(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    normalized = cleaned.map(
        lambda value: (
            "automatic" if pd.notna(value) and ("automatic" in str(value).lower() or str(value).lower() in {"auto", "a/t"})
            else "manual" if pd.notna(value) and ("manual" in str(value).lower() or str(value).lower() in {"man", "m/t", "stick"})
            else "cvt" if pd.notna(value) and "cvt" in str(value).lower()
            else str(value).lower() if pd.notna(value) else value
        )
    )
    return normalized.astype(object).where(pd.notna(normalized), np.nan)


def _clean_cylinders(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _clean_fuel_type(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    normalized = cleaned.map(
        lambda value: FUEL_ALIASES.get(str(value).lower(), str(value).lower()) if pd.notna(value) else value
    )
    return normalized.astype(object).where(pd.notna(normalized), np.nan)


def _safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float | None:
    mask = y_true.notna() & (y_true != 0)
    if not mask.any():
        return None
    truth = y_true[mask].to_numpy(dtype=float)
    pred = np.asarray(y_pred)[mask.to_numpy()]
    return float(np.mean(np.abs((truth - pred) / truth)) * 100.0)


def _metrics_payload(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mask = y_true.notna()
    if not mask.any():
        return {"mae": None, "mape": None, "rmse": None, "bias": None}

    truth = y_true[mask].to_numpy(dtype=float)
    pred = np.asarray(y_pred)[mask.to_numpy()]
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "mape": _safe_mape(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "bias": float(np.mean(pred - truth)),
    }


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in [
        "make", "model", "color", "city", "state", "zip_code", "transmission", "fuel_type",
        "price", "year", "mileage", "cylinders", "post_id", "scraped_at"
    ]:
        if column not in prepared.columns:
            prepared[column] = np.nan

    prepared["scraped_at_dt_utc"] = pd.to_datetime(prepared["scraped_at"], errors="coerce", utc=True)
    try:
        prepared["scraped_at_local"] = prepared["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        prepared["scraped_at_local"] = prepared["scraped_at_dt_utc"]

    prepared["date_local"] = prepared["scraped_at_local"].dt.date
    prepared["listing_hour"] = prepared["scraped_at_local"].dt.hour
    prepared["day_of_week"] = prepared["scraped_at_local"].dt.dayofweek
    prepared["month"] = prepared["scraped_at_local"].dt.month
    prepared["price_num"] = _clean_numeric(prepared["price"])
    prepared["year_num"] = _clean_numeric(prepared["year"])
    prepared["mileage_num"] = _clean_numeric(prepared["mileage"])
    prepared["cylinders_num"] = _clean_cylinders(prepared["cylinders"])

    prepared["make"] = _clean_text(prepared["make"])
    prepared["model"] = _clean_text(prepared["model"])
    prepared["color"] = _clean_text(prepared["color"])
    prepared["transmission"] = _clean_transmission(prepared["transmission"])
    prepared["fuel_type"] = _clean_fuel_type(prepared["fuel_type"])
    prepared["city"] = _clean_text(prepared["city"])
    prepared["state"] = _clean_state(prepared["state"])
    prepared["zip_code"] = _clean_zip(prepared["zip_code"])

    local_year = prepared["scraped_at_local"].dt.year.astype("float")
    prepared["vehicle_age"] = local_year - prepared["year_num"]
    prepared.loc[(prepared["vehicle_age"] < 0) | (prepared["vehicle_age"] > 80), "vehicle_age"] = np.nan
    return prepared


def _build_pipeline() -> tuple[Pipeline, dict, list[str], list[str]]:
    cat_cols = ["make", "model", "color", "transmission", "fuel_type", "city", "state", "zip_code"]
    num_cols = ["year_num", "mileage_num", "cylinders_num", "vehicle_age", "listing_hour", "day_of_week", "month"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 16],
        "model__min_samples_leaf": [1, 5, 10],
    }
    return pipeline, param_grid, cat_cols, num_cols


def _append_metrics_history(client: storage.Client, bucket: str, key: str, row: dict):
    blob = client.bucket(bucket).blob(key)
    existing = pd.read_csv(io.BytesIO(blob.download_as_bytes())) if blob.exists() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = updated.sort_values(by=["run_ts"], kind="stable")
    _write_csv_to_gcs(client, bucket, key, updated)


def _choose_pdp_features(importance_df: pd.DataFrame, feature_frame: pd.DataFrame) -> list[str]:
    chosen: list[str] = []
    for feature in importance_df["feature"].tolist():
        if feature not in feature_frame.columns:
            continue
        series = feature_frame[feature]
        if series.isna().all():
            continue
        if pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= 20:
            chosen.append(feature)
        if len(chosen) == 3:
            return chosen

    fallback = [column for column in feature_frame.columns if pd.api.types.is_numeric_dtype(feature_frame[column]) and not feature_frame[column].isna().all()]
    for feature in fallback:
        if feature not in chosen:
            chosen.append(feature)
        if len(chosen) == 3:
            break
    return chosen


def _bundle_latest_artifacts(client: storage.Client) -> tuple[bytes, str]:
    manifest_key = f"{OUTPUT_PREFIX}/latest_manifest.json"
    manifest = json.loads(_read_text_from_gcs(client, GCS_BUCKET, manifest_key))
    gcs_paths = manifest.get("gcs_paths", {})

    files_to_bundle: list[tuple[str, str]] = [
        (manifest_key, "latest_manifest.json"),
    ]
    singletons = {
        "predictions": "predictions.csv",
        "permutation_importance": "permutation_importance.csv",
        "metrics_json": "metrics.json",
        "best_params_json": "best_params.json",
        "metrics_history": "metrics_history.csv",
    }
    for key, archive_name in singletons.items():
        uri = gcs_paths.get(key)
        if uri and uri.startswith(f"gs://{GCS_BUCKET}/"):
            files_to_bundle.append((uri.replace(f"gs://{GCS_BUCKET}/", ""), archive_name))

    for uri in gcs_paths.get("pdp", []):
        if uri and uri.startswith(f"gs://{GCS_BUCKET}/"):
            key = uri.replace(f"gs://{GCS_BUCKET}/", "")
            files_to_bundle.append((key, os.path.basename(key)))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for gcs_key, archive_name in files_to_bundle:
            archive.writestr(archive_name, _read_bytes_from_gcs(client, GCS_BUCKET, gcs_key))

    filename = f"training_artifacts_{manifest.get('run_ts', 'latest')}.zip"
    return buffer.getvalue(), filename


def run_once(dry_run: bool = False) -> dict:
    client = storage.Client(project=PROJECT_ID)
    df = _prepare_dataframe(_read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY))

    valid_dates = sorted(date for date in df["date_local"].dropna().unique())
    if len(valid_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates", "dates": [str(date) for date in valid_dates]}

    today_local = valid_dates[-1]
    train_df = df[(df["date_local"] < today_local) & df["price_num"].notna()].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}
    if holdout_df.empty:
        return {"status": "noop", "reason": "no today rows found", "today_local": str(today_local)}

    pipeline, param_grid, cat_cols, num_cols = _build_pipeline()
    feature_cols = cat_cols + num_cols
    train_df = train_df.sort_values(by=["scraped_at_dt_utc", "post_id"], kind="stable")
    holdout_df = holdout_df.sort_values(by=["scraped_at_dt_utc", "post_id"], kind="stable")

    X_train = train_df[feature_cols]
    y_train = train_df["price_num"]
    X_holdout = holdout_df[feature_cols]

    train_unique_dates = train_df["date_local"].nunique(dropna=True)
    can_tune = len(train_df) >= 60 and train_unique_dates >= 3

    if can_tune:
        n_splits = min(3, train_unique_dates - 1, len(train_df) - 1)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=max(2, n_splits)),
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        cv_mae = float(-search.best_score_)
        tuning_strategy = "grid_search"
    else:
        best_estimator = pipeline.set_params(
            model__n_estimators=300,
            model__max_depth=16,
            model__min_samples_leaf=3,
        )
        best_estimator.fit(X_train, y_train)
        best_params = {
            "model__n_estimators": 300,
            "model__max_depth": 16,
            "model__min_samples_leaf": 3,
        }
        cv_mae = None
        tuning_strategy = "fallback_defaults"

    predictions = best_estimator.predict(X_holdout)
    metrics = _metrics_payload(holdout_df["price_num"], predictions)

    predictions_df = holdout_df[["post_id", "scraped_at", "make", "model", "year", "mileage", "color", "city", "state", "zip_code", "price"]].copy()
    predictions_df["actual_price"] = holdout_df["price_num"]
    predictions_df["pred_price"] = np.round(predictions, 2)
    predictions_df["prediction_error"] = predictions_df["pred_price"] - predictions_df["actual_price"]

    eval_df = holdout_df[holdout_df["price_num"].notna()].copy()
    if len(eval_df) >= 10:
        X_eval = eval_df[feature_cols]
        y_eval = eval_df["price_num"]
    else:
        X_eval = X_train.head(min(len(X_train), 250))
        y_eval = y_train.head(len(X_eval))

    permutation = permutation_importance(
        best_estimator,
        X_eval,
        y_eval,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="neg_mean_absolute_error",
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False, kind="stable")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = f"{OUTPUT_PREFIX}/runs/{run_ts}"
    predictions_key = f"{run_prefix}/predictions.csv"
    importance_key = f"{run_prefix}/permutation_importance.csv"
    metrics_key = f"{run_prefix}/metrics.json"
    best_params_key = f"{run_prefix}/best_params.json"
    history_key = f"{OUTPUT_PREFIX}/metrics_history.csv"

    pdp_keys = []
    pdp_features = _choose_pdp_features(importance_df, train_df[feature_cols])
    pdp_frame = X_train.head(min(len(X_train), 500))
    for feature in pdp_features:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(best_estimator, pdp_frame, [feature], ax=ax)
            ax.set_title(f"Partial Dependence: {feature}")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            pdp_key = f"{run_prefix}/pdp_{feature}.png"
            if not dry_run:
                _write_bytes_to_gcs(client, GCS_BUCKET, pdp_key, buf.getvalue(), "image/png")
            pdp_keys.append(pdp_key)
        except Exception as exc:
            logging.warning("Unable to generate PDP for %s: %s", feature, exc)

    manifest = {
        "status": "ok",
        "run_ts": run_ts,
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_today_prices": int(holdout_df["price_num"].notna().sum()),
        "tuning_strategy": tuning_strategy,
        "cv_mae": cv_mae,
        "best_params": best_params,
        "metrics": metrics,
        "pdp_features": pdp_features,
        "gcs_paths": {
            "predictions": f"gs://{GCS_BUCKET}/{predictions_key}",
            "permutation_importance": f"gs://{GCS_BUCKET}/{importance_key}",
            "metrics_json": f"gs://{GCS_BUCKET}/{metrics_key}",
            "best_params_json": f"gs://{GCS_BUCKET}/{best_params_key}",
            "metrics_history": f"gs://{GCS_BUCKET}/{history_key}",
            "pdp": [f"gs://{GCS_BUCKET}/{key}" for key in pdp_keys],
        },
    }

    history_row = {
        "run_ts": run_ts,
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_today_prices": int(holdout_df["price_num"].notna().sum()),
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "rmse": metrics["rmse"],
        "bias": metrics["bias"],
        "cv_mae": cv_mae,
        "tuning_strategy": tuning_strategy,
        "best_params_json": json.dumps(best_params, sort_keys=True),
    }

    if not dry_run:
        _write_csv_to_gcs(client, GCS_BUCKET, predictions_key, predictions_df)
        _write_csv_to_gcs(client, GCS_BUCKET, importance_key, importance_df)
        _write_json_to_gcs(client, GCS_BUCKET, metrics_key, {**metrics, **history_row})
        _write_json_to_gcs(client, GCS_BUCKET, best_params_key, best_params)
        _append_metrics_history(client, GCS_BUCKET, history_key, history_row)
        _write_json_to_gcs(client, GCS_BUCKET, f"{run_prefix}/manifest.json", manifest)
        _write_json_to_gcs(client, GCS_BUCKET, f"{OUTPUT_PREFIX}/latest_manifest.json", manifest)

    return manifest | {"dry_run": dry_run}


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        if body.get("healthcheck") is True:
            result = {"status": "ok", "healthcheck": True, "function": "train-dt"}
            return json.dumps(result), 200, {"Content-Type": "application/json"}

        if body.get("download_latest_bundle") is True:
            client = storage.Client(project=PROJECT_ID)
            bundle_bytes, filename = _bundle_latest_artifacts(client)
            return bundle_bytes, 200, {
                "Content-Type": "application/zip",
                "Content-Disposition": f'attachment; filename="{filename}"',
            }

        result = run_once(dry_run=bool(body.get("dry_run", False)))
        code = 200 if result.get("status") == "ok" else 204
        return json.dumps(result), code, {"Content-Type": "application/json"}
    except Exception as exc:
        logging.error("Error: %s", exc)
        logging.error("Trace:\n%s", traceback.format_exc())
        return json.dumps({"status": "error", "error": str(exc)}), 500, {"Content-Type": "application/json"}
