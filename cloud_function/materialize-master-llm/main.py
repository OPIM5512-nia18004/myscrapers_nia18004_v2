import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage


BUCKET_NAME = os.getenv("GCS_BUCKET")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured-v2")

storage_client = storage.Client()

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

CSV_COLUMNS = [
    "post_id",
    "run_id",
    "scraped_at",
    "price",
    "year",
    "make",
    "model",
    "mileage",
    "color",
    "transmission",
    "cylinders",
    "fuel_type",
    "city",
    "state",
    "zip_code",
    "source_txt",
    "source_url",
    "llm_provider",
    "llm_model",
    "llm_ts",
]


def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass

    run_ids = []
    for prefix in getattr(it, "prefixes", []):
        tail = prefix.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            run_id = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(run_id) or RUN_ID_PLAIN_RE.match(run_id):
                run_ids.append(run_id)
    return sorted(run_ids)


def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    bucket_obj = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"
    for blob in bucket_obj.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        raw = blob.download_as_text().strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
            record.setdefault("run_id", run_id)
            yield record
        except Exception:
            continue


def _run_id_to_dt(run_id: str) -> datetime:
    if RUN_ID_ISO_RE.match(run_id):
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if RUN_ID_PLAIN_RE.match(run_id):
        return datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _open_gcs_text_writer(bucket: str, key: str):
    return storage_client.bucket(bucket).blob(key).open("w")


def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    rows_written = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in columns})
            rows_written += 1
    return rows_written


def materialize_http(request: Request):
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        body = request.get_json(silent=True) or {}
        if body.get("healthcheck") is True:
            return jsonify({"ok": True, "healthcheck": True, "function": "materialize-master-llm"}), 200

        run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not run_ids:
            return jsonify({"ok": False, "error": f"no runs found under {STRUCTURED_PREFIX}/"}), 200

        latest_by_post: Dict[str, Dict] = {}
        for run_id in run_ids:
            for record in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, run_id):
                post_id = record.get("post_id")
                if not post_id:
                    continue
                previous = latest_by_post.get(post_id)
                if previous is None or _run_id_to_dt(record.get("run_id", run_id)) > _run_id_to_dt(previous.get("run_id", "")):
                    latest_by_post[post_id] = record

        sorted_records = sorted(
            latest_by_post.values(),
            key=lambda record: (record.get("scraped_at") or "", record.get("post_id") or ""),
        )

        final_key = f"{STRUCTURED_PREFIX}/datasets/listings_master.csv"
        rows = _write_csv(sorted_records, final_key)
        return jsonify(
            {
                "ok": True,
                "runs_scanned": len(run_ids),
                "unique_listings": len(latest_by_post),
                "rows_written": rows,
                "output_csv": f"gs://{BUCKET_NAME}/{final_key}",
            }
        ), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
