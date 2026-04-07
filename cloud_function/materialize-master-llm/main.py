# main.py
# Build ONE combined CSV from regex JSONL + LLM JSONL files.
# Reads:
#   gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl/*.jsonl
#   gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl_llm/*.jsonl
# Writes:
#   gs://<bucket>/<STRUCTURED_PREFIX>/datasets/listings_master.csv

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME        = os.getenv("GCS_BUCKET")                         # REQUIRED
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured-v2") # e.g., "structured-v2"
OUTPUT_FILENAME    = os.getenv("OUTPUT_FILENAME", "listings_master.csv")

storage_client = storage.Client()

# Accept BOTH runIDs:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

# Stable CSV schema:
# - canonical fields (`price`, `year`, `make`, `model`, `mileage`) choose LLM first, then regex fallback
# - keep both extraction sources for auditability/debugging
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage", "color",
    "price_regex", "year_regex", "make_regex", "model_regex", "mileage_regex", "transmission_regex", "cylinders_regex",
    "price_llm", "year_llm", "make_llm", "model_llm", "mileage_llm", "color_llm",
    "llm_provider", "llm_model", "llm_ts",
    "source_txt"
]

def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:  # populate it.prefixes
        pass
    run_ids = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]           # e.g. run_id=20251026170002
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)
    return sorted(run_ids)

def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str, subdir: str):
    """Yield dict records from .jsonl under .../run_id=<run_id>/<subdir>/."""
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/{subdir}/"
    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        data = blob.download_as_text()
        line = data.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            # ensure required keys exist
            rec.setdefault("run_id", run_id)
            yield rec
        except Exception:
            continue

def _run_id_to_dt(rid: str) -> datetime:
    if RUN_ID_ISO_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if RUN_ID_PLAIN_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    # fallback: now
    return datetime.now(timezone.utc)

def _open_gcs_text_writer(bucket: str, key: str):
    """Open a text-mode writer to GCS; close() will finalize the upload."""
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    # Text mode avoids the flush/finalize pitfall of binary+TextIOWrapper
    return blob.open("w")  # newline handled by csv module


def _pick_llm_then_regex(llm_rec: Dict, regex_rec: Dict, key: str):
    v = llm_rec.get(key) if llm_rec else None
    if v is not None and v != "":
        return v
    return (regex_rec or {}).get(key)


def _merged_record(regex_rec: Dict, llm_rec: Dict, run_id: str) -> Dict:
    regex_rec = regex_rec or {}
    llm_rec = llm_rec or {}

    return {
        "post_id": llm_rec.get("post_id") or regex_rec.get("post_id"),
        "run_id": llm_rec.get("run_id") or regex_rec.get("run_id") or run_id,
        "scraped_at": llm_rec.get("scraped_at") or regex_rec.get("scraped_at"),

        # Canonical values used downstream
        "price": _pick_llm_then_regex(llm_rec, regex_rec, "price"),
        "year": _pick_llm_then_regex(llm_rec, regex_rec, "year"),
        "make": _pick_llm_then_regex(llm_rec, regex_rec, "make"),
        "model": _pick_llm_then_regex(llm_rec, regex_rec, "model"),
        "mileage": _pick_llm_then_regex(llm_rec, regex_rec, "mileage"),
        "color": llm_rec.get("color"),

        # Source-specific values
        "price_regex": regex_rec.get("price"),
        "year_regex": regex_rec.get("year"),
        "make_regex": regex_rec.get("make"),
        "model_regex": regex_rec.get("model"),
        "mileage_regex": regex_rec.get("mileage"),
        "transmission_regex": regex_rec.get("transmission"),
        "cylinders_regex": regex_rec.get("cylinders"),

        "price_llm": llm_rec.get("price"),
        "year_llm": llm_rec.get("year"),
        "make_llm": llm_rec.get("make"),
        "model_llm": llm_rec.get("model"),
        "mileage_llm": llm_rec.get("mileage"),
        "color_llm": llm_rec.get("color"),
        "llm_provider": llm_rec.get("llm_provider"),
        "llm_model": llm_rec.get("llm_model"),
        "llm_ts": llm_rec.get("llm_ts"),

        "source_txt": llm_rec.get("source_txt") or regex_rec.get("source_txt"),
    }


def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            row = {c: rec.get(c, None) for c in columns}
            w.writerow(row)
            n += 1
    return n  # close() finalizes the upload

def materialize_http(request: Request):
    """
    HTTP POST (no body needed).
    Crawls ALL structured-v2 run folders, combines regex + LLM per listing,
    de-dupes by post_id (keep newest run), and writes one combined CSV.
    Returns JSON with counts and output path.
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            body = {}

        if body.get("healthcheck") is True:
            return jsonify({
                "ok": True,
                "healthcheck": True,
                "structured_prefix": STRUCTURED_PREFIX,
                "output_filename": OUTPUT_FILENAME,
            }), 200

        run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not run_ids:
            return jsonify({"ok": False, "error": f"no runs found under {STRUCTURED_PREFIX}/"}), 200

        latest_by_post: Dict[str, Dict] = {}
        for rid in run_ids:
            regex_by_post: Dict[str, Dict] = {}
            llm_by_post: Dict[str, Dict] = {}

            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid, "jsonl"):
                pid = rec.get("post_id")
                if pid:
                    regex_by_post[pid] = rec

            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid, "jsonl_llm"):
                pid = rec.get("post_id")
                if not pid:
                    continue
                llm_by_post[pid] = rec

            all_post_ids = set(regex_by_post.keys()) | set(llm_by_post.keys())
            for pid in all_post_ids:
                rec = _merged_record(regex_by_post.get(pid), llm_by_post.get(pid), rid)
                prev = latest_by_post.get(pid)
                if (prev is None) or (_run_id_to_dt(rec.get("run_id", rid)) > _run_id_to_dt(prev.get("run_id", ""))):
                    latest_by_post[pid] = rec

        base = f"{STRUCTURED_PREFIX}/datasets"
        final_key = f"{base}/{OUTPUT_FILENAME}"
        rows = _write_csv(latest_by_post.values(), final_key)

        return jsonify({
            "ok": True,
            "runs_scanned": len(run_ids),
            "unique_listings": len(latest_by_post),
            "rows_written": rows,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200
    except Exception as e:
        # Return a JSON error so you don't just see a plain 500
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
