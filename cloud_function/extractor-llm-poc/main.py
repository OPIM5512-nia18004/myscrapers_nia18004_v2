# main.py
# Purpose: PoC LLM extractor that reads your existing per-listing JSONL records,
# fetches the original TXT, asks an LLM (Vertex AI) to extract fields, and writes
# a sibling "<post_id>_llm.jsonl" to the `structured-v2/.../jsonl_llm/` sub-directory.
#
# FINAL FIXES INCLUDED:
# 1. Schema updated to use "type": "string" + "nullable": True.
# 2. system_instruction removed from GenerationConfig and merged into prompt.
# 3. LLM_MODEL set to 'gemini-2.5-flash' (Fixes 404/NotFound error).
# 4. "additionalProperties": False removed from schema (Fixes internal ParseError).
# 5. Non-breaking spaces (U+00A0) replaced with standard spaces (U+0020). <--- FIX FOR THIS ERROR

import csv
import io
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.api_core.exceptions import Aborted, DeadlineExceeded, InternalServerError, ResourceExhausted
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel


PROJECT_ID = os.getenv("PROJECT_ID", "")
REGION = os.getenv("REGION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET", "")
SCRAPES_PREFIX = os.getenv("SCRAPES_PREFIX", "scrapes")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "vertex").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
OVERWRITE_DEFAULT = os.getenv("OVERWRITE", "false").lower() == "true"
MAX_FILES_DEFAULT = int(os.getenv("MAX_FILES", "0") or 0)

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    deadline=120.0,
)

storage_client = storage.Client()
_CACHED_MODEL_OBJ = None

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
FUEL_ALIASES = {
    "gas": "gas",
    "gasoline": "gas",
    "petrol": "gas",
    "diesel": "diesel",
    "hybrid": "hybrid",
    "plug-in hybrid": "plug-in hybrid",
    "plug in hybrid": "plug-in hybrid",
    "phev": "plug-in hybrid",
    "electric": "electric",
    "ev": "electric",
    "flex fuel": "flex-fuel",
    "flex-fuel": "flex-fuel",
    "e85": "flex-fuel",
    "cng": "cng",
}


def _if_llm_retryable(exception: Exception) -> bool:
    return isinstance(exception, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))


def _get_vertex_model() -> GenerativeModel:
    global _CACHED_MODEL_OBJ
    if _CACHED_MODEL_OBJ is None:
        if not PROJECT_ID:
            raise RuntimeError("PROJECT_ID environment variable is missing.")
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL_OBJ = GenerativeModel(LLM_MODEL)
        logging.info("Initialized Vertex AI model %s in %s", LLM_MODEL, REGION)
    return _CACHED_MODEL_OBJ


def _list_run_ids(bucket: str, scrapes_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{scrapes_prefix}/", delimiter="/")
    for _ in it:
        pass

    run_ids: list[str] = []
    for pref in getattr(it, "prefixes", []):
        tail = pref.rstrip("/").split("/")[-1]
        candidate = tail.split("run_id=", 1)[1] if tail.startswith("run_id=") else tail
        if RUN_ID_ISO_RE.match(candidate) or RUN_ID_PLAIN_RE.match(candidate):
            run_ids.append(candidate)
    return sorted(run_ids)


def _txt_objects_for_run(run_id: str) -> list[str]:
    bucket = storage_client.bucket(BUCKET_NAME)
    candidates = [
        f"{SCRAPES_PREFIX}/run_id={run_id}/txt/",
        f"{SCRAPES_PREFIX}/run_id={run_id}/",
        f"{SCRAPES_PREFIX}/{run_id}/txt/",
        f"{SCRAPES_PREFIX}/{run_id}/",
    ]
    for prefix in candidates:
        names = [blob.name for blob in bucket.list_blobs(prefix=prefix) if blob.name.endswith(".txt")]
        if names:
            return names
    return []


def _index_blob_for_run(run_id: str) -> str | None:
    bucket = storage_client.bucket(BUCKET_NAME)
    candidates = [
        f"{SCRAPES_PREFIX}/run_id={run_id}/index.csv",
        f"{SCRAPES_PREFIX}/{run_id}/index.csv",
    ]
    for name in candidates:
        if bucket.blob(name).exists():
            return name
    return None


def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).download_as_text(retry=READ_RETRY, timeout=120)


def _load_source_url_map(run_id: str) -> dict[str, str]:
    blob_name = _index_blob_for_run(run_id)
    if not blob_name:
        return {}

    try:
        raw = _download_text(blob_name)
        reader = csv.DictReader(io.StringIO(raw))
        return {
            str(row.get("post_id", "")).strip(): str(row.get("url", "")).strip()
            for row in reader
            if row.get("post_id") and row.get("url")
        }
    except Exception:
        logging.warning("Unable to load source URL map for run %s from %s", run_id, blob_name)
        return {}


def _llm_output_key(run_id: str, post_id: str) -> str:
    return f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl_llm/{post_id}_llm.jsonl"


def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    bucket.blob(blob_name).upload_from_string(line, content_type="application/x-ndjson")


def _blob_exists(blob_name: str) -> bool:
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).exists()


def _normalize_run_id_iso(run_id: str) -> str:
    try:
        if RUN_ID_ISO_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        elif RUN_ID_PLAIN_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            raise ValueError("unsupported run_id")
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value):
    try:
        if value in (None, ""):
            return None
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


def _norm_text(value):
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _norm_state(value):
    normalized = _norm_text(value)
    if not normalized:
        return None
    return normalized.upper() if len(normalized) <= 3 else normalized.title()


def _norm_zip(value):
    normalized = _norm_text(value)
    if not normalized:
        return None
    match = ZIP_RE.search(normalized)
    return match.group(0) if match else normalized


def _norm_transmission(value):
    normalized = _norm_text(value)
    if not normalized:
        return None
    lowered = normalized.lower()
    if "automatic" in lowered or lowered in {"auto", "a/t"}:
        return "automatic"
    if "manual" in lowered or lowered in {"man", "m/t", "stick"}:
        return "manual"
    if "cvt" in lowered:
        return "cvt"
    return lowered


def _norm_cylinders(value):
    normalized = _norm_text(value)
    if not normalized:
        return None
    match = re.search(r"(\d+)", normalized)
    if not match:
        return None
    return _safe_int(match.group(1))


def _norm_fuel_type(value):
    normalized = _norm_text(value)
    if not normalized:
        return None
    lowered = normalized.lower()
    return FUEL_ALIASES.get(lowered, lowered)


def _vertex_extract_fields(raw_text: str, source_url: str | None = None) -> dict:
    model = _get_vertex_model()
    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "integer", "nullable": True},
            "year": {"type": "integer", "nullable": True},
            "make": {"type": "string", "nullable": True},
            "model": {"type": "string", "nullable": True},
            "mileage": {"type": "integer", "nullable": True},
            "color": {"type": "string", "nullable": True},
            "transmission": {"type": "string", "nullable": True},
            "cylinders": {"type": "integer", "nullable": True},
            "fuel_type": {"type": "string", "nullable": True},
            "city": {"type": "string", "nullable": True},
            "state": {"type": "string", "nullable": True},
            "zip_code": {"type": "string", "nullable": True},
        },
        "required": [
            "price", "year", "make", "model", "mileage", "color",
            "transmission", "cylinders", "fuel_type", "city", "state", "zip_code"
        ],
    }

    prompt = (
        "Extract ONLY the requested vehicle and location fields from the listing text. "
        "Return strict JSON that conforms to the schema. If a value is missing, return null. "
        "Use integers for price, year, and mileage. For zip_code keep it as a string so leading zeros survive. "
        "For cylinders use just the cylinder count as an integer when explicit. "
        "For transmission prefer values like automatic, manual, or cvt. "
        "For fuel_type prefer values like gas, diesel, hybrid, plug-in hybrid, electric, or flex-fuel. "
        "For state, prefer a 2-letter abbreviation when explicit. Do not invent values.\n\n"
        f"LISTING URL: {source_url or 'unknown'}\n\n"
        f"LISTING TEXT:\n{raw_text[:20000]}"
    )

    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        candidate_count=1,
        response_mime_type="application/json",
        response_schema=schema,
    )

    response = None
    for attempt in range(3):
        try:
            response = model.generate_content(prompt, generation_config=gen_cfg)
            break
        except Exception as exc:
            if not _if_llm_retryable(exc) or attempt == 2:
                logging.error("Fatal/non-retryable LLM error: %s", exc)
                raise
            sleep_time = min(5 * (2 ** attempt), 30)
            logging.warning("Transient LLM error on attempt %d/3. Retrying in %.2fs", attempt + 1, sleep_time)
            time.sleep(sleep_time)

    if response is None:
        raise RuntimeError("LLM call failed after all retries")

    parsed = json.loads(response.text)
    parsed["price"] = _safe_int(parsed.get("price"))
    parsed["year"] = _safe_int(parsed.get("year"))
    parsed["mileage"] = _safe_int(parsed.get("mileage"))
    parsed["make"] = _norm_text(parsed.get("make"))
    parsed["model"] = _norm_text(parsed.get("model"))
    parsed["color"] = _norm_text(parsed.get("color"))
    parsed["transmission"] = _norm_transmission(parsed.get("transmission"))
    parsed["cylinders"] = _norm_cylinders(parsed.get("cylinders"))
    parsed["fuel_type"] = _norm_fuel_type(parsed.get("fuel_type"))
    parsed["city"] = _norm_text(parsed.get("city"))
    parsed["state"] = _norm_state(parsed.get("state"))
    parsed["zip_code"] = _norm_zip(parsed.get("zip_code"))
    return parsed


def _process_run(run_id: str, max_files: int, overwrite: bool) -> dict:
    structured_iso = _normalize_run_id_iso(run_id)
    source_url_map = _load_source_url_map(run_id)
    txt_blobs = _txt_objects_for_run(run_id)
    if max_files > 0:
        txt_blobs = txt_blobs[:max_files]

    processed = written = skipped = errors = 0
    for txt_blob in txt_blobs:
        processed += 1
        try:
            post_id = os.path.splitext(os.path.basename(txt_blob))[0]
            if not post_id:
                raise ValueError("missing post_id from txt path")

            out_key = _llm_output_key(run_id, post_id)
            if not overwrite and _blob_exists(out_key):
                skipped += 1
                continue

            raw_listing = _download_text(txt_blob)
            source_url = source_url_map.get(post_id)
            parsed = _vertex_extract_fields(raw_listing, source_url=source_url)

            out_record = {
                "post_id": post_id,
                "run_id": run_id,
                "scraped_at": structured_iso,
                "source_txt": txt_blob,
                "source_url": source_url,
                "price": parsed.get("price"),
                "year": parsed.get("year"),
                "make": parsed.get("make"),
                "model": parsed.get("model"),
                "mileage": parsed.get("mileage"),
                "color": parsed.get("color"),
                "transmission": parsed.get("transmission"),
                "cylinders": parsed.get("cylinders"),
                "fuel_type": parsed.get("fuel_type"),
                "city": parsed.get("city"),
                "state": parsed.get("state"),
                "zip_code": parsed.get("zip_code"),
                "llm_provider": "vertex",
                "llm_model": LLM_MODEL,
                "llm_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            _upload_jsonl_line(out_key, out_record)
            written += 1
        except Exception as exc:
            errors += 1
            logging.error("LLM extraction failed for %s: %s\n%s", txt_blob, exc, traceback.format_exc())

    return {
        "run_id": run_id,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
    }


def llm_extract_http(request: Request):
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500
    if not PROJECT_ID:
        return jsonify({"ok": False, "error": "missing PROJECT_ID env"}), 500
    if LLM_PROVIDER != "vertex":
        return jsonify({"ok": False, "error": "PoC supports LLM_PROVIDER='vertex' only"}), 400

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    if body.get("healthcheck") is True:
        return jsonify({"ok": True, "healthcheck": True, "function": "extractor-llm-poc"}), 200

    run_id = body.get("run_id")
    all_runs = bool(body.get("all_runs", False))
    max_files = int(body.get("max_files") or MAX_FILES_DEFAULT or 0)
    overwrite = bool(body.get("overwrite")) if "overwrite" in body else OVERWRITE_DEFAULT
    run_limit = int(body.get("run_limit") or 0)

    if all_runs:
        run_ids = _list_run_ids(BUCKET_NAME, SCRAPES_PREFIX)
    elif run_id:
        run_ids = [run_id]
    else:
        all_run_ids = _list_run_ids(BUCKET_NAME, SCRAPES_PREFIX)
        if not all_run_ids:
            return jsonify({"ok": False, "error": f"no run_ids found under {SCRAPES_PREFIX}/"}), 200
        run_ids = [all_run_ids[-1]]

    if run_limit > 0:
        run_ids = run_ids[-run_limit:]

    if not run_ids:
        return jsonify({"ok": True, "processed": 0, "written": 0, "skipped": 0, "errors": 0, "runs": []}), 200

    totals = {"processed": 0, "written": 0, "skipped": 0, "errors": 0}
    per_run = []
    for current_run_id in run_ids:
        run_result = _process_run(current_run_id, max_files=max_files, overwrite=overwrite)
        per_run.append(run_result)
        for key in totals:
            totals[key] += run_result[key]

    result = {
        "ok": True,
        "version": "extractor-llm-poc",
        "runs_requested": len(run_ids),
        "processed": totals["processed"],
        "written": totals["written"],
        "skipped": totals["skipped"],
        "errors": totals["errors"],
        "runs": per_run,
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
