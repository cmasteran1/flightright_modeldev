#!/usr/bin/env python3
"""Publish post-parity-fix v11 artifacts to the production S3 bucket.

Uploads:
  1. flightrightdata/data/meta/airports.csv       -> s3://flightright-models/meta/airports.csv
  2. flightrightdata/data/models/dep_<AIR>_100_v11/  -> s3://flightright-models/models/dep_<AIR>_100_v11/
     (includes the joblib bundle, bins_meta.json, registry.json,
     resolved_features.json, dep_train_metrics_*.json, prediction_samples.parquet,
     and every thr_<T>/ subdirectory file).

Production picks these up two ways:
  - At API startup (`_startup_bootstrap_meta`) airports.csv is downloaded from
    S3 to /data/meta/airports.csv on the Fly volume.
  - At first prediction (or via `_startup_warm_models`), each airline's
    bundle is downloaded from S3 to ~/.flightright/remote_models/.

After running this, the next Fly deploy / restart of the flightright API
will load the post-parity-fix v11 bundles and the corrected airports.csv
(ELP=America/Denver, all hub airports validated).

Usage:
  .venv/bin/python scripts/publish_v11_to_s3.py --dry-run            # default: list what would change
  .venv/bin/python scripts/publish_v11_to_s3.py --execute             # actually upload
  .venv/bin/python scripts/publish_v11_to_s3.py --execute --skip-airports
  .venv/bin/python scripts/publish_v11_to_s3.py --execute --airline AA  # one airline only

Reads AWS creds + E2_ENDPOINT from flightright/.env.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[2]  # total_flightproject/
FLIGHTRIGHT = REPO / "flightright"
DATA = REPO / "flightrightdata"

BUCKET = "flightright-models"


def _load_env() -> None:
    env_file = FLIGHTRIGHT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _s3():
    import boto3
    endpoint = os.environ.get("E2_ENDPOINT", "")
    if endpoint and not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_bundle_dir(bundle_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield (local_path, s3_relative_key) for every file we want to ship.

    Excludes:
      * Non-canonical bundle variants (e.g. *_2025_v10regime_pf040.joblib) —
        those are research-only and should not be the deployable artifact.
      * __pycache__, .DS_Store, hidden files.
    """
    SKIP_NAMES = {".DS_Store"}
    SKIP_BUNDLE_SUFFIXES = ("_2025_v10regime_pf040.joblib", "_2025_pf040.joblib")
    for p in bundle_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name in SKIP_NAMES or p.name.startswith("."):
            continue
        if any(p.name.endswith(suf) for suf in SKIP_BUNDLE_SUFFIXES):
            continue
        rel = p.relative_to(bundle_dir.parent)  # dep_AA_100_v11/...
        yield p, str(rel)


def plan_uploads(skip_airports: bool, airlines: list[str]) -> list[tuple[Path, str]]:
    """Return ordered (local_path, s3_key) pairs."""
    plan = []

    if not skip_airports:
        local = DATA / "data" / "meta" / "airports.csv"
        if not local.exists():
            sys.exit(f"FATAL: missing {local}")
        plan.append((local, "meta/airports.csv"))

    for al in airlines:
        bundle_dir = DATA / "data" / "models" / f"dep_{al}_100_v11"
        if not bundle_dir.exists():
            print(f"WARN: missing {bundle_dir}; skipping {al}")
            continue
        for local, rel_key in _walk_bundle_dir(bundle_dir):
            plan.append((local, f"models/{rel_key}"))
    return plan


def diff_against_s3(s3, plan: list[tuple[Path, str]]) -> list[tuple[Path, str, str]]:
    """For each planned upload, check if S3 has same size+md5.

    Returns list of (local, key, status) where status is 'NEW', 'CHANGED', or 'IDENTICAL'.
    """
    out = []
    for local, key in plan:
        local_size = local.stat().st_size
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            remote_size = int(head.get("ContentLength", -1))
            remote_etag = (head.get("ETag", "") or "").strip('"')
            if remote_size == local_size:
                # Compare md5 only if size matches (fast path)
                # ETag is md5 for non-multipart uploads
                if "-" not in remote_etag:
                    if remote_etag == _md5(local):
                        out.append((local, key, "IDENTICAL"))
                        continue
                out.append((local, key, "CHANGED"))
            else:
                out.append((local, key, "CHANGED"))
        except Exception:
            out.append((local, key, "NEW"))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true",
                   help="Actually upload. Default is dry-run.")
    p.add_argument("--skip-airports", action="store_true",
                   help="Don't upload airports.csv (only the model bundles).")
    p.add_argument("--airline", action="append", default=None,
                   help="Restrict to one or more airlines (default: AA,DL,UA,WN).")
    args = p.parse_args()

    _load_env()
    airlines = [a.upper() for a in (args.airline or ["AA", "DL", "UA", "WN"])]

    s3 = _s3()
    plan = plan_uploads(skip_airports=args.skip_airports, airlines=airlines)
    diff = diff_against_s3(s3, plan)

    new = [d for d in diff if d[2] == "NEW"]
    changed = [d for d in diff if d[2] == "CHANGED"]
    same = [d for d in diff if d[2] == "IDENTICAL"]

    print(f"\nPLAN: {len(plan)} files | new={len(new)}  changed={len(changed)}  identical={len(same)}\n")
    for status, label in [("NEW", new), ("CHANGED", changed)]:
        if not label:
            continue
        print(f"--- {status} ({len(label)}) ---")
        for local, key, _ in label[:30]:
            sz = local.stat().st_size
            print(f"  {key:<70}  {sz:>12,}")
        if len(label) > 30:
            print(f"  ... and {len(label) - 30} more")
        print()

    if not args.execute:
        print("Dry run only. Pass --execute to actually upload.")
        return 0

    upload = new + changed
    if not upload:
        print("Nothing to upload.")
        return 0

    print(f"\nUploading {len(upload)} files to s3://{BUCKET}/...")
    for i, (local, key, status) in enumerate(upload, 1):
        s3.upload_file(str(local), BUCKET, key)
        print(f"  [{i}/{len(upload)}] {status:<8} {key}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
