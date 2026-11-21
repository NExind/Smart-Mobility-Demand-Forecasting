#!/usr/bin/env python3
"""
upload_to_kaggle.py — Kaggle Model and Dataset Publishing CLI Tool

Smart Mobility Demand Forecasting and Capacity Allocation
Academic Project Submission Bundle

This script validates local model artifacts, checks Kaggle credentials,
verifies handle syntax according to Kaggle Models conventions, and uploads
the model bundle to Kaggle Models (or Kaggle Datasets as a fallback).

Usage:
    # 1. Dry run verification (does not upload anything):
    python kaggle_export/upload_to_kaggle.py --dry-run --handle <username>/smart-mobility-demand/scikit-learn/hourly-demand

    # 2. Upload to Kaggle Models:
    python kaggle_export/upload_to_kaggle.py --handle <username>/smart-mobility-demand/scikit-learn/hourly-demand

    # 3. Fallback upload to Kaggle Datasets:
    python kaggle_export/upload_to_kaggle.py --mode dataset --handle <username>/smart-mobility-demand-model
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


# Frameworks recognized by Kaggle Models
SUPPORTED_FRAMEWORKS = [
    "scikit-learn",
    "other",
    "transformers",
    "pytorch",
    "tensorflow",
    "jax",
    "keras",
]

RECOMMENDED_FRAMEWORK = "scikit-learn"
RECOMMENDED_VARIATION = "hourly-demand"
DEFAULT_LICENSE = "CC-BY-NC-4.0"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_artifacts(model_dir: Path) -> dict:
    """
    Verify all required model bundle files exist and are valid.
    Returns dictionary with status and metadata.
    """
    results = {
        "valid": True,
        "errors": [],
        "files": {},
    }

    if not model_dir.exists():
        results["valid"] = False
        results["errors"].append(f"Model directory does not exist: {model_dir}")
        return results

    # 1. Check bike_demand_model.pkl
    pkl_path = model_dir / "bike_demand_model.pkl"
    if not pkl_path.exists():
        results["valid"] = False
        results["errors"].append(f"Missing required model artifact: {pkl_path}")
    elif pkl_path.stat().st_size == 0:
        results["valid"] = False
        results["errors"].append(f"Model artifact file is empty: {pkl_path}")
    else:
        file_size = pkl_path.stat().st_size
        file_sha256 = compute_sha256(pkl_path)
        results["files"]["bike_demand_model.pkl"] = {
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "sha256": file_sha256,
        }

    # 2. Check model_metadata.json
    json_path = model_dir / "model_metadata.json"
    if not json_path.exists():
        results["valid"] = False
        results["errors"].append(f"Missing required metadata file: {json_path}")
    else:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            results["files"]["model_metadata.json"] = {
                "size_bytes": json_path.stat().st_size,
                "model_name": metadata.get("model_name", "unknown"),
                "artifact_version": metadata.get("model_version") or metadata.get("artifact_version", "unknown"),
                "evaluation_metrics": metadata.get("evaluation_metrics", {}).get("ml_model_test", {}),
            }
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Invalid JSON in {json_path}: {e}")

    return results


def check_authentication() -> dict:
    """
    Check Kaggle credentials securely without printing or leaking secrets.
    Returns authentication status and username (if found).
    """
    result = {
        "authenticated": False,
        "username": None,
        "source": None,
        "details": "",
    }

    # Check environment variables
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        result["authenticated"] = True
        result["username"] = env_user.strip()
        result["source"] = "Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)"
        result["details"] = f"Username '{result['username']}' configured via environment variables."
        return result

    # Check ~/.kaggle/kaggle.json
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                creds = json.load(f)
            user = creds.get("username")
            key = creds.get("key")
            if user and key:
                result["authenticated"] = True
                result["username"] = str(user).strip()
                result["source"] = f"Credentials file ({kaggle_json_path})"
                result["details"] = f"Username '{result['username']}' configured via {kaggle_json_path}."
                return result
            else:
                result["details"] = f"Found {kaggle_json_path} but missing 'username' or 'key' field."
                return result
        except Exception as e:
            result["details"] = f"Error reading {kaggle_json_path}: {e}"
            return result

    result["details"] = (
        "No Kaggle credentials found. To authenticate, do ONE of the following:\n"
        "  1. Download 'kaggle.json' from https://www.kaggle.com/settings -> 'Create New Token'\n"
        f"     and place it in: {kaggle_json_path}\n"
        "  2. Set environment variables:\n"
        "     $env:KAGGLE_USERNAME = '<your-kaggle-username>'\n"
        "     $env:KAGGLE_KEY = '<your-kaggle-api-key>'"
    )
    return result


def check_dependencies() -> dict:
    """Check whether kagglehub and kaggle packages are installed."""
    deps = {
        "kagglehub": {"installed": False, "version": None},
        "kaggle": {"installed": False, "version": None},
    }

    try:
        import kagglehub
        deps["kagglehub"]["installed"] = True
        deps["kagglehub"]["version"] = getattr(kagglehub, "__version__", "installed")
    except ImportError:
        deps["kagglehub"]["installed"] = False

    try:
        import kaggle
        deps["kaggle"]["installed"] = True
        deps["kaggle"]["version"] = getattr(kaggle, "__version__", "installed")
    except ImportError:
        deps["kaggle"]["installed"] = False

    return deps


def validate_model_handle(handle: str) -> tuple[bool, str, dict]:
    """
    Validate a Kaggle Models handle string.
    Expected format: <owner>/<model>/<framework>/<variation>
    Example: username/smart-mobility-demand/scikit-learn/hourly-demand
    """
    parts = handle.strip().split("/")
    if len(parts) != 4:
        return (
            False,
            f"Invalid Kaggle Models handle format: '{handle}'.\n"
            f"Expected exactly 4 components: <owner>/<model>/<framework>/<variation>\n"
            f"Example: your-username/smart-mobility-demand/scikit-learn/hourly-demand",
            {},
        )

    owner, model, framework, variation = parts
    slug_regex = re.compile(r"^[a-zA-Z0-9_\-]+$")

    for name, val in [("owner", owner), ("model", model), ("framework", framework), ("variation", variation)]:
        if not val or not slug_regex.match(val):
            return (
                False,
                f"Invalid handle component '{name}': '{val}'. Must contain only alphanumeric characters, underscores, or hyphens.",
                {},
            )

    framework_lower = framework.lower()
    note = ""
    if framework_lower not in SUPPORTED_FRAMEWORKS:
        note = (
            f"Note: Kaggle Models officially recommends frameworks such as {SUPPORTED_FRAMEWORKS}.\n"
            f"Using '{framework}'. If Kaggle rejects custom frameworks, use '{RECOMMENDED_FRAMEWORK}' or 'other'."
        )

    parsed = {
        "owner": owner,
        "model": model,
        "framework": framework,
        "variation": variation,
        "note": note,
    }
    return True, "Valid Kaggle Models handle", parsed


def validate_dataset_handle(handle: str) -> tuple[bool, str, dict]:
    """
    Validate a Kaggle Dataset handle string.
    Expected format: <owner>/<dataset-slug>
    Example: username/smart-mobility-demand-model
    """
    parts = handle.strip().split("/")
    if len(parts) != 2:
        return (
            False,
            f"Invalid Kaggle Dataset handle format: '{handle}'.\n"
            f"Expected exactly 2 components: <owner>/<dataset-slug>\n"
            f"Example: your-username/smart-mobility-demand-model",
            {},
        )

    owner, dataset = parts
    slug_regex = re.compile(r"^[a-zA-Z0-9_\-]+$")

    for name, val in [("owner", owner), ("dataset", dataset)]:
        if not val or not slug_regex.match(val):
            return (
                False,
                f"Invalid handle component '{name}': '{val}'. Must contain only alphanumeric characters, underscores, or hyphens.",
                {},
            )

    return True, "Valid Kaggle Dataset handle", {"owner": owner, "dataset": dataset}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish Smart Mobility Demand Forecasting & Capacity Allocation model to Kaggle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check everything in dry-run mode (no network calls or uploads):
  python kaggle_export/upload_to_kaggle.py --dry-run --handle username/smart-mobility-demand/scikit-learn/hourly-demand

  # Upload to Kaggle Models:
  python kaggle_export/upload_to_kaggle.py --handle username/smart-mobility-demand/scikit-learn/hourly-demand

  # Fallback: Upload as a Kaggle Dataset:
  python kaggle_export/upload_to_kaggle.py --mode dataset --handle username/smart-mobility-demand-model
        """,
    )

    parser.add_argument(
        "--handle",
        type=str,
        help="Target handle. For models: <owner>/<model>/<framework>/<variation>. For datasets: <owner>/<dataset-slug>.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["model", "dataset"],
        default="model",
        help="Upload destination: 'model' for Kaggle Models (default) or 'dataset' for Kaggle Datasets fallback.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).parent / "model"),
        help="Path to directory containing the model files (default: kaggle_export/model).",
    )
    parser.add_argument(
        "--version-notes",
        type=str,
        default="V1: Complete bundle with HistGradientBoostingRegressor, KMeans, preprocessors, baselines, and metadata.",
        help="Description of changes or release notes for this version.",
    )
    parser.add_argument(
        "--license",
        type=str,
        default=DEFAULT_LICENSE,
        help=f"License identifier for the model (default: {DEFAULT_LICENSE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all local validations and check credentials without executing the upload.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Bypass interactive confirmation prompt.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print("  Kaggle Model Publishing — Smart Mobility Demand Forecasting")
    print("=" * 72)

    model_dir = Path(args.model_dir).resolve()
    print(f"\n[1/5] Checking local artifacts in: {model_dir}")
    artifact_check = check_artifacts(model_dir)

    if not artifact_check["valid"]:
        print("  [ERROR] Artifact check failed:")
        for err in artifact_check["errors"]:
            print(f"    - {err}")
        sys.exit(1)

    print("  [OK] Model artifacts verified:")
    for fname, info in artifact_check["files"].items():
        if fname == "bike_demand_model.pkl":
            print(f"    - {fname}: {info['size_bytes']:,} bytes ({info['size_mb']} MB)")
            print(f"      SHA-256: {info['sha256']}")
        elif fname == "model_metadata.json":
            metrics = info.get("evaluation_metrics", {})
            print(f"    - {fname}: {info['size_bytes']:,} bytes")
            print(f"      Model: {info.get('model_name')} (v{info.get('artifact_version')})")
            if metrics:
                print(f"      Test Metrics: MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}, R2={metrics.get('r2')}")

    # 2. Check dependencies
    print("\n[2/5] Checking Python Kaggle dependencies:")
    deps = check_dependencies()
    for pkg, info in deps.items():
        if info["installed"]:
            print(f"  [OK] {pkg}: installed (v{info['version']})")
        else:
            print(f"  [WARNING] {pkg}: NOT installed.")

    if not deps["kagglehub"]["installed"] and not deps["kaggle"]["installed"]:
        print("\n  To install the required upload libraries, run:")
        print("    pip install -r kaggle_export/requirements-kaggle.txt")
        if not args.dry_run:
            print("\n  [ERROR] Cannot proceed with upload without kagglehub or kaggle installed.")
            sys.exit(1)

    # 3. Check credentials
    print("\n[3/5] Checking Kaggle authentication:")
    auth = check_authentication()
    if auth["authenticated"]:
        print(f"  [OK] Authenticated via: {auth['source']}")
        print(f"  Kaggle Username: {auth['username']}")
    else:
        print("  [WARNING] Kaggle credentials not found.")
        print(f"  {auth['details']}")
        if not args.dry_run:
            print("\n  [ERROR] Cannot upload without Kaggle credentials.")
            sys.exit(1)

    # 4. Handle validation
    print(f"\n[4/5] Validating upload target (mode: {args.mode}):")

    # If handle not provided, try to build default
    handle = args.handle
    if not handle:
        username = auth["username"] if auth["username"] else "<your-username>"
        if args.mode == "model":
            handle = f"{username}/smart-mobility-demand/{RECOMMENDED_FRAMEWORK}/{RECOMMENDED_VARIATION}"
        else:
            handle = f"{username}/smart-mobility-demand-model"
        print(f"  No --handle specified. Using suggested default: {handle}")

    if args.mode == "model":
        is_valid, msg, parsed = validate_model_handle(handle)
        if not is_valid:
            print(f"  [ERROR] {msg}")
            sys.exit(1)
        print(f"  [OK] Valid Kaggle Models Handle:")
        print(f"    - Owner:     {parsed['owner']}")
        print(f"    - Model:     {parsed['model']}")
        print(f"    - Framework: {parsed['framework']}")
        print(f"    - Variation: {parsed['variation']}")
        if parsed.get("note"):
            print(f"    - {parsed['note']}")
    else:
        is_valid, msg, parsed = validate_dataset_handle(handle)
        if not is_valid:
            print(f"  [ERROR] {msg}")
            sys.exit(1)
        print(f"  [OK] Valid Kaggle Dataset Handle:")
        print(f"    - Owner:   {parsed['owner']}")
        print(f"    - Dataset: {parsed['dataset']}")

    # 5. Execution or Dry Run
    print("\n[5/5] Upload Execution Plan:")
    print(f"  Target Mode:    {args.mode.upper()}")
    print(f"  Target Handle:  {handle}")
    print(f"  Model Folder:   {model_dir}")
    print(f"  Version Notes:  {args.version_notes}")
    print(f"  License:        {args.license}")
    print(f"  Dry Run:        {args.dry_run}")

    if args.dry_run:
        print("\n" + "=" * 72)
        print("  DRY RUN COMPLETED SUCCESSFULLY.")
        print("  All artifact files verified, handle structure validated.")
        if auth["authenticated"]:
            print(f"  Kaggle authentication is ready for user: '{auth['username']}'.")
        else:
            print("  Note: Authenticate before running without --dry-run.")
        print("=" * 72)
        return

    # Check for confirmation
    if not args.yes:
        print("\n" + "-" * 72)
        confirm = input(f"Confirm upload to Kaggle ({args.mode}) at handle '{handle}'? [y/N]: ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("Upload cancelled by user.")
            sys.exit(0)

    # Perform Upload
    if args.mode == "model":
        if not deps["kagglehub"]["installed"]:
            print("[ERROR] 'kagglehub' package is required for Kaggle Models upload.")
            print("Run: pip install kagglehub>=0.2.0")
            sys.exit(1)

        import kagglehub
        print(f"\nUploading model to Kaggle Models ({handle})...")
        try:
            kagglehub.model_upload(
                handle=handle,
                local_model_dir=str(model_dir),
                version_notes=args.version_notes,
                license_name=args.license,
            )
            print("\n[SUCCESS] Model successfully uploaded to Kaggle Models!")
            print(f"Access your model at: https://www.kaggle.com/models/{handle}")
        except Exception as e:
            print(f"\n[ERROR] Model upload failed: {e}")
            print("\nTroubleshooting tips:")
            print("  1. Verify the model owner in the handle matches your Kaggle username.")
            print("  2. Ensure the model exists on Kaggle or that your account has model creation permissions.")
            print("  3. If Kaggle Models framework restrictions occur, consider fallback:")
            print(f"     python kaggle_export/upload_to_kaggle.py --mode dataset --handle {parsed['owner']}/smart-mobility-demand-model")
            sys.exit(1)

    elif args.mode == "dataset":
        # Check/create dataset-metadata.json
        meta_file = model_dir / "dataset-metadata.json"
        if not meta_file.exists():
            ds_meta = {
                "title": "Smart Mobility Demand Forecasting & Capacity Allocation Model",
                "id": handle,
                "licenses": [{"name": "CC-BY-NC-4.0"}],
                "description": "Trained machine learning models and metadata for hourly bicycle demand forecasting and contextual capacity allocation.",
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(ds_meta, f, indent=2)
            print(f"  Created temporary dataset metadata file: {meta_file}")

        try:
            if deps["kagglehub"]["installed"]:
                import kagglehub
                print(f"\nUploading dataset via kagglehub ({handle})...")
                kagglehub.dataset_upload(
                    handle=handle,
                    local_dataset_dir=str(model_dir),
                    version_notes=args.version_notes,
                )
            elif deps["kaggle"]["installed"]:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                print(f"\nCreating dataset via Kaggle API ({handle})...")
                api.dataset_create_new(
                    folder=str(model_dir),
                    dir_mode="zip",
                    quiet=False,
                )
            else:
                print("[ERROR] Neither 'kagglehub' nor 'kaggle' package is installed.")
                sys.exit(1)

            print("\n[SUCCESS] Dataset successfully uploaded to Kaggle!")
            print(f"Access your dataset at: https://www.kaggle.com/datasets/{handle}")
        except Exception as e:
            print(f"\n[ERROR] Dataset upload failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
