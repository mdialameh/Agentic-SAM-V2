#!/usr/bin/env python3
"""Restore Agentic-SAM v2 checkpoints and the MedSAM2 source checkout.

Usage:
    python downloader.py            # fetch whatever is missing
    python downloader.py --list     # show what is present/missing
    python downloader.py --only medsam2 medsam2_source
    python downloader.py --force    # re-download even if present

The SAM 3.1 checkpoint lives in a gated Hugging Face repo: accept the license
at https://huggingface.co/facebook/sam3.1 and set HF_TOKEN in .env or the
environment (or pass --token).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MEDSAM2_SOURCE_REPO = "https://github.com/bowang-lab/MedSAM2.git"


def _load_env() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.is_file():
        return
    for raw in env.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


@dataclass
class Artifact:
    key: str
    repo_id: str
    filename: str
    target: Path
    gated: bool = False


ARTIFACTS = [
    Artifact("medsam2", "wanglab/MedSAM2", "MedSAM2_latest.pt",
             PROJECT_ROOT / "checkpoints" / "MedSAM2_latest.pt"),
    Artifact("medical_sam3", "ChongCong/Medical-SAM3", "checkpoint_2D.pt",
             PROJECT_ROOT / "checkpoints" / "medical_sam3_2D.pt"),
    Artifact("sam3", "facebook/sam3.1", "sam3.1_multiplex.pt",
             PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt", gated=True),
]
SOURCE_TARGET = PROJECT_ROOT / "third_party" / "MedSAM2"


def _present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def _download(artifact: Artifact, token: str | None, force: bool) -> str:
    if _present(artifact.target) and not force:
        return f"already present ({artifact.target.stat().st_size / 1e6:.0f} MB), skipped"
    from huggingface_hub import hf_hub_download

    artifact.target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=artifact.repo_id,
            filename=artifact.filename,
            token=token,
            local_dir=str(artifact.target.parent),
        )
    )
    if downloaded.resolve() != artifact.target.resolve():
        shutil.move(str(downloaded), str(artifact.target))
    return f"downloaded ({artifact.target.stat().st_size / 1e6:.0f} MB)"


def _clone_source(force: bool) -> str:
    if SOURCE_TARGET.is_dir() and any(SOURCE_TARGET.iterdir()):
        if not force:
            return "already present, skipped"
        shutil.rmtree(SOURCE_TARGET)
    subprocess.run(
        ["git", "clone", "--depth", "1", MEDSAM2_SOURCE_REPO, str(SOURCE_TARGET)],
        check=True,
    )
    return "cloned"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    keys = [a.key for a in ARTIFACTS] + ["medsam2_source"]
    parser.add_argument("--only", nargs="+", choices=keys)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    _load_env()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HF_KEY")
    selected = set(args.only or keys)

    if args.list:
        for artifact in ARTIFACTS:
            state = "present" if _present(artifact.target) else "MISSING"
            print(f"{artifact.key:16s} {artifact.repo_id}/{artifact.filename} -> "
                  f"{artifact.target.relative_to(PROJECT_ROOT)}  [{state}]")
        src_state = (
            "present" if SOURCE_TARGET.is_dir() and any(SOURCE_TARGET.iterdir()) else "MISSING"
        )
        print(f"{'medsam2_source':16s} {MEDSAM2_SOURCE_REPO} -> third_party/MedSAM2  [{src_state}]")
        return 0

    failed = False
    for artifact in ARTIFACTS:
        if artifact.key not in selected:
            continue
        print(f"[{artifact.key}] {artifact.repo_id}/{artifact.filename}")
        try:
            print("  " + _download(artifact, token, args.force))
        except Exception as exc:
            failed = True
            message = f"  FAILED: {exc.__class__.__name__}: {exc}"
            if artifact.gated:
                message += (
                    f"\n  hint: accept the license at https://huggingface.co/{artifact.repo_id}"
                    " and set HF_TOKEN"
                )
            print(message)

    if "medsam2_source" in selected:
        print("[medsam2_source] " + MEDSAM2_SOURCE_REPO)
        try:
            print("  " + _clone_source(args.force))
        except Exception as exc:
            failed = True
            print(f"  FAILED: {exc.__class__.__name__}: {exc}")

    print("\nDone with errors." if failed else "\nAll artifacts in place. Run: bash run.sh")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
