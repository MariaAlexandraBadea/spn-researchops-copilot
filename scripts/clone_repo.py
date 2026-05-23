from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="GitHub repository URL")
    parser.add_argument("--out", default="data/source_repo", help="Output folder")
    parser.add_argument("--reset", action="store_true", help="Delete output folder before cloning")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and args.reset:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        print(f"[INFO] Output folder already exists and is not empty: {out}")
        print("[INFO] Use --reset to clone again.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", args.repo, str(out)]
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[OK] Repository cloned to {out}")


if __name__ == "__main__":
    main()
