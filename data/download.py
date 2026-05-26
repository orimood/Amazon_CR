"""Download Amazon Reviews 2023 (5 verticals) from HuggingFace.

Files per vertical:
  - benchmark/0core/rating_only/<Cat>.csv         (user, item, rating, timestamp)
  - raw/meta_categories/meta_<Cat>.jsonl          (item metadata)

We do our own 5-core filter downstream, so we pull 0core ratings.
"""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"
CATS = ["Books", "Movies_and_TV", "CDs_and_Vinyl", "Video_Games", "Toys_and_Games"]

ROOT = Path(__file__).resolve().parent
REVIEWS_DIR = ROOT / "raw" / "reviews"
META_DIR = ROOT / "raw" / "meta"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


def fetch(repo_path: str, local_dir: Path) -> Path:
    name = repo_path.split("/")[-1]
    out = local_dir / name
    if out.exists():
        print(f"  [skip] {name} ({out.stat().st_size/1e9:.2f} GB already on disk)", flush=True)
        return out
    t0 = time.time()
    print(f"  [pull] {repo_path}", flush=True)
    p = hf_hub_download(
        repo_id=REPO,
        filename=repo_path,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    # hf_hub_download nests the path; move to flat layout
    src = Path(p)
    if src != out:
        src.rename(out)
        # clean up empty parent dirs created by hf_hub_download
        parent = src.parent
        while parent != local_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    sz = out.stat().st_size / 1e9
    dt = time.time() - t0
    print(f"  [done] {name} -> {sz:.2f} GB in {dt:.0f}s", flush=True)
    return out


def main():
    # smaller files first so the disk-hungry ones (Books meta 14.7 GB) come last
    plan = []
    for c in CATS:
        plan.append((f"benchmark/0core/rating_only/{c}.csv", REVIEWS_DIR))
    for c in CATS:
        plan.append((f"raw/meta_categories/meta_{c}.jsonl", META_DIR))

    print(f"Downloading {len(plan)} files from {REPO}...", flush=True)
    for repo_path, local_dir in plan:
        try:
            fetch(repo_path, local_dir)
        except Exception as e:
            print(f"  [FAIL] {repo_path}: {e}", flush=True)
            sys.exit(1)

    print("\nAll done. Final layout:", flush=True)
    for d in (REVIEWS_DIR, META_DIR):
        for p in sorted(d.iterdir()):
            print(f"  {p.stat().st_size/1e9:7.2f} GB  {p.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
