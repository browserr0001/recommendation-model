#!/usr/bin/env python3
"""
Create slimmed-down parquet fixtures under tests/fixtures/ from data_sample/.

Each output file keeps the original schema but contains only a small number of rows
to keep CI runs fast. Existing files will be overwritten.
"""

from pathlib import Path
import pandas as pd


SOURCE_DIR = Path("data_sample")
TARGET_DIR = Path("tests/fixtures")

# Number of rows to keep for each parquet (per file path relative to SOURCE_DIR)
ROW_LIMITS = {
    Path("meta/movies.parquet"): 50,
    Path("meta/users.parquet"): 50,
    Path("ratings/ratings.parquet"): 200,
    Path("watches/watches.parquet"): 200,
    # users_new.parquet is intentionally skipped (too large / not needed)
}


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Expected source directory {SOURCE_DIR} to exist.")

    for relative_path, limit in ROW_LIMITS.items():
        src = SOURCE_DIR / relative_path
        if not src.exists():
            print(f"[WARN] Source file missing, skipping: {src}")
            continue

        df = pd.read_parquet(src)
        trimmed = (
            df.sample(n=min(limit, len(df)), random_state=42)
            .reset_index(drop=True)
        )

        dest = TARGET_DIR / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        trimmed.to_parquet(dest, index=False)
        print(f"[OK] Wrote {len(trimmed):>4} rows -> {dest}")

    print("\nFixture generation complete. Files are in tests/fixtures/.")


if __name__ == "__main__":
    main()
