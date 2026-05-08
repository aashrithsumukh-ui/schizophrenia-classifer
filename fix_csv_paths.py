"""
fix_csv_paths.py
----------------
Run this ONCE before training. It rewrites the 'filepath' column in
harmonized_labels.csv so every row points to the correct absolute path
on THIS machine, based on where the .npy files actually live.

Usage:
    python fix_csv_paths.py
"""

import os
import pandas as pd

# ── Folder that contains both this script AND harmonized_labels.csv ──────────
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "harmonized_labels.csv")

df = pd.read_csv(CSV)

print(f"Loaded CSV with {len(df)} rows.")
print(f"Current 'filepath' sample:\n  {df['filepath'].iloc[0]}\n")

fixed, missing = 0, []

for i, row in df.iterrows():
    # Extract just the filename (e.g. ds000030_sub-10159.npy)
    # regardless of what path prefix is stored in the CSV
    fname = os.path.basename(str(row["filepath"]))
    full_path = os.path.join(HERE, fname)

    if os.path.exists(full_path):
        df.at[i, "filepath"] = full_path
        fixed += 1
    else:
        missing.append(fname)

print(f"Fixed : {fixed} paths")
if missing:
    print(f"Missing ({len(missing)} files not found in {HERE}):")
    for m in missing[:10]:   # show first 10 only
        print(f"  {m}")
    if len(missing) > 10:
        print(f"  ... and {len(missing)-10} more")
else:
    print("All .npy files found ✓")

df.to_csv(CSV, index=False)
print(f"\nSaved updated CSV → {CSV}")
print("You can now run schizophrenia_classifier.py")
