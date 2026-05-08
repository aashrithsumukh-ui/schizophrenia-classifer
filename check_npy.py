"""
check_npy.py - inspect the raw bytes of a .npy file
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Check first file
import pandas as pd
df = pd.read_csv(os.path.join(HERE, "harmonized_labels.csv"))
fp = df.iloc[0]['filepath']

print(f"File: {fp}")
print(f"File size: {os.path.getsize(fp):,} bytes")

# Load raw
vol = np.load(fp)
print(f"Shape : {vol.shape}")
print(f"Dtype : {vol.dtype}")
print(f"Total elements : {vol.size}")
print(f"NaN count  : {np.isnan(vol).sum()}")
print(f"Zero count : {(vol == 0).sum()}")
print(f"Non-zero, non-NaN count: {(~np.isnan(vol) & (vol != 0)).sum()}")

# Check raw values of first few elements
flat = vol.flatten()
print(f"\nFirst 20 raw values: {flat[:20]}")
print(f"Last  20 raw values: {flat[-20:]}")

# Check a specific slice range
print(f"\nChecking slices 44-84 for any non-NaN values...")
for i in range(44, 84, 10):
    slc = vol[:, :, i]
    non_nan = (~np.isnan(slc)).sum()
    print(f"  Slice {i}: {non_nan}/{slc.size} non-NaN values")

# Try loading with allow_pickle
print("\nTrying np.load with allow_pickle=True...")
try:
    vol2 = np.load(fp, allow_pickle=True)
    print(f"  Type: {type(vol2)}")
    if hasattr(vol2, 'shape'):
        print(f"  Shape: {vol2.shape}")
    elif vol2.ndim == 0:
        # might be a pickled object
        obj = vol2.item()
        print(f"  Pickled object type: {type(obj)}")
        if hasattr(obj, 'shape'):
            print(f"  Inner shape: {obj.shape}")
            print(f"  Inner dtype: {obj.dtype}")
            print(f"  Inner NaN count: {np.isnan(obj).sum()}")
            print(f"  Inner min/max: {np.nanmin(obj):.4f} / {np.nanmax(obj):.4f}")
except Exception as e:
    print(f"  Error: {e}")
