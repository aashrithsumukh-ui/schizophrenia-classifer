# Schizophrenia Detection from Brain MRI using 2D CNN Transfer Learning

A deep learning classifier that detects schizophrenia from 3D structural T1-weighted brain MRI scans using 2D axial slice extraction and a fine-tuned ResNet-18 model.

> Based on: *Zhang et al. "Detecting schizophrenia with 3D structural brain MRI using deep learning." Scientific Reports (2023)*

---

## Results

| Metric | Slice-Level |
|--------|-------------|
| Accuracy | 88.17% |
| AUC-ROC | 0.9580 |
| Sensitivity (SCHZ recall) | 87.98% |
| Specificity (CTRL recall) | 88.35% |

Subject-level aggregation (mean probability + majority vote) is also computed automatically.

---

## Dataset

| Dataset | Subjects |
|---------|----------|
| ds000030 | 175 |
| NUSDAST | 163 |
| COBRE | 159 |
| ds004302 | 71 |
| **Total** | **568** |

- **SCHZ:** 271 subjects
- **CTRL:** 297 subjects
- Input format: `.npy` files, `128×128×128` float32 arrays
- Labels provided via `harmonized_labels.csv` with columns: `subject_id`, `filepath`, `label`, `dataset`, `site`

### Preprocessing (done prior to training)
- MNI152 affine registration
- Skull stripping
- Z-score intensity normalization
- Resized to 128×128×128

---

## Approach

### Why 2D slices instead of 3D volumes?
- Enables **ImageNet transfer learning** — pretrained ResNet-18 weights transfer directly to 2D brain slices
- Drastically reduces GPU memory requirements
- Faster training — achieves competitive AUC (0.958) vs the paper's 3D model (0.987)

### Slice Extraction
- Axis: axial (Z-axis, `axis=2`)
- Indices: **44 to 84** (40 slices per volume)
- These indices cover the subcortical region — hippocampus, thalamus, ventricles — the brain areas most affected in schizophrenia

### Per-Slice Preprocessing
```
Raw slice (128×128, float32)
    → NaN/Inf replacement (np.nan_to_num)
    → Min-max normalization → [0, 1]
    → Scale to uint8 [0, 255]
    → Stack 3× → (128, 128, 3) RGB-like
    → ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

---

## Model Architecture

**ResNet-18** pretrained on ImageNet with custom classification head:

```
Pretrained ResNet-18 backbone (layers 1-4)
    → Global Average Pool → (512,)
    → Dropout(0.25)
    → Linear(512 → 2)
    → Softmax → [P(CTRL), P(SCHZ)]
```

Total parameters: ~11.17 million (all trainable)

### Training Details
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Backbone LR | 1e-5 |
| Head LR | 1e-4 |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Loss | Weighted CrossEntropyLoss |
| LR Scheduler | CosineAnnealingLR |
| Seed | 42 |

### Class Imbalance Handling
- `WeightedRandomSampler` — balances SCHZ/CTRL at the batch level
- `CrossEntropyLoss` with inverse-frequency class weights

### Data Split
- **70/15/15** train/val/test
- Stratified by **label AND acquisition site** simultaneously
- Prevents inter-site data leakage
- Split at **subject level** — all slices from one subject stay in one split

---

## Evaluation

Two levels of evaluation are computed:

**Slice-level** — each of the 40 slices per subject evaluated independently

**Subject-level** — slices aggregated per subject using:
- Mean probability: average P(SCHZ) across all 40 slices, threshold at 0.5
- Majority vote: count how many slices predict SCHZ

Metrics: Accuracy, AUC-ROC, Sensitivity, Specificity, Confusion Matrix

---

## Project Structure

```
harmonized/
│
├── 2dcnn_subject.py          ← Main training + evaluation script
├── schizophrenia_classifier.py  ← Original classifier (slice-level only)
├── gradcam_2d.py             ← Grad-CAM visualization script
├── fix_csv_paths.py          ← Fixes filepath column in CSV
├── check_npy.py              ← Diagnostic script for .npy files
│
├── harmonized_labels.csv     ← Subject labels and file paths (not tracked)
├── best_model.pth            ← Saved model checkpoint (not tracked)
├── training_curves.png       ← Loss and accuracy curves (not tracked)
│
└── *.npy                     ← MRI volumes (not tracked — too large)
```

---

## Installation

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib scipy
```

For GPU (NVIDIA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Tested on: Python 3.12, PyTorch 2.x, NVIDIA RTX 3050 6GB, Windows 11

---

## Usage

**Step 1 — Fix CSV paths** (run once after moving files):
```bash
python fix_csv_paths.py
```

**Step 2 — Train and evaluate:**
```bash
python 2dcnn_subject.py
```

**Step 3 — Generate Grad-CAM visualizations** (no retraining needed):
```bash
python gradcam_2d.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint saved during training |
| `training_curves.png` | Loss and accuracy per epoch |
| `gradcam_outputs/` | Per-subject Grad-CAM overlay images |

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| 2D slices not 3D volumes | Enables ImageNet transfer learning; fits in 6GB VRAM |
| Stratify split by site | Prevents model learning scanner artifacts instead of disease |
| Subject-level split | Prevents slice leakage between train and test |
| Differential LRs | Protects pretrained backbone from being overwritten |
| WeightedSampler + weighted loss | Dual-strategy imbalance handling |
| Subject-level aggregation | More clinically meaningful than slice-level prediction |

---

## Reference

Zhang, J. et al. *Detecting schizophrenia with 3D structural brain MRI using deep learning.* Scientific Reports 13, 14433 (2023). https://doi.org/10.1038/s41598-023-41359-z
