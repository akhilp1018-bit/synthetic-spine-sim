# Synthetic Spine Simulation Pipeline

This repository contains the workflow used for the synthetic spine microscopy simulation and DeepD3 evaluation/training-data generation.

The pipeline can be used to:

1. Render synthetic microscopy stacks from labelled neuron meshes.
2. Compare different PSF modes.
3. Run DeepD3 on the rendered stacks.
4. Export DeepD3 predictions to probability TIFFs.
5. Evaluate DeepD3 detections.
6. Visualize matching results.
7. Generate DeepD3 training data.
8. Create MIP review sheets for dataset inspection.

---

## 1. Before Running

This README assumes the following HPC setup:

```text
Repository path:
  /home/hpc/iwb3/iwb3119h/synthetic-spine-sim

Main Python environment:
  /home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env

DeepD3 conda environment:
  deepd3_env

Mesh input folders:
  neuron/sample_001/
  neuron/sample_004/
```

If the repository is located somewhere else, change the `REPO` variable below.

```bash
export REPO=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim
cd $REPO
```

Most commands use:

```bash
PYTHONPATH=. $REPO/thesis_env/bin/python <script.py>
```

Check that the main environment can see the GPU:

```bash
PYTHONPATH=. $REPO/thesis_env/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected output should include:

```text
True
NVIDIA GeForce RTX 3080
```

---

## 2. Repository Inputs

The mesh folders should contain one dendrite mesh and multiple spine meshes:

```text
neuron/sample_001/
├── dendrite00.ply
├── spine1.ply
├── spine2.ply
└── ...

neuron/sample_004/
├── dendrite00.ply
├── spine1.ply
├── spine2.ply
└── ...
```

Current known scale settings:

```text
sample_001: scale_to_nm = 1_000_000.0
sample_004: scale_to_nm = 1.0
```

Only use samples where the mesh scale is known.

---

## 3. Useful HPC Commands

Check jobs:

```bash
squeue -u iwb3119h
```

Cancel a job:

```bash
scancel <job_id>
```

Check GPU nodes:

```bash
sinfo -N -o "%N %P %t %G"
```

Start an interactive GPU session:

```bash
salloc.tinygpu --partition=rtx3080 --gres=gpu:rtx3080:1 --time=04:00:00
```

---

## 4. Run the Synthetic Rendering Pipeline

The rendering pipeline creates synthetic microscopy stacks from the labelled meshes.

### Option A: Run interactively on a GPU node

First request an interactive GPU session:

```bash
salloc.tinygpu --partition=rtx3080 --gres=gpu:rtx3080:1 --time=04:00:00
```

Then run:

```bash
export REPO=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/run_pipeline.py
```

### Option B: Run with Slurm

Create a job file:

```bash
cat > run_pipeline_sample001.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=run_pipeline
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/run_pipeline_%j.log

export REPO=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/run_pipeline.py
EOF
```

Submit:

```bash
sbatch run_pipeline_sample001.sh
```

Monitor:

```bash
squeue -u iwb3119h
tail -f logs/run_pipeline_*.log
```

### Expected output

For `sample_001`, the current output folder is:

```text
outputs/sample_001/xy94_z500_spacing100/
```

It contains PSF-specific subfolders:

```text
outputs/sample_001/xy94_z500_spacing100/
├── bornwolf_1p/
├── bornwolf_2p/
└── gaussian_2p/
```

Each PSF folder contains rendered image stacks, masks, DeepD3 outputs, and evaluation outputs.

---

## 5. PSF Modes

The current PSF comparison uses:

```text
bornwolf_1p
bornwolf_2p
gaussian_2p
```

Meaning:

```text
bornwolf_1p  = original Born-Wolf PSF
bornwolf_2p  = 2P-like Born-Wolf effective PSF
gaussian_2p  = 2P-like Gaussian effective PSF
```

For final training-data generation, use:

```text
gaussian_2p
```

---

## 6. Run DeepD3 on Rendered Stacks

DeepD3 inference uses the `deepd3_env` conda environment.

Submit the DeepD3 job:

```bash
cd $REPO
sbatch scripts/run_deepd3_all_psf.sh
```

Check job status:

```bash
squeue -u iwb3119h
```

Check generated prediction files:

```bash
find outputs/sample_001/xy94_z500_spacing100 -path "*deepd3_predictions*" -type f
```

Expected prediction files include:

```text
bornwolf_1p/deepd3_predictions/32F.prediction
bornwolf_1p/deepd3_predictions/32F_94nm.prediction
bornwolf_2p/deepd3_predictions/32F.prediction
bornwolf_2p/deepd3_predictions/32F_94nm.prediction
gaussian_2p/deepd3_predictions/32F.prediction
gaussian_2p/deepd3_predictions/32F_94nm.prediction
```

---

## 7. Export DeepD3 Predictions

Export `.prediction` files to TIFF probability maps:

```bash
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/export_deep3_predictions.py
```

Outputs are saved inside each PSF folder:

```text
<psf_mode>/deepd3_exports/
├── 32F_spine_probability.tif
├── 32F_dendrite_probability.tif
├── 32F_94nm_spine_probability.tif
└── 32F_94nm_dendrite_probability.tif
```

For spine evaluation, the important files are:

```text
32F_spine_probability.tif
32F_94nm_spine_probability.tif
```

---

## 8. Generate Spine Center CSV

Generate the GT spine center CSV used for object-wise evaluation:

```bash
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/generate_spine_csv.py
```

Expected output:

```text
outputs/sample_001/xy94_z500_spacing100/spine_annotations.csv
```

Check:

```bash
head outputs/sample_001/xy94_z500_spacing100/spine_annotations.csv
```

---

## 9. Evaluate DeepD3

Run the evaluation:

```bash
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/evaluate_spines.py
```

The evaluation produces:

```text
PR curve / AP / F1
recall vs matching distance
IoU / Dice
summary tables
```

Outputs are saved inside each PSF folder:

```text
outputs/sample_001/xy94_z500_spacing100/<psf_mode>/evaluation/
```

Example files:

```text
pr_curve.png
pr_curve_results.csv
recall_vs_distance.png
recall_vs_distance_results.csv
iou_dice_curves.png
iou_dice_results.csv
summary.csv
```

Combined summary:

```text
outputs/sample_001/xy94_z500_spacing100/evaluation_summary_all_psfs.csv
```

---

## 10. Current DeepD3 Evaluation Summary

For `sample_001`, the current object-wise evaluation gave:

| PSF mode | Model | AP | Best F1 | Best threshold | Best precision | Best recall |
|---|---|---:|---:|---:|---:|---:|
| bornwolf_1p | DeepD3_32F_94nm | 0.582 | 0.697 | 0.22 | 0.721 | 0.674 |
| bornwolf_1p | DeepD3_32F | 0.534 | 0.621 | 0.20 | 0.561 | 0.696 |
| bornwolf_2p | DeepD3_32F_94nm | 0.531 | 0.667 | 0.20 | 0.593 | 0.761 |
| bornwolf_2p | DeepD3_32F | 0.587 | 0.667 | 0.36 | 0.682 | 0.652 |
| gaussian_2p | DeepD3_32F_94nm | 0.503 | 0.651 | 0.22 | 0.581 | 0.739 |
| gaussian_2p | DeepD3_32F | 0.573 | 0.684 | 0.26 | 0.644 | 0.728 |

Notes:

```text
DeepD3 is mainly evaluated as an object-detection method.
PR/AP/F1 and recall-vs-distance are the main DeepD3-related results.
IoU/Dice are secondary mask-level measurements.
gaussian_2p is used for final training-data generation.
```

---

## 11. Visualize Pipeline and Matching

Run:

```bash
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/visualize_pipeline_and_matching.py
```

Outputs:

```text
outputs/sample_001/xy94_z500_spacing100/<psf_mode>/visualizations/<model>_thr<threshold>_match1000nm/
├── pipeline_visualization.png
├── pipeline_visualization_zoom.png
├── xy_matching.png
├── xz_matching.png
├── yz_matching.png
├── matching_metrics.csv
├── matching_gt_table.csv
└── prediction_table.csv
```

Legend:

```text
white circles = GT centers
cyan points   = matched predictions
magenta       = unmatched predictions / false positives
red x         = missed GT spines
```

---

## 12. Generate Training Data

The final training-data generation uses:

```text
Gaussian 2P-like PSF
sample_001 + sample_004
random rotation
random XY resolution from 60 to 300 nm/px
128 × 128 px patches
16 Z slices
Z step = 500 nm
filtered FOV selection
```

Final output folder:

```text
training_data_gaussian_2p_render_masks_1000/
```

Each instance contains:

```text
image.tif
spine_mask.tif
dendrite_mask.tif
metadata.txt
```

There is also:

```text
training_data_gaussian_2p_render_masks_1000/index.csv
```

### Run 1000 training instances

Create a job file:

```bash
cat > run_training_1000.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=train1000
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/train1000_%j.log

export PYTORCH_ALLOC_CONF=expandable_segments:True

export REPO=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/generate_training_data.py
EOF
```

Submit:

```bash
sbatch run_training_1000.sh
```

Monitor:

```bash
squeue -u iwb3119h
tail -f logs/train1000_*.log
```

Check progress:

```bash
find training_data_gaussian_2p_render_masks_1000 -maxdepth 1 -type d -name "instance_*" | wc -l
```

---

## 13. Create MIP Review Sheets

After generating the dataset, create MIP review sheets:

```bash
cd $REPO

PYTHONPATH=. $REPO/thesis_env/bin/python scripts/visualize_training_mips.py
```

Outputs:

```text
training_data_gaussian_2p_render_masks_1000/review_mips/
├── review_image_mips.png
├── review_overlay_mips.png
├── review_mask_mips.png
└── review_index.csv
```

Meaning:

```text
review_image_mips.png   = synthetic image only
review_mask_mips.png    = masks only
review_overlay_mips.png = synthetic image with masks overlaid
```

The most useful review file is:

```text
review_overlay_mips.png
```

---

## 14. Full Workflow Summary

```bash
export REPO=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim
cd $REPO

# 1. Run synthetic rendering pipeline
sbatch run_pipeline_sample001.sh

# 2. Run DeepD3
sbatch scripts/run_deepd3_all_psf.sh

# 3. Export DeepD3 predictions
PYTHONPATH=. $REPO/thesis_env/bin/python scripts/export_deep3_predictions.py

# 4. Generate GT spine center CSV
PYTHONPATH=. $REPO/thesis_env/bin/python scripts/generate_spine_csv.py

# 5. Evaluate DeepD3
PYTHONPATH=. $REPO/thesis_env/bin/python scripts/evaluate_spines.py

# 6. Visualize matching
PYTHONPATH=. $REPO/thesis_env/bin/python scripts/visualize_pipeline_and_matching.py

# 7. Generate training data
sbatch run_training_1000.sh

# 8. Create training-data review MIPs
PYTHONPATH=. $REPO/thesis_env/bin/python scripts/visualize_training_mips.py
```

---

## 15. Quick Checks

Check one generated training instance:

```bash
$REPO/thesis_env/bin/python - << 'PY'
import os
import numpy as np
import tifffile

root = "training_data_gaussian_2p_render_masks_1000/instance_0001"

for name in ["image.tif", "spine_mask.tif", "dendrite_mask.tif"]:
    arr = tifffile.imread(os.path.join(root, name))
    print(name, arr.shape, arr.dtype, arr.min(), arr.max(), "nonzero:", np.count_nonzero(arr))
PY
```

Expected shapes:

```text
image.tif          (16, 128, 128) uint8
spine_mask.tif     (16, 128, 128) uint8
dendrite_mask.tif  (16, 128, 128) uint8
```

Check generated instance count:

```bash
find training_data_gaussian_2p_render_masks_1000 -maxdepth 1 -type d -name "instance_*" | wc -l
```

Check evaluation results:

```bash
cat outputs/sample_001/xy94_z500_spacing100/evaluation_summary_all_psfs.csv
```

---

## 16. Notes

- The current quantitative evaluation is based on `sample_001`.
- More samples should be evaluated before making strong final conclusions.
- The training-data generation currently uses `sample_001` and `sample_004`.
- `gaussian_2p` is used for the final training dataset.
- The MIP review sheets are used to quickly inspect whether the generated training patches are reasonable.
