# HPC DeepD3 Prediction Workflow

Complete guide for running DeepD3 predictions on synthetic microscopy images.

---

## 1. Login to HPC

```bash
ssh iwb3119h@tinyx.nhr.fau.de
cd ~/synthetic-spine-sim
```

---

## 2. Available Models

```
models/
├── DeepD3_32F.h5        ← original model (200nm)
└── DeepD3_32F_94nm.h5   ← fine-tuned model (94nm)
```

Always run **both models** and compare results!

---

## 3. Run Prediction — Model 1 (94nm)

### 3.1 Submit batch job

```bash
cat > run_deepd3_94nm.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=deepd3_94nm
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/deepd3_94nm_%j.log

cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim
source ~/miniconda3/bin/activate
conda activate deepd3_env

python -m deepd3.inference.batch \
outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image.tif \
models/DeepD3_32F_94nm.h5
EOF

sbatch run_deepd3_94nm.sh
```

### 3.2 Rename output after job finishes

```bash
mv outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image.prediction \
   outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image_94nm.prediction
```

---

## 4. Run Prediction — Model 2 (32F original)

### 4.1 Submit batch job

```bash
cat > run_deepd3_32F.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=deepd3_32F
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/deepd3_32F_%j.log

cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim
source ~/miniconda3/bin/activate
conda activate deepd3_env

python -m deepd3.inference.batch \
outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image.tif \
models/DeepD3_32F.h5
EOF

sbatch run_deepd3_32F.sh
```

### 4.2 Rename output after job finishes

```bash
mv outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image.prediction \
   outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image_32F.prediction
```

---

## 5. Expected outputs

```
outputs/sample_004/xy94_z500_spacing100/
├── zstack_..._image.tif              ← input image
├── zstack_..._image_94nm.prediction  ← DeepD3_32F_94nm predictions
└── zstack_..._image_32F.prediction   ← DeepD3_32F predictions
```

---

## 6. Monitor jobs

```bash
squeue -u iwb3119h
tail -f logs/deepd3_94nm_*.log
tail -f logs/deepd3_32F_*.log
```

Expected log output:
```
Loading stack...
Predicting inset 0/17
Predicting inset 1/17
...
Predicting inset 17/17
```

Takes ~15-30 minutes on V100.

---

## 7. Generate Spine CSV (ground truth)

After pipeline job finishes:

```bash
source ~/miniconda3/bin/activate
conda activate deepd3_env

# Update SAMPLE_NAME and EXP_TAG in script first!
# SAMPLE_NAME = "sample_004"
# EXP_TAG     = "xy94_z500_spacing100"

PYTHONPATH=. python scripts/generate_spine_csv.py
```

Output:
```
outputs/sample_004/xy94_z500_spacing100/spine_annotations.csv
```

---

## 8. Download for evaluation

Run from **local terminal:**

```bash
# Download predictions
scp iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image_94nm.prediction .
scp iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image_32F.prediction .

# Download spine CSV
scp iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_004/xy94_z500_spacing100/spine_annotations.csv .

# Download image
scp iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_004/xy94_z500_spacing100/zstack_sample_004_membrane_bornwolf_fiji_xy94_z500_spacing100_image.tif .
```

---

## 9. Evaluation in Colab

Upload to Andreas's notebook:
- `image.tif` → input image
- `spine_annotations.csv` → ground truth spine locations
- `image_94nm.prediction` → DeepD3_32F_94nm predictions
- `image_32F.prediction` → DeepD3_32F predictions

Fix distance resolution in notebook:
```python
# For 94nm/500nm data
d = np.array([94, 94, 500])
```

Fix labels_avg line:
```python
labels_avg = labels[['X', 'Y', 'Pos']].values.astype(float) / r.values[..., None]
```

---

## 10. Partition reference for DeepD3

| Partition | GPU | Memory | Notes |
|-----------|-----|--------|-------|
| v100 | V100 | 32GB | Best for DeepD3 — no quota issues |
| rtx3080 | RTX 3080 | 10GB | Works but slower |
| a100 | A100 | 80GB | Quota issues for some users |
