# HPC Workflow — synthetic-spine-sim

Complete guide for running the synthetic microscopy simulation pipeline on TinyGPU (FAU HPC).

---

## 1. Login to HPC

```bash
ssh iwb3119h@tinyx.nhr.fau.de
```

---

## 2. Setup (first time only)

```bash
cd ~/synthetic-spine-sim
mkdir -p logs
```

---

## 3. Pull latest code

Always pull before running to get latest changes:

```bash
cd ~/synthetic-spine-sim
git pull
```

---

## 4. Run Pipeline (sample rendering)

### 4.1 Update settings in `scripts/run_pipeline.py`

```python
SAMPLE_NAME             = "sample_004"   # change per sample
SCALE_TO_NM             = 1              # 1 for nm export, 1_000_000 for mm
SAVE_DEBUG_COMPONENTS   = True           # save individual spine masks
SAVE_DEBUG_CLEAN_IMAGES = False          # skip clean images (saves memory)
```

### 4.2 Create job script

```bash
cat > run_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=spine_sim
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/run_%j.log

export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim
/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/run_pipeline.py
EOF
```

### 4.3 Submit job

```bash
sbatch run_job.sh
```

### 4.4 Expected outputs

```
outputs/sample_004/xy94_z500_spacing100/
├── zstack_..._image.tif           ← combined image (for DeepD3)
├── zstack_..._spine_mask.tif      ← combined spine mask
├── zstack_..._dendrite_mask.tif   ← dendrite mask
├── zstack_..._spine1_mask.tif     ← individual spine masks
├── zstack_..._spine2_mask.tif
...
└── metadata_..._image.txt
```

---

## 5. Generate Training Data (1000 instances)

### 5.1 Update settings in `scripts/generate_training_data.py`

```python
SAMPLE_NAME   = "sample_004"
SCALE_TO_NM   = 1.0
NUM_INSTANCES = 1000
```

### 5.2 Create job script

```bash
cat > run_training.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=train_data
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/training_%j.log

cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim
/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/generate_training_data.py
EOF
```

### 5.3 Submit job

```bash
sbatch run_training.sh
```

### 5.4 Expected outputs

```
training_data/
├── instance_0001/
│   ├── image.tif          ← 128×128×16, 8-bit
│   ├── spine_mask.tif     ← 128×128×16, 8-bit
│   └── dendrite_mask.tif  ← 128×128×16, 8-bit
...
└── instance_1000/
```

---

## 6. Generate Spine CSV (for evaluation)

After pipeline job finishes, generate center-of-mass CSV:

```bash
# Activate conda environment (has scipy, pandas)
source ~/miniconda3/bin/activate
conda activate deepd3_env

# Update SAMPLE_NAME in script first if needed
PYTHONPATH=. python scripts/generate_spine_csv.py
```

Output:
```
outputs/sample_004/xy94_z500_spacing100/spine_annotations.csv
```

---

## 7. Monitor Jobs

### Check job status
```bash
squeue -u iwb3119h
```

### Watch live log output
```bash
tail -f logs/run_*.log         # pipeline job
tail -f logs/training_*.log    # training data job
```

### Check output files
```bash
ls outputs/sample_004/xy94_z500_spacing100/ | wc -l
ls training_data/ | wc -l
```

### Check disk quota
```bash
quota -s
```

---

## 8. Download outputs to local machine

Run from **local terminal** (not HPC):

```bash
# Download spine masks output
scp -r iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_004/ .

# Download training data
scp -r iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/training_data/ .

# Download spine CSV only
scp iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_004/xy94_z500_spacing100/spine_annotations.csv .
```

---

## 9. SCALE_TO_NM reference

| Sample | Blender export | SCALE_TO_NM |
|--------|---------------|-------------|
| sample_001 | default (mm) | 1,000,000 |
| sample_002 | default (mm) | 1,000,000 |
| sample_003 | default (mm) | 1,000,000 |
| sample_004 | nm export    | 1          |
| sample_005+ | nm export   | 1          |

---

## 10. Partition reference

| Partition | GPU | Memory | Use for |
|-----------|-----|--------|---------|
| rtx3080   | RTX 3080 | 10GB | standard runs |
| a100      | A100     | 80GB | large samples |
| v100      | V100     | 32GB | alternative |

---

## Notes

- Always `git pull` before running
- Use `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation
- Files are compressed with zlib — ~100x smaller than uncompressed
- Jobs keep running even after logout
- Resume support: script skips already completed instances
