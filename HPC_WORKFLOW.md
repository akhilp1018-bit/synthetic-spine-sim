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






# HPC Workflow — synthetic-spine-sim

Complete guide for running the synthetic microscopy simulation pipeline and training-data generation on TinyGPU / FAU HPC.

---

## 1. Login to HPC

```bash
ssh iwb3119h@tinyx.nhr.fau.de
```

---

## 2. Go to repository

```bash
cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim
```

Check location:

```bash
pwd
ls
```

---

## 3. Pull latest code

Always pull before running:

```bash
git pull origin main
git log --oneline -3
```

---

## 4. Check GPU availability

```bash
sinfo -N -o "%N %P %t %G"
```

Meaning:

```text
idle  = free
mix   = partly used, maybe usable
alloc = busy/full
PD    = pending
R     = running
```

Check your jobs:

```bash
squeue -u iwb3119h
```

Cancel a job if needed:

```bash
scancel JOBID
```

---

## 5. Run Pipeline: PSF comparison

This renders one full labelled sample with different PSF modes.

### 5.1 Update settings in `scripts/run_pipeline.py`

For `sample_001`:

```python
SAMPLE_NAME = "sample_001"
SCALE_TO_NM = 1000000

PSF_MODES = [
    "bornwolf_1p",
    "bornwolf_2p",
    "gaussian_2p",
]

SAVE_DEBUG_COMPONENTS = False
SAVE_DEBUG_CLEAN_IMAGES = False
```

For `sample_004`:

```python
SAMPLE_NAME = "sample_004"
SCALE_TO_NM = 1
```

Use debug only if individual spine masks are needed:

```python
SAVE_DEBUG_COMPONENTS = True
SAVE_DEBUG_CLEAN_IMAGES = False
```

For PSF comparison, keep both debug settings `False`.

---

### 5.2 Create batch job

Important: on TinyGPU, do **not** use `#SBATCH --mem` for GPU jobs.

```bash
mkdir -p logs

cat > run_sample001_psf.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=sample001_psf
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/sample001_psf_%j.log

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim

git pull origin main

/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python -m py_compile scripts/run_pipeline.py
/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/run_pipeline.py
EOF
```

Submit:

```bash
sbatch run_sample001_psf.sh
```

Check:

```bash
squeue -u iwb3119h
```

Watch log:

```bash
tail -f logs/sample001_psf_*.log
```

Check GPU usage after job starts:

```bash
srun --jobid=JOBID nvidia-smi
```

---

### 5.3 If RTX3080 is busy

Edit the job file:

```bash
nano run_sample001_psf.sh
```

For A100:

```bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
```

For V100:

```bash
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
```

Submit again:

```bash
sbatch run_sample001_psf.sh
```

---

### 5.4 Expected output for PSF comparison

For 3 PSF modes, expected folders:

```text
outputs/sample_001/xy94_z500_spacing100/
├── bornwolf_1p/
├── bornwolf_2p/
└── gaussian_2p/
```

Inside each folder:

```text
zstack_..._image.tif
zstack_..._spines_clean.tif
zstack_..._dendrite_clean.tif
zstack_..._spine_mask.tif
zstack_..._dendrite_mask.tif
metadata_..._image.txt
```

Check output:

```bash
ls outputs/sample_001/xy94_z500_spacing100/
ls outputs/sample_001/xy94_z500_spacing100/bornwolf_2p
ls outputs/sample_001/xy94_z500_spacing100/gaussian_2p
```

---

## 6. Generate Training Data

This generates 128×128 image snippets for DeepD3 training.

Current output format:

```text
training_data/
├── instance_0001/
│   ├── image.tif
│   ├── spine_mask.tif
│   └── dendrite_mask.tif
...
└── instance_1000/
```

Each file is 8-bit.

---

### 6.1 Important current status

Current `generate_training_data.py` uses one sample at a time:

```python
SAMPLE_NAME = "sample_004"
BASE_DIR = f"neuron/{SAMPLE_NAME}"
```

Later, update this script to randomly choose from multiple samples:

```text
sample_001
sample_002
sample_003
sample_004
...
```

Do not run the full 1000 instances until the multi-sample version is ready and the best PSF is selected.

---

### 6.2 First test with small number

In `scripts/generate_training_data.py`, first set:

```python
NUM_INSTANCES = 5
```

Use:

```python
SAMPLE_NAME = "sample_004"
SCALE_TO_NM = 1.0
```

For sample_001, use:

```python
SAMPLE_NAME = "sample_001"
SCALE_TO_NM = 1000000
```

---

### 6.3 Create training-data batch job

```bash
cat > run_training_test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=train_test
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/training_test_%j.log

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim

git pull origin main

/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python -m py_compile scripts/generate_training_data.py
/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/generate_training_data.py
EOF
```

Submit:

```bash
sbatch run_training_test.sh
```

Watch:

```bash
squeue -u iwb3119h
tail -f logs/training_test_*.log
```

Check output:

```bash
ls training_data/
ls training_data/instance_0001/
```

Expected:

```text
image.tif
spine_mask.tif
dendrite_mask.tif
```

---

### 6.4 Full training-data generation

Only after the test works, change:

```python
NUM_INSTANCES = 1000
```

Then create full job:

```bash
cat > run_training_1000.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=train1000
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/logs/training1000_%j.log

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim

git pull origin main

/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python -m py_compile scripts/generate_training_data.py
/home/hpc/iwb3/iwb3119h/synthetic-spine-sim/thesis_env/bin/python scripts/generate_training_data.py
EOF
```

Submit:

```bash
sbatch run_training_1000.sh
```

Watch:

```bash
tail -f logs/training1000_*.log
```

Count generated instances:

```bash
ls training_data/ | wc -l
```

---

## 7. Generate Spine CSV for Evaluation

After full sample rendering finishes, generate center-of-mass CSV:

```bash
cd /home/hpc/iwb3/iwb3119h/synthetic-spine-sim

source ~/miniconda3/bin/activate
conda activate deepd3_env

PYTHONPATH=. python scripts/generate_spine_csv.py
```

Expected output:

```text
outputs/sample_001/xy94_z500_spacing100/spine_annotations.csv
```

If using PSF subfolders, check the exact path before running evaluation.

---

## 8. Monitor Jobs

Check job status:

```bash
squeue -u iwb3119h
```

Detailed job info:

```bash
scontrol show job JOBID | grep -E "JobState|Reason|Partition|NumNodes|TRES|SubmitTime|StartTime"
```

Watch logs:

```bash
tail -f logs/sample001_psf_*.log
tail -f logs/training_test_*.log
tail -f logs/training1000_*.log
```

Check disk quota:

```bash
quota -s
```

Check folder size:

```bash
du -sh outputs/
du -sh training_data/
```

---

## 9. Download outputs to local machine

Run these from local terminal, not HPC.

Download PSF comparison output:

```bash
scp -r iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_001/ .
```

Download training data:

```bash
scp -r iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/training_data/ .
```

Download only CSV:

```bash
scp iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/outputs/sample_001/xy94_z500_spacing100/spine_annotations.csv .
```

Download logs:

```bash
scp -r iwb3119h@tinyx.nhr.fau.de:~/synthetic-spine-sim/logs/ .
```

---

## 10. SCALE_TO_NM reference

| Sample      | Blender export | SCALE_TO_NM |
| ----------- | -------------- | ----------- |
| sample_001  | default / mm   | 1000000     |
| sample_002  | default / mm   | 1000000     |
| sample_003  | default / mm   | 1000000     |
| sample_004  | nm export      | 1           |
| sample_005+ | nm export      | 1           |

---

## 11. Partition reference

| Partition | GPU      | Use for                     |
| --------- | -------- | --------------------------- |
| rtx3080   | RTX 3080 | standard runs               |
| a100      | A100     | large samples / faster runs |
| v100      | V100     | alternative                 |

Do not use `#SBATCH --mem` for GPU jobs on this system.

---

## 12. Recommended workflow now

1. Finish current PSF comparison job.
2. Inspect:

   ```text
   bornwolf_1p
   bornwolf_2p
   gaussian_2p
   ```
3. Choose best PSF for training-data generation.
4. Update `generate_training_data.py` for multiple samples.
5. Test with:

   ```python
   NUM_INSTANCES = 5
   ```
6. Then run:

   ```python
   NUM_INSTANCES = 1000
   ```

---

## Notes

* Always `git pull origin main` before running.
* Jobs keep running after logout.
* Use `PYTORCH_ALLOC_CONF=expandable_segments:True`.
* Avoid debug outputs for PSF comparison unless individual spine masks are needed.
* For training data, first test with 5 instances before running 1000.
* Full 1000 training instances should use multiple samples and the selected final PSF.
