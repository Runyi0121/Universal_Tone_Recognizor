
#!/bin/bash
#$ -N wav2vec2_gpu_ep40        # Job name
#$ -cwd                   # Run from current directory
#$ -q *@@nlp-a10           # Queue
#$ -l gpu_card=1          # Request 2 GPU
#$ -pe smp 8

# Initialize modules system
source /afs/crc.nd.edu/x86_64_linux/Modules/4.7.0/init/bash
# -----------------------------
# Load modules
# -----------------------------
module load python/3.12.11
module load cuda/12.1
module load cudnn/8.9.3

# -----------------------------
# Activate venv
# -----------------------------
source ~/toneenv312/bin/activate

# -----------------------------
# Run training
# -----------------------------
python3 /afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/train/khum-khum-mand/train.py
