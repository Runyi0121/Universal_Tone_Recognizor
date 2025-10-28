#!/bin/bash
#$ -N wav2vec2_eng        # Job name
#$ -cwd                   # Run from current directory
#$ -q *@@nlp-a10          # Queue
#$ -l gpu_card=1         
#$ -pe smp 2
#$ -l h_vmem=300G

# -----------------------------
# 0. MODULES & ENV
# -----------------------------
source /afs/crc.nd.edu/x86_64_linux/Modules/4.7.0/init/bash
module load cuda/12.1
module load cudnn/8.9.3
module load python/3.12.11

# Activate virtual environment
source ~/toneenv312/bin/activate

# -----------------------------
# 1. SCRATCH SETUP
# -----------------------------
if [ -z "$TMPDIR" ]; then
    export TMPDIR="/tmp/$USER/$JOB_ID"
    mkdir -p "$TMPDIR"
fi

DATA_DEST_DIR="$TMPDIR/my_dataset"
AUDIO_DEST_DIR="$DATA_DEST_DIR/audio_root"
mkdir -p "$AUDIO_DEST_DIR"
echo "Using scratch directory: $DATA_DEST_DIR"

# -----------------------------
# 2. COPY METADATA
# -----------------------------
DATA_SOURCE_DIR="/afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/train/sge-eng+thai-mand"
echo "Copying metadata.csv..."
rsync -avh "$DATA_SOURCE_DIR/metadata.csv" "$DATA_DEST_DIR/"
echo "Sample of metadata.csv:"
head -n 5 "$DATA_DEST_DIR/metadata.csv"

# -----------------------------
# 3. COPY AUDIO FILES
# -----------------------------
echo "Copying audio files to $AUDIO_DEST_DIR..."

# Extract all audio paths (last column of CSV) and feed into rsync
tail -n +2 "$DATA_DEST_DIR/metadata.csv" | rev | cut -d',' -f1 | rev \
  | rsync -avhR --progress --files-from=- / "$AUDIO_DEST_DIR/" 2>&1 | tee "$DATA_DEST_DIR/rsync.log"

echo "Data copy complete."

# Verify how many files were actually copied
COPIED_COUNT=$(find "$AUDIO_DEST_DIR" -type f -name "*.mp3" | wc -l)
echo "✅ Copied $COPIED_COUNT audio files to scratch."

# -----------------------------
# 4. ENV VARS FOR TRAINING
# -----------------------------

# --- MODIFIED LINE ---
# Point the cache to your persistent AFS project directory
export HF_DATASETS_CACHE="/afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/hf_cache"
# ---------------------
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$HF_DATASETS_CACHE"
export WANDB_MODE=online

echo "Set HF_DATASETS_CACHE to: $HF_DATASETS_CACHE"
echo "Set WANDB_MODE to: $WANDB_MODE"

# -----------------------------
# 5. RUN TRAINING
# -----------------------------
echo "Starting training..."
python3 /afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/train/sge-eng+thai-mand/train.py \
    --scratch_dir "$DATA_DEST_DIR"

# -----------------------------
# 6. CLEANUP
# -----------------------------
echo "Cleaning up $DATA_DEST_DIR"
rm -rf "$DATA_DEST_DIR"
