import os
import json
from pathlib import Path
import wandb
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    Wav2Vec2Config
)
import torchaudio
import torch
from sklearn.model_selection import train_test_split
import evaluate
import panphon
import unicodedata
import torch, gc
import argparse
from datasets import load_dataset
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import librosa

# ---------- top-of-script safety boilerplate ----------
import os, multiprocessing, gc
# safer multiprocessing start method
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

# limit host threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# prefer scratch for temp and dataset cache (edit path if needed)
os.environ.setdefault("TMPDIR", f"/scratch/{os.getenv('USER')}/tmp")
os.environ.setdefault("HF_DATASETS_CACHE", f"/scratch/{os.getenv('USER')}/hf_cache")

# import torch after setting sharing strategy
import torch
torch.multiprocessing.set_sharing_strategy("file_system")

# optional: small helper to free memory in places you call it
def free_mem():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
# ----------------------------------------------------


# Collect unused memory from Python garbage
gc.collect()

# Empty the PyTorch CUDA cache
torch.cuda.empty_cache()

# Optional: display current usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
# -----------------------
# IPA Normalization Setup
# -----------------------
ft = panphon.FeatureTable()

def normalize_ipa(s):
    s = unicodedata.normalize("NFC", s)
    segs = ft.segs_safe(s)
    return "".join(segs)

wer_metric = evaluate.load("wer")

# -----------------------
# Device
# -----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

# -----------------------
# SCRIPT ARGUMENTS
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--scratch_dir", 
    type=str, 
    required=True, 
    help="Path to the scratch directory containing metadata.csv and audio_root/"
)
args = parser.parse_args()
print(f"Using scratch directory: {args.scratch_dir}")

# This will be the root for the copied audio files
# e.g., /tmp/.../my_dataset/audio_root
scratch_audio_root = os.path.join(args.scratch_dir, "audio_root")


# -----------------------
# WandB init
# -----------------------
wandb.init(project="thai-tone-ctc")

# -----------------------
# Paths
# -----------------------
# Use the new scratch path to read the CSV
thai_csv_path = os.path.join(args.scratch_dir, "metadata.csv")

# -----------------------
# 2. NEW: Fix file paths to point to scratch
# -----------------------
def fix_path_to_scratch(batch):
    # This function will now be used inside .map()
    # original_absolute_path is like "/afs/crc.nd.edu/..."
    original_path_stripped = batch["file"].strip().lstrip('/')
    
    # This creates the new path:
    # /tmp/.../my_dataset/audio_root/afs/crc.nd.edu/...
    batch["file"] = os.path.join(scratch_audio_root, original_path_stripped)
    return batch

print("Loading dataset directly from CSV (this is memory-efficient)...")
# Load the CSV directly using datasets. This is a lazy-load.
# It will stream the file, not load it all into memory.
full_ds = load_dataset("csv", data_files=thai_csv_path, split="train")

# 2. --- FIX: Select your 10,000 file subset FIRST ---
print(f"Original dataset size: {len(full_ds)}. Subsetting to 300,000 files.")
full_ds_subset = full_ds.select(range(15000))
# --------------------------------------------------

print("Updating file paths to point to scratch directory (on 10k subset)...")
# 3. Now run the first map on the SMALL subset
full_ds = full_ds_subset.map(fix_path_to_scratch, num_proc=1)

#print("Updating file paths to point to scratch directory...")
# Run the path fixing function using .map()
# This is also processed in batches and is memory-efficient.
#full_ds = full_ds.map(fix_path_to_scratch, num_proc=2) # Use num_proc for speed

# -----------------------
# Create vocab.json from train
# -----------------------
print("Creating vocab from dataset...")
all_lines = []
# You can iterate over the dataset object directly
for text in full_ds["transcription"]:
    if text: # Add a check for empty transcriptions
        text = normalize_ipa(text)
        all_lines.append(text)

vocab_set = set()
for line in all_lines:
    vocab_set.update(list(line))
vocab_set.add("|")

vocab_list = sorted(list(vocab_set))
vocab_dict = {c: i for i, c in enumerate(vocab_list)}
vocab_dict["[PAD]"] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict) + 1

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print("✅ Total vocab size:", len(vocab_dict))
print("\n🧩 Vocabulary Tokens:")
for token, idx in vocab_dict.items():
    print(f"{idx:3d}: {repr(token)}")

# -----------------------
# 🪄 3. Initialize a new tokenizer + processor and save
# -----------------------
tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)
tokenizer.save_pretrained("./ipa_tokenizer")

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)
processor.save_pretrained("./ipa_tokenizer")

print("✅ Saved tokenizer and processor to ./ipa_tokenizer")

# -----------------------
# Train/Val split
# -----------------------
print("Splitting dataset...")
# Now, split the main dataset object. This is also memory-efficient.
ds_splits = full_ds.train_test_split(test_size=0.2, seed=42)
train_ds = ds_splits["train"]
valid_ds = ds_splits["test"]

# -----------------------
# Ensure pad token exists BEFORE dataset.map
# -----------------------
if processor.tokenizer.pad_token is None:
    if getattr(processor.tokenizer, "eos_token", None) is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    elif getattr(processor.tokenizer, "unk_token", None) is not None:
        processor.tokenizer.pad_token = processor.tokenizer.unk_token
    else:
        processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
pad_id = processor.tokenizer.pad_token_id
print("✅ Pad token ID:", pad_id)

# -----------------------
# Preprocessing
# -----------------------
missing_log = open("missing_files.log", "a")
def log_missing(path, err):
    missing_log.write(f"{path}\t{err}\n")
    missing_log.flush()

def prepare_dataset(batch):
    
    # 1. Load and resample audio using Librosa
    speech_arrays = []
    target_sr = 16000

    for file_path in batch["file"]:
        try:
            # librosa.load does all the work:
            # - Loads the MP3 (using ffmpeg)
            # - Resamples to target_sr (16000)
            # - Converts to mono
            speech_array, _ = librosa.load(file_path, sr=target_sr, mono=True)
            speech_arrays.append(speech_array)

        except Exception as e:
            # If loading fails, log it and append a silent array to keep batches aligned
            print(f"⚠️ Skipping {file_path}: {e}")
            log_missing(file_path, e)
            speech_arrays.append(np.array([0.0], dtype=np.float32)) # Add silent dummy array
    
    # 2. Process audio (feature extraction)
    # The feature extractor works perfectly with a list of numpy arrays.
    batch["input_values"] = processor.feature_extractor(
        speech_arrays, sampling_rate=target_sr
    ).input_values

    # 3. Process text (tokenization)
    normalized_transcriptions = [normalize_ipa(text) for text in batch["transcription"]]
    
    tokenized = processor.tokenizer(
        normalized_transcriptions,
        add_special_tokens=False,
        return_attention_mask=False
    )
    batch["labels"] = tokenized["input_ids"]
    return batch

train_ds = train_ds.map(
    prepare_dataset,
    batched=True,
    batch_size=8,
    writer_batch_size=5,
    num_proc=1,
    load_from_cache_file=False,
)

valid_ds = valid_ds.map(
    prepare_dataset,
    batched=True,
    batch_size=8,
    writer_batch_size=5,
    num_proc=1,
    load_from_cache_file=False,
)
missing_log.close()
# -----------------------
# Data collator
# -----------------------
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True, debug_first_n=1):
        self.processor = processor
        self.padding = padding
        self.debug_first_n = debug_first_n
        self._debug_count = 0

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"]
        labels_attention_mask = labels_batch.get("attention_mask", (labels != self.processor.tokenizer.pad_token_id).long())
        labels = labels.masked_fill(labels_attention_mask == 0, -100)

        batch["labels"] = labels

        if self._debug_count < self.debug_first_n:
            num_all_minus100 = (labels == -100).sum().item()
            total_label_tokens = labels.numel()
            print(f"***DEBUG*** labels -100 count: {num_all_minus100}/{total_label_tokens}")
            if num_all_minus100 == total_label_tokens:
                raise ValueError("All label tokens are -100 — check pad token setup.")
            self._debug_count += 1

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# -----------------------
# Model
# -----------------------
# -----------------------
# Model
# -----------------------

# Get special token IDs
pad_token_id = processor.tokenizer.pad_token_id
vocab_size = len(processor.tokenizer.get_vocab())

print(f"✅ Pad token ID: {pad_token_id}")
print(f"✅ Vocab size: {vocab_size}")
print(f"✅ Word delimiter ID: {processor.tokenizer.word_delimiter_token_id}")

# 1. Load the base model configuration
config = Wav2Vec2Config.from_pretrained(
    "facebook/wav2vec2-base",
    trust_remote_code=True,
)

# 2. Update the config with YOUR custom settings
config.update({
    "vocab_size": vocab_size,
    "pad_token_id": pad_token_id,
    # ------------------- THE FIX -------------------
    # Use the PAD token ID as the CTC blank token ID
    "ctc_blank_token": pad_token_id, 
    # -----------------------------------------------
    "ctc_loss_reduction": "mean",
    "ctc_zero_infinity": True,
})

# 3. Load the model using the modified config
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    trust_remote_code=True,
    use_safetensors=True,
    config=config,# <-- Pass the updated config object here
    ignore_mismatched_sizes=True
)

model.freeze_feature_extractor()
print("✅ Model loaded with custom config.")

# -----------------------
# Training args
# -----------------------
training_args = TrainingArguments(
    output_dir="./model_15k",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_strategy="epoch",
    learning_rate=1e-4,
    warmup_steps=500,
    lr_scheduler_type="linear",
    num_train_epochs=20,
    save_total_limit=2,
    fp16=False,
    logging_dir="./logs",
    logging_steps=50,
    eval_steps=100,
    save_steps=200,
    report_to=["wandb"],
    dataloader_num_workers=0,
)

# -----------------------
# Metrics
# -----------------------
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=processor,
)

# -----------------------
# Sanity forward pass before training
# -----------------------
examples = [train_ds[i] for i in range(2)]
batch = data_collator(examples)
batch = {k: v.to(model.device) for k, v in batch.items()}
model.to(model.device)
model.train()
out = model(**batch)
print("Sanity forward pass loss:", out.loss.item() if out.loss is not None else "no loss")

# -----------------------
# Train
# -----------------------
trainer.train()
