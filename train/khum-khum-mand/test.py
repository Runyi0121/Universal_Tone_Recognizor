import torch
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import evaluate
import os
from tqdm import tqdm
import librosa
import numpy as np
import re


# ====== CONFIG ======
MODEL_DIR = "/afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/train/khum-khum-mand/wav2vec2-thai-ctc/checkpoint-49000"
TEST_CSV = "/afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/train/khum-khum-mand/test.csv"
OUTPUT_CSV = "output.csv"
SAMPLE_RATE = 16000  # change if you trained with a different rate

# ====== LOAD MODEL + PROCESSOR ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ====== LOAD DATA ======
mandarin_audio_dir = "/afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/preprocessed_data/mandarin_data/audio"

df = pd.read_csv(TEST_CSV)

def speech_file_to_array_fn(path):
    speech, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        speech = transform(speech)
    return speech.squeeze().numpy()

for fname in df["file"]:
    full_path = os.path.join(mandarin_audio_dir, fname)
    print("DEBUG full path:", full_path)
    speech = speech_file_to_array_fn(full_path)

import os
import re
import pandas as pd
from tqdm import tqdm
import torch
import evaluate

# ====== SETUP ======
metric = evaluate.load("wer")  # or "cer" depending on your preference
records = []

def clean_text(text):
    """Remove special tokens like <unk>, <pad>, etc. and strip whitespace."""
    return re.sub(r"<.*?>", "", text).strip()

# ====== EVALUATION LOOP ======
for idx, row in tqdm(df.iterrows(), total=len(df)):
    path, transcription = row["file"], row["transcription"]
    speech = speech_file_to_array_fn(path)  # should be np.ndarray

    # Process input
    inputs = processor(
        speech,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Model forward
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode logits -> prediction string
    pred_ids = torch.argmax(logits, dim=-1)
    print("Pred IDs:", pred_ids)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    print("Pred string:", pred_str)

    # Clean up prediction and reference
    pred_str = clean_text(pred_str)  # can be empty string ''
    ref_str = clean_text(transcription)

    # Add to metric (empty predictions allowed)
    metric.add_batch(predictions=[pred_str], references=[ref_str])

    # Save record for later inspection
    records.append({
        "path": os.path.basename(path),
        "prediction": pred_str,
        "ground_truth": ref_str
    })

# ====== FINAL METRIC ======
final_score = metric.compute()
print("Test WER:", final_score)

# ====== SAVE OUTPUT ======
out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")


# # ====== METRIC ======
# metric = evaluate.load("accuracy")

# # ====== STORAGE ======
# records = []

# # ====== EVALUATION LOOP ======

# def clean_text(text):
#     # remove special tokens like <unk>, <pad>, etc.
#     return re.sub(r"<.*?>", "", text).strip()

# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     path, transcription = row["file"], row["transcription"]
#     speech = speech_file_to_array_fn(path)

#     inputs = processor(
#         speech,
#         sampling_rate=SAMPLE_RATE,
#         return_tensors="pt",
#         padding=True
#     ).to(device)

#     with torch.no_grad():
#         logits = model(**inputs).logits

#     # decode logits -> prediction string
#     pred_ids = torch.argmax(logits, dim=-1)
#     pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

#     # clean up
#     pred_str = clean_text(pred_str)
#     ref_str = clean_text(transcription)

#     if pred_str and ref_str:
#         metric.add_batch(predictions=[pred_str], references=[ref_str])
#     else:
#         print(f"Skipping empty pred/ref: pred='{pred_str}' ref='{ref_str}'")

#     # Add to metric + records
#     metric.add_batch(predictions=[pred_str], references=[ref_str])
#     records.append({
#         "path": os.path.basename(path),
#         "prediction": pred_str,
#         "ground_truth": transcription.strip()
#     })

# # ====== FINAL METRIC ======
# final_score = metric.compute()
# print("Test Accuracy:", final_score["accuracy"])

# # ====== SAVE OUTPUT ======
# out_df = pd.DataFrame(records)
# out_df.to_csv(OUTPUT_CSV, index=False)
# print(f"Predictions saved to {OUTPUT_CSV}")
