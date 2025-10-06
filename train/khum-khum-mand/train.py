import os
import json
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
    Trainer
)
import torchaudio
import torch
from sklearn.model_selection import train_test_split
import evaluate

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
# WandB init
# -----------------------
wandb.init(project="thai-tone-ctc")

# -----------------------
# Paths
# -----------------------
thai_csv = "/afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/preprocessed_data/prev-thai-dialect/khummuang/audio/metadata.csv"
thai_audio_dir = "/afs/crc.nd.edu/group/nlp/08/rshi2/master_tone/preprocessed_data/prev-thai-dialect/khummuang/audio"

thai_df = pd.read_csv(thai_csv)
thai_df["file"] = thai_df["file"].apply(lambda x: os.path.join(thai_audio_dir, x))

# -----------------------
# Train/Val split
# -----------------------
train_df, valid_df = train_test_split(thai_df, test_size=0.2, random_state=42)
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

# -----------------------
# Tokenizer + Processor
# -----------------------
# Collect characters for vocab
vocab_chars = set()
for text in thai_df["transcription"]:
    vocab_chars.update(list(text))

vocab_list = sorted(list(vocab_chars))
vocab_dict = {ch: i for i, ch in enumerate(vocab_list)}
vocab_dict["[PAD]"] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict) + 1
vocab_dict["|"] = len(vocab_dict) + 2

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False)

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

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
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16000
    return batch

train_ds = train_ds.map(speech_file_to_array_fn)
valid_ds = valid_ds.map(speech_file_to_array_fn)

def prepare_dataset(batch):
    batch["input_values"] = processor.feature_extractor(
        batch["speech"], sampling_rate=batch["sampling_rate"]
    ).input_values[0]
    # Explicit tokenizer call for labels
    tokenized = processor.tokenizer(
        batch["transcription"],
        add_special_tokens=False,
        return_attention_mask=False
    )
    batch["labels"] = tokenized["input_ids"]
    return batch

train_ds = train_ds.map(prepare_dataset, remove_columns=["file", "transcription", "speech", "sampling_rate"])
valid_ds = valid_ds.map(prepare_dataset, remove_columns=["file", "transcription", "speech", "sampling_rate"])

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
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    trust_remote_code=True,
    use_safetensors=True,
    vocab_size=len(processor.tokenizer.get_vocab()),
    pad_token_id=processor.tokenizer.pad_token_id,
    ignore_mismatched_sizes=True
)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.ctc_loss_reduction = "mean"
model.config.ctc_zero_infinity = True
print("✅ Model config pad_token_id:", model.config.pad_token_id)

# -----------------------
# Training args
# -----------------------
training_args = TrainingArguments(
    output_dir="./wav2vec2-thai-ctc",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=500,
    lr_scheduler_type="linear",
    num_train_epochs=40,
    save_total_limit=2,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    eval_steps=100,
    save_steps=500,
    report_to=["wandb"],
)

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

