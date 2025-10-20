from transformers import pipeline
import os
import csv
print("Initializing the Thai Grapheme-to-Phoneme model...")

try:
    generator = pipeline("text2text-generation", model="pythainlp/thaig2p-v2.0", device=0)
    print("✅ Using Hugging Face model: pythainlp/thaig2p-v2.0 on CPU")
except Exception as e:
    print(f"❌ Failed to load G2P model: {e}")
    print("Try installing transformers with:")
    print("  python3 -m pip install --user transformers")
    exit(1)

# --- Tone mapping (Chao tone conversion) ---
chao_tone_map = {
    '1': '33',   # Mid Tone
    '2': '21',   # Low Tone
    '3': '51',   # Falling Tone
    '4': '55',   # High Tone
    '5': '214'   # Rising Tone
}

# --- Input/output paths ---
input_csv_path = 'metadata1.csv'
output_csv_path = 'metadata2_gpu.csv'


def convert_thai_to_chao_phonetics(text: str) -> str:
    """Convert Thai text to phonemes with Chao tone mapping."""
    if not text or not text.strip():
        return ""
    
    try:
        result = generator(text, max_new_tokens=128)
        phonemes = result[0]["generated_text"]
    except Exception as e:
        print(f"⚠️ Error processing '{text}': {e}")
        return ""
    
    # Apply Chao tone mapping if tone digits appear
    for tone_num, chao_tone in chao_tone_map.items():
        phonemes = phonemes.replace(tone_num, chao_tone)
    
    return phonemes


# --- Main CSV processing ---
if not os.path.exists(input_csv_path):
    print(f"Error: Input file '{input_csv_path}' not found.")
    exit(1)

with open(input_csv_path, 'r', encoding='utf-8') as infile, \
     open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader, None)
    if header:
        writer.writerow(header)
    
    print(f"Processing '{input_csv_path}'...")

    for i, row in enumerate(reader):
        if len(row) >= 3:
            thai_text = row[1]
            phonetic = convert_thai_to_chao_phonetics(thai_text)
            writer.writerow([row[0], phonetic, row[2]])

        if (i + 1) % 100 == 0:
            print(f"  ...processed {i + 1} rows.")

print(f"\n✅ Done! Output saved to '{output_csv_path}'")

