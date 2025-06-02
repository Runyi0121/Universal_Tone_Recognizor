# csv file
# combine dev and train
# cat dev.csv train.csv > metadata.csv

# remove the first and last column, swap second third column
# import csv

# with open('metadata_raw.csv', 'r', newline='') as infile, open('metadata.csv', 'w', newline='') as outfile:
#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)

#     for row in reader:
#         if len(row) > 4:
#             # Remove first, second, and last columns
#             trimmed = row[2:-1]

#             # Swap the third and fourth columns in the trimmed row (i.e., index 2 and 3)
#             if len(trimmed) > 3:
#                 trimmed[2], trimmed[3] = trimmed[3], trimmed[2]

#             writer.writerow(trimmed)
#         else:
#             writer.writerow([])  # Or skip: use `continue` instead if preferred

import re

# Set of all Thai consonants (used to find initial consonant)
THAI_CONSONANTS = LOW_CLASS | MID_CLASS | HIGH_CLASS

def extract_initial_consonant(syllable):
    # Return the first consonant character that appears in the syllable
    for ch in syllable:
        if ch in THAI_CONSONANTS:
            return ch
    return None

def classify_tone(syllable):
    if not syllable:
        return ''

    initial = extract_initial_consonant(syllable)
    tone_mark = extract_tone_mark(syllable)
    consonant_class = get_consonant_class(initial)

    is_live = True
    is_short = False

    tone = None

    if tone_mark == '่':
        if consonant_class == 'mid':
            tone = 'low'
        elif consonant_class == 'low':
            tone = 'falling'
        elif consonant_class == 'high':
            tone = 'low'
    elif tone_mark == '้':
        if consonant_class == 'mid':
            tone = 'falling'
        elif consonant_class == 'low':
            tone = 'high'
        elif consonant_class == 'high':
            tone = 'falling'
    elif tone_mark == '๊':
        tone = 'high'
    elif tone_mark == '๋':
        tone = 'rising'
    else:
        if consonant_class == 'mid':
            tone = 'mid' if is_live else 'low'
        elif consonant_class == 'low':
            tone = 'mid' if is_live else 'falling' if is_short else 'high'
        elif consonant_class == 'high':
            tone = 'rising' if is_live else 'low'

    if tone is None:
        print(f"[WARN] No tone assigned for syllable: '{syllable}' (initial: '{initial}', tone mark: '{tone_mark}', class: '{consonant_class}')")
        tone = 'mid'

    return TONE_TO_CHAO.get(tone, '??')

import csv
import re
from pythainlp.transliterate import romanize

def separate_word_and_tone(token):
    # Match Thai word followed by superscript digits (e.g., เพื่อน³³)
    match = re.match(r"(.+?)([¹²³⁴⁵⁰⁶⁷⁸⁹]+)$", token)
    if match:
        return match.group(1), match.group(2)
    else:
        return token, ''  # fallback: return token as-is

with open('metadata_tone_inserted.csv', 'r', newline='') as infile, open('metadata_romanized.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if not row:
            writer.writerow([])
            continue

        annotated_text = row[0]  # first column: tone-annotated Thai
        tokens = annotated_text.strip().split()

        romanized_tokens = []
        for token in tokens:
            thai_word, tone = separate_word_and_tone(token)
            roman = romanize(thai_word)
            romanized_tokens.append(roman + tone)

        romanized_text = ' '.join(romanized_tokens)
        writer.writerow([romanized_text] + row[1:])


import csv

# Read from the romanized file and write to the final metadata.csv
with open('metadata_romanized.csv', 'r', newline='') as infile, open('metadata.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if not row or len(row) < 3:
            writer.writerow(row)
            continue

        # Swap first and third columns
        row[0], row[2] = row[2], row[0]

        # Remove the second column (now at index 1)
        new_row = [row[0], row[2]] if len(row) > 2 else row

        # Remove "dev_audio00/" and "train_audio00/" from the first column
        new_row[0] = new_row[0].replace("dev_audio00/", "").replace("train_audio00/", "")

        writer.writerow(new_row)
