import re
import os
import sys
import json
from collections import Counter
import pypdf
from config import *

def clean_text(text):
    '''
    Remove numbers, non-English characters, extra spaces, punctuation, everything except the 26 letters of the English alphabet and " "  . ,
    '''
    cleaned_text = re.sub(r'[\n\r\t]+', ' ', text)
    cleaned_text = re.sub(r'[^a-zA-Z., ]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.lower().strip()
    return cleaned_text

def process_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        cleaned_text = clean_text(text)
        return cleaned_text

def process_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    cleaned_text = clean_text(text)
    return cleaned_text

def process_raw_data(directory_path):
    corpus = {}
    for file in os.listdir(directory_path):
        print("Processing file ", file)
        file_path = os.path.join(directory_path, file)
        if file.endswith('.txt'):
            corpus[file] = process_txt(file_path)
        if file.endswith('.pdf'):
            corpus[file] = process_pdf(file_path)
    save_corpus(corpus)
    get_data_stats(corpus)

def get_data_stats(corpus):
    '''
    Get information about the training corpus, including total number of words and number of letters, size in bytes, number of times each character occurs.
    '''
    total_words = 0
    total_letters = 0
    char_counts = Counter()
    for file_name, text in corpus.items():
        words = len(text.split())
        print(f"Words from {file_name}: {words} words")
        total_words += words
        for char in text:
            char_counts[char] += 1
            if char.isalpha():
                total_letters += 1

    all_text = "".join(corpus.values())
    size_in_bytes = sys.getsizeof(all_text)

    stats = {
        "total_words": total_words,
        "total_letters": total_letters,
        "size_in_bytes": size_in_bytes,
        "character_frequencies": dict(char_counts.most_common())
    }
    for k,v in stats.items():
        print(k, v)

def save_corpus(corpus):
    with open(CORPUS, 'w') as f:
        json.dump(corpus, f, ensure_ascii=True, indent=2)

if __name__=='__main__':
    process_raw_data(DIRECTORY)