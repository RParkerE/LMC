import os
from utils.data_utils import save_vocab_and_metadata
import json
import glob
from collections import Counter
import tqdm

def process_casi_file(file_path, word_counts, metadata_set):
    """Process a CASI file and update word counts and metadata set."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        for example in data:
            # Count acronyms and expansions
            acronym = example['acronym']
            expansion = example['expansion']
            word_counts[acronym] += 1
            
            # Split expansion into words and count each
            for word in expansion.split():
                word_counts[word] += 1
            
            # Count context words
            for word in example['context']:
                word_counts[word] += 1
            
            # Collect metadata (section headers)
            metadata_set.add(example['section_header'])

def process_mimic_file(file_path, word_counts, metadata_set):
    """Process a MIMIC-III chunk file and update word counts and metadata set."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        for example in data:
            # Process text and metadata based on MIMIC format
            if 'text' in example:
                # Split text into words and count each
                for word in example['text'].split():
                    word_counts[word] += 1
            
            # Collect metadata (section headers or note types)
            if 'section_header' in example:
                metadata_set.add(example['section_header'])
            elif 'note_type' in example:
                metadata_set.add(example['note_type'])

def main():
    data_dir = 'data/processed'
    output_dir = data_dir
    
    # Initialize counters
    word_counts = Counter()
    metadata_set = set()
    
    # Process CASI data
    casi_dir = os.path.join(data_dir, 'casi')
    if os.path.exists(casi_dir):
        print("Processing CASI data...")
        for split in ['train', 'test', 'val']:
            file_path = os.path.join(casi_dir, f'{split}.json')
            if os.path.exists(file_path):
                print(f"Processing {split}.json...")
                process_casi_file(file_path, word_counts, metadata_set)
    
    # Process MIMIC-III chunks
    mimic_dir = os.path.join(data_dir, 'sections')
    if os.path.exists(mimic_dir):
        print("\nProcessing MIMIC-III chunks...")
        chunk_files = sorted(glob.glob(os.path.join(mimic_dir, 'processed_notes_chunk_*.json')))
        for chunk_file in tqdm.tqdm(chunk_files):
            try:
                process_mimic_file(chunk_file, word_counts, metadata_set)
            except Exception as e:
                print(f"\nError processing {chunk_file}: {str(e)}")
                # Print first few lines of the file to debug
                with open(chunk_file, 'r') as f:
                    print("First few lines of file:")
                    for i, line in enumerate(f):
                        if i < 5:  # Print first 5 lines
                            print(line.strip())
                        else:
                            break
                continue
    
    print("\nCreating vocabulary and metadata...")
    # Create vocabulary
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3
    }
    
    # Add words that meet frequency threshold
    min_freq = 5
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    # Create metadata mapping
    metadata = {'<UNK>': 0}
    for meta in sorted(metadata_set):
        metadata[meta] = len(metadata)
    
    # Save the mappings
    save_vocab_and_metadata(vocab, metadata, output_dir)
    print(f"\nCreated vocabulary with {len(vocab)} words")
    print(f"Created metadata mapping with {len(metadata)} categories")
    
    # Print some statistics
    print("\nTop 10 most frequent words:")
    for word, count in word_counts.most_common(10):
        print(f"{word}: {count}")
    
    print("\nTop 10 most frequent acronyms:")
    acronym_counts = {word: count for word, count in word_counts.items() if word.isupper() and len(word) >= 2}
    for acronym, count in sorted(acronym_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{acronym}: {count}")

if __name__ == '__main__':
    main() 