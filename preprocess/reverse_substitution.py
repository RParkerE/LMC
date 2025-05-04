import argparse
import os
import json
import re
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'reverse_substitution.log')),
            logging.StreamHandler()
        ]
    )

def load_acronym_mapping(mapping_path: str) -> Dict[str, List[str]]:
    """
    Load acronym mapping from CASI dataset.
    
    Args:
        mapping_path: Path to CASI dataset CSV file
    
    Returns:
        Dictionary mapping short forms to possible long forms
    """
    df = pd.read_csv(mapping_path)
    mapping = defaultdict(list)
    
    for _, row in df.iterrows():
        sf = row['Short Form'].lower()
        lf = row['Long Form'].lower()
        if sf not in mapping[sf]:
            mapping[sf].append(lf)
    
    return dict(mapping)

def find_acronym_occurrences(text: str, acronym: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of an acronym in text.
    
    Args:
        text: Input text
        acronym: Acronym to find
    
    Returns:
        List of (start, end) positions
    """
    pattern = r'\b' + re.escape(acronym) + r'\b'
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]

def replace_acronym(text: str, acronym: str, expansion: str) -> str:
    """
    Replace an acronym with its expansion in text.
    
    Args:
        text: Input text
        acronym: Acronym to replace
        expansion: Expansion to insert
    
    Returns:
        Text with acronym replaced
    """
    pattern = r'\b' + re.escape(acronym) + r'\b'
    return re.sub(pattern, expansion, text, flags=re.IGNORECASE)

def process_notes(input_path: str, acronym_mapping: Dict[str, List[str]], output_path: str):
    """
    Process notes and generate synthetic data via reverse substitution.
    
    Args:
        input_path: Path to MIMIC-III NOTEEVENTS.csv
        acronym_mapping: Dictionary mapping short forms to possible long forms
        output_path: Path to save processed data
    """
    # Read notes
    logging.info("Reading MIMIC-III notes...")
    notes_df = pd.read_csv(input_path)
    
    # Process each note
    processed_data = []
    for _, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Processing notes"):
        note_id = row['ROW_ID']
        text = row['TEXT']
        note_type = row['CATEGORY']
        
        # Find all acronyms in text
        for sf, expansions in acronym_mapping.items():
            occurrences = find_acronym_occurrences(text, sf)
            
            for start, end in occurrences:
                # Get context window
                context_start = max(0, start - 100)
                context_end = min(len(text), end + 100)
                context = text[context_start:context_end]
                
                # Replace acronym with each possible expansion
                for expansion in expansions:
                    # Create synthetic example
                    synthetic_text = replace_acronym(context, sf, expansion)
                    
                    # Split into sentences
                    sentences = re.split(r'[.!?]+', synthetic_text)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    # Find sentence containing the expansion
                    for sentence in sentences:
                        if expansion.lower() in sentence.lower():
                            # Create example
                            example = {
                                'note_id': note_id,
                                'note_type': note_type,
                                'acronym': sf,
                                'expansion': expansion,
                                'context': sentence.split(),
                                'section_header': 'MAIN'  # We don't have section headers in synthetic data
                            }
                            processed_data.append(example)
                            break
    
    # Save processed data
    logging.info("Saving processed data...")
    output_file = os.path.join(output_path, 'synthetic_data.json')
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logging.info(f"Generated {len(processed_data)} synthetic examples")

def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
    """
    Split data into train/val/test sets.
    
    Args:
        data: List of examples
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        Dictionary containing split datasets
    """
    # Shuffle data
    import random
    random.shuffle(data)
    
    # Calculate split indices
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }

def main(args):
    setup_logging(args.log_dir)
    logging.info(f"Starting reverse substitution with args: {args}")
    
    # Load acronym mapping
    logging.info("Loading acronym mapping...")
    acronym_mapping = load_acronym_mapping(args.mapping_path)
    logging.info(f"Loaded {len(acronym_mapping)} acronym mappings")
    
    # Process notes
    process_notes(args.input_path, acronym_mapping, args.output_path)
    
    # Load processed data
    with open(os.path.join(args.output_path, 'synthetic_data.json'), 'r') as f:
        data = json.load(f)
    
    # Split data
    logging.info("Splitting data...")
    splits = split_data(data)
    
    # Save splits
    for split_name, split_data in splits.items():
        output_file = os.path.join(args.output_path, f'{split_name}.json')
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        logging.info(f"Saved {len(split_data)} examples to {split_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to MIMIC-III NOTEEVENTS.csv')
    parser.add_argument('--mapping_path', type=str, required=True,
                      help='Path to CASI dataset CSV file')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Directory to save processed data')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args) 