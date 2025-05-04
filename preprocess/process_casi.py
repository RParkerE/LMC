import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, List, Set

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'process_casi.log')),
            logging.StreamHandler()
        ]
    )

def load_casi_data(input_path: str) -> pd.DataFrame:
    """
    Load CASI dataset.
    
    Args:
        input_path: Path to CASI dataset CSV file
    
    Returns:
        DataFrame containing CASI data
    """
    return pd.read_csv(input_path)

def filter_casi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter CASI data according to paper's criteria.
    
    Args:
        df: Raw CASI DataFrame
    
    Returns:
        Filtered DataFrame
    """
    # Remove cases where short form equals long form
    df = df[df['Short Form'] != df['Long Form']]
    
    # Remove parsing issues (e.g., missing expansions)
    df = df.dropna(subset=['Short Form', 'Long Form', 'Context'])
    
    # Remove very short contexts
    df = df[df['Context'].str.len() > 10]
    
    # Remove cases with multiple expansions in context
    def has_multiple_expansions(row):
        context = row['Context'].lower()
        sf = row['Short Form'].lower()
        lf = row['Long Form'].lower()
        return context.count(sf) > 1 or context.count(lf) > 1
    
    df = df[~df.apply(has_multiple_expansions, axis=1)]
    
    return df

def create_examples(df: pd.DataFrame) -> List[Dict]:
    """
    Create training examples from filtered CASI data.
    
    Args:
        df: Filtered CASI DataFrame
    
    Returns:
        List of example dictionaries
    """
    examples = []
    
    for _, row in tqdm(df.iterrows(), desc="Creating examples"):
        # Get context words
        context = row['Context'].split()
        
        # Find position of short form
        sf_pos = -1
        for i, word in enumerate(context):
            if word.lower() == row['Short Form'].lower():
                sf_pos = i
                break
        
        if sf_pos == -1:
            continue
        
        # Create context window
        start = max(0, sf_pos - 5)
        end = min(len(context), sf_pos + 6)
        context_window = context[start:sf_pos] + context[sf_pos+1:end]
        
        if not context_window:
            continue
        
        # Create example
        example = {
            'acronym': row['Short Form'],
            'expansion': row['Long Form'],
            'context': context_window,
            'section_header': 'MAIN'  # CASI doesn't have section headers
        }
        examples.append(example)
    
    return examples

def split_data(examples: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
    """
    Split data into train/val/test sets.
    
    Args:
        examples: List of examples
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    
    Returns:
        Dictionary containing split datasets
    """
    # Shuffle examples
    import random
    random.shuffle(examples)
    
    # Calculate split indices
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': examples[:train_end],
        'val': examples[train_end:val_end],
        'test': examples[val_end:]
    }

def main(args):
    setup_logging(args.log_dir)
    logging.info(f"Starting CASI processing with args: {args}")
    
    # Load data
    logging.info("Loading CASI data...")
    df = load_casi_data(args.input_path)
    
    # Filter data
    logging.info("Filtering data...")
    df = filter_casi_data(df)
    logging.info(f"Filtered to {len(df)} examples")
    
    # Create examples
    logging.info("Creating examples...")
    examples = create_examples(df)
    logging.info(f"Created {len(examples)} examples")
    
    # Split data
    logging.info("Splitting data...")
    splits = split_data(examples)
    
    # Save processed data
    logging.info("Saving processed data...")
    os.makedirs(args.output_path, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = os.path.join(args.output_path, f'{split_name}.json')
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        logging.info(f"Saved {len(split_data)} examples to {split_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, required=True,
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