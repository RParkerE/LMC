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
    Load CASI dataset from text file.
    
    Args:
        input_path: Path to CASI dataset text file
    
    Returns:
        DataFrame containing CASI data
    """
    data = []
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(input_path, 'r', encoding=encoding) as f:
                for line in f:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Split line into components using pipe separator
                    parts = line.strip().split('|')
                    if len(parts) >= 6:  # Expecting at least 6 fields based on the format
                        # Extract short form, long form, and context
                        short_form = parts[0].strip()
                        long_form = parts[1].strip()
                        # Combine remaining parts for context, skipping empty fields
                        context = ' '.join([p.strip() for p in parts[5:] if p.strip()])
                        
                        if short_form and long_form and context:
                            data.append({
                                'Short Form': short_form,
                                'Long Form': long_form,
                                'Context': context
                            })
                        else:
                            logging.warning(f"Skipping line with missing required fields: {line.strip()}")
                    else:
                        logging.warning(f"Skipping malformed line: {line.strip()}")
                # If we get here without error, we found the right encoding
                logging.info(f"Successfully read file with encoding: {encoding}")
                break
        except UnicodeDecodeError:
            logging.warning(f"Failed to read with encoding: {encoding}")
            continue
    
    if not data:
        raise ValueError(f"Could not read file with any of the attempted encodings: {encodings}")
    
    logging.info(f"Loaded {len(data)} examples from CASI dataset")
    return pd.DataFrame(data)

def filter_casi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter CASI data according to paper's criteria.
    
    Args:
        df: Raw CASI DataFrame
    
    Returns:
        Filtered DataFrame
    """
    print(f"Initial number of examples: {len(df)}")
    
    # Remove cases where short form equals long form
    df = df[df['Short Form'] != df['Long Form']]
    print(f"After removing cases where short form equals long form: {len(df)}")
    
    # Remove parsing issues (e.g., missing expansions)
    df = df.dropna(subset=['Short Form', 'Long Form', 'Context'])
    print(f"After removing missing values: {len(df)}")
    
    # Remove very short contexts
    df = df[df['Context'].str.len() > 10]
    print(f"After removing short contexts: {len(df)}")
    
    # Remove cases with multiple expansions in context
    def has_multiple_expansions(row):
        context = row['Context'].lower()
        sf = row['Short Form'].lower()
        lf = row['Long Form'].lower()
        return context.count(sf) > 1 or context.count(lf) > 1
    
    df = df[~df.apply(has_multiple_expansions, axis=1)]
    print(f"After removing multiple expansions: {len(df)}")
    
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
    
    for idx, row in tqdm(df.iterrows(), desc="Creating examples", total=len(df)):
        try:
            # Get context words
            context = row['Context'].split()
            
            # Find position of short form, handling punctuation and word boundaries
            sf_pos = -1
            sf = row['Short Form'].lower()
            
            # Handle special characters in short form
            sf_clean = ''.join(c.lower() for c in sf if c.isalnum() or c in '&')
            
            for i, word in enumerate(context):
                # Remove punctuation but preserve special characters like '&'
                clean_word = ''.join(c.lower() for c in word if c.isalnum() or c in '&')
                
                # Check if the short form appears as a standalone word or as part of a compound
                if (clean_word == sf_clean or  # Exact match
                    (len(clean_word) > len(sf_clean) and  # Part of a longer word
                     (clean_word.startswith(sf_clean) or  # At the start
                      clean_word.endswith(sf_clean) or    # At the end
                      f"_{sf_clean}_" in f"_{clean_word}_"))):  # In the middle
                    sf_pos = i
                    break
            
            if sf_pos == -1:
                logging.warning(f"Could not find short form '{row['Short Form']}' in context: {row['Context']}")
                continue
            
            # Create context window
            start = max(0, sf_pos - 5)
            end = min(len(context), sf_pos + 6)
            context_window = context[start:sf_pos] + context[sf_pos+1:end]
            
            if not context_window:
                logging.warning(f"Empty context window for short form '{row['Short Form']}'")
                continue
            
            # Create example
            example = {
                'acronym': row['Short Form'],
                'expansion': row['Long Form'],
                'context': context_window,
                'section_header': 'MAIN'  # CASI doesn't have section headers
            }
            examples.append(example)
        except Exception as e:
            logging.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    logging.info(f"Created {len(examples)} examples")
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
    if not examples:
        raise ValueError("No examples provided for splitting")
    
    # Shuffle examples
    import random
    random.shuffle(examples)
    
    # Calculate split indices
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': examples[:train_end],
        'val': examples[train_end:val_end],
        'test': examples[val_end:]
    }
    
    logging.info(f"Split data into {len(splits['train'])} train, {len(splits['val'])} val, and {len(splits['test'])} test examples")
    return splits

def main(args):
    setup_logging(args.log_dir)
    logging.info(f"Starting CASI processing with args: {args}")
    
    try:
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
        
        if not examples:
            raise ValueError("No valid examples were created from the data")
        
        # Split data
        logging.info("Splitting data...")
        splits = split_data(examples)
        
        # Save processed data
        logging.info("Saving processed data...")
        os.makedirs(args.output_path, exist_ok=True)
        
        for split_name, split_examples in splits.items():
            output_file = os.path.join(args.output_path, f'{split_name}.json')
            with open(output_file, 'w') as f:
                json.dump(split_examples, f, indent=2)
            logging.info(f"Saved {len(split_examples)} examples to {split_name}")
            
    except Exception as e:
        logging.error(f"Error processing CASI data: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to CASI dataset text file')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Directory to save processed data')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args) 