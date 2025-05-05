import argparse
import os
import json
import re
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'extract_sections.log')),
            logging.StreamHandler()
        ]
    )

def extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract sections from clinical note text.
    
    Args:
        text: Raw note text
    
    Returns:
        List of (section_header, section_text) tuples
    """
    if not isinstance(text, str) or not text.strip():
        return []
        
    # Common section headers in clinical notes
    section_patterns = [
        r'(?:^|\n)([A-Z][A-Za-z\s]+):',
        r'(?:^|\n)([A-Z][A-Za-z\s]+)\n',
        r'(?:^|\n)([A-Z][A-Za-z\s]+)\s*[-–—]',
    ]
    
    # Combine patterns with case-insensitive flag
    pattern = '(?i)' + '|'.join(f'({p})' for p in section_patterns)
    
    # Find all section headers
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        # If no sections found, treat entire text as one section
        return [('MAIN', text.strip())]
    
    # Extract sections
    sections = []
    for i in range(len(matches)):
        try:
            start = matches[i].start()
            end = matches[i+1].start() if i < len(matches)-1 else len(text)
            
            # Get section header
            header = matches[i].group(1).strip() if matches[i].group(1) else None
            if not header:
                # Try other capture groups
                for j in range(2, len(matches[i].groups())+1):
                    if matches[i].group(j):
                        header = matches[i].group(j).strip()
                        break
            
            if not header:
                continue
                
            # Get section text
            section_text = text[start:end].strip()
            
            # Remove header from text
            section_text = re.sub(f'^{re.escape(header)}[:\\s-]*', '', section_text).strip()
            
            if section_text:  # Only add non-empty sections
                sections.append((header, section_text))
        except Exception as e:
            logging.warning(f"Error extracting section: {str(e)}")
            continue
    
    return sections

def process_mimic_notes(input_path: str, output_path: str):
    """
    Process MIMIC-III notes and extract sections.
    
    Args:
        input_path: Path to MIMIC-III NOTEEVENTS.csv
        output_path: Path to save processed data
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Read notes with explicit dtype specification
    logging.info("Reading MIMIC-III notes...")
    notes_df = pd.read_csv(
        input_path, 
        encoding='utf-8',
        low_memory=False,
        dtype={
            'ROW_ID': 'Int64',
            'SUBJECT_ID': 'Int64',
            'HADM_ID': 'Int64',
            'CHARTDATE': 'str',
            'CHARTTIME': 'str',
            'STORETIME': 'str',
            'CATEGORY': 'str',
            'DESCRIPTION': 'str',
            'CGID': 'str',
            'ISERROR': 'str',
            'TEXT': 'str'
        }
    )
    
    # Drop rows where TEXT or CATEGORY is null
    notes_df = notes_df.dropna(subset=['TEXT', 'CATEGORY'])
    
    # Process each note
    processed_data = []
    for _, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Processing notes"):
        try:
            note_id = row['ROW_ID']
            text = row['TEXT']
            note_type = row['CATEGORY']
            
            # Skip empty notes
            if not isinstance(text, str) or not text.strip():
                continue
            
            # Extract sections
            sections = extract_sections(text)
            if not sections:
                continue
            
            # Create examples for each section
            for header, section_text in sections:
                try:
                    # Skip empty sections
                    if not section_text or not isinstance(section_text, str) or not section_text.strip():
                        continue
                    
                    # Split into sentences
                    sentences = re.split(r'[.!?]+', section_text)
                    sentences = [s.strip() for s in sentences if s and isinstance(s, str) and s.strip()]
                    
                    # Create examples
                    for i, sentence in enumerate(sentences):
                        words = sentence.split()
                        if len(words) < 3:  # Skip very short sentences
                            continue
                        
                        # Create context windows
                        for j in range(len(words)):
                            # Get context window
                            start = max(0, j - 5)
                            end = min(len(words), j + 6)
                            context = words[start:j] + words[j+1:end]
                            
                            if not context:  # Skip if no context
                                continue
                            
                            # Create example
                            example = {
                                'note_id': note_id,
                                'note_type': note_type,
                                'section_header': header,
                                'word': words[j],
                                'context': context
                            }
                            processed_data.append(example)
                except Exception as e:
                    logging.warning(f"Error processing section in note {note_id}: {str(e)}")
                    continue
        except Exception as e:
            logging.warning(f"Error processing note {note_id}: {str(e)}")
            continue
    
    # Save processed data
    logging.info("Saving processed data...")
    output_file = os.path.join(output_path, 'processed_notes.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    logging.info(f"Processed {len(processed_data)} examples")

def main(args):
    setup_logging(args.log_dir)
    logging.info(f"Starting section extraction with args: {args}")
    
    process_mimic_notes(args.input_path, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to MIMIC-III NOTEEVENTS.csv')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Directory to save processed data')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args) 