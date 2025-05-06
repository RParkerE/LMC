import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import logging
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from models.lmc import LMC
from utils.data_utils import ClinicalDataset, collate_fn

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'evaluate.log')),
            logging.StreamHandler()
        ]
    )

def load_model(model_path, vocab_size, metadata_size, device):
    """Load a trained model."""
    # Initialize model
    model = LMC(
        vocab_size=vocab_size,
        metadata_size=metadata_size
    ).to(device)
    
    # Load state dictionary directly
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def evaluate_acronym_expansion(model, dataloader, device):
    """Evaluate model on reconstruction task."""
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move batch to device
            word_idx = batch['word_idx'].to(device)
            metadata_idx = batch['metadata_idx'].to(device)
            context_words = batch['context_words'].to(device)
            target_words = batch['target_words'].to(device)
            
            # Forward pass
            outputs = model(word_idx, metadata_idx, context_words, target_words)
            
            # Update statistics
            total_loss += outputs['loss'].item()
            total_recon_loss += outputs['reconstruction_loss'].item()
            total_kl_loss += outputs['kl_div'].item()
    
    # Calculate average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }
    
    return metrics

def main(args):
    # Setup logging
    setup_logging(args.log_dir)
    logging.info(f"Starting evaluation with args: {args}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load test data
    test_dataset = ClinicalDataset(args.test_path, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Load model
    model = load_model(
        args.model_path,
        test_dataset.vocab_size,
        test_dataset.metadata_size,
        device
    )
    
    # Evaluate
    metrics = evaluate_acronym_expansion(model, test_loader, device)
    
    # Log results
    logging.info("Evaluation results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--test_path', type=str, required=True,
                      help='Path to test data directory')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Directory to save evaluation results')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args) 