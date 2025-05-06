import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import logging
from models.lmc import LMC
from utils.data_utils import ClinicalDataset, collate_fn

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

def train_epoch(model, dataloader, optimizer, device, mask_prob=0.2):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # Move batch to device
        word_idx = batch['word_idx'].to(device)
        metadata_idx = batch['metadata_idx'].to(device)
        context_words = batch['context_words'].to(device)
        target_words = batch['target_words'].to(device)
        
        # Apply masking
        mask = torch.rand_like(word_idx.float()) < mask_prob
        word_idx = torch.where(mask, torch.zeros_like(word_idx), word_idx)
        
        # Forward pass
        outputs = model(word_idx, metadata_idx, context_words, target_words)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_recon_loss += outputs['reconstruction_loss'].item()
        total_kl_loss += outputs['kl_div'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon_loss': f"{outputs['reconstruction_loss'].item():.4f}",
            'kl_loss': f"{outputs['kl_div'].item():.4f}"
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'kl_loss': total_kl_loss / len(dataloader)
    }

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
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
    
    return {
        'loss': total_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'kl_loss': total_kl_loss / len(dataloader)
    }

def main(args):
    # Setup logging
    setup_logging(args.log_dir)
    logging.info("Starting training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create datasets
    logging.info("Loading datasets...")
    train_dataset = ClinicalDataset(
        args.data_dir,
        split='train',
        window_size=args.window_size
    )
    val_dataset = ClinicalDataset(
        args.data_dir,
        split='val',
        window_size=args.window_size
    )
    
    # Create data loaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Initialize model
    model = LMC(
        vocab_size=train_dataset.vocab_size,
        metadata_size=train_dataset.metadata_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.mask_prob
        )
        logging.info(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Recon Loss: {train_metrics['recon_loss']:.4f}, "
            f"KL Loss: {train_metrics['kl_loss']:.4f}"
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        logging.info(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Recon Loss: {val_metrics['recon_loss']:.4f}, "
            f"KL Loss: {val_metrics['kl_loss']:.4f}"
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))
            logging.info("Saved best model")
    
    logging.info("Training completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing processed data')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    parser.add_argument('--window_size', type=int, default=10,
                      help='Size of context window')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=100,
                      help='Dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Dimension of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=1000,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--mask_prob', type=float, default=0.2,
                      help='Probability of masking words')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args) 