import argparse
import os
import subprocess
import logging
from pathlib import Path

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
            logging.StreamHandler()
        ]
    )

def run_command(command, description):
    """Run a shell command and log its output."""
    logging.info(f"Running {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(f"{description} completed successfully")
        if result.stdout:
            logging.info(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with error:\n{e.stderr}")
        raise

def main(args):
    setup_logging(args.log_dir)
    logging.info("Starting LMC pipeline")
    
    # Create necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Step 1: Process MIMIC-III notes
    logging.info("Step 1: Processing MIMIC-III notes")
    run_command(
        f"python preprocess/extract_sections.py "
        f"--input_path {args.mimic_path} "
        f"--output_path {os.path.join(args.data_dir, 'mimic')} "
        f"--log_dir {args.log_dir}",
        "MIMIC-III processing"
    )
    
    # Step 2: Process CASI dataset
    logging.info("Step 2: Processing CASI dataset")
    run_command(
        f"python preprocess/process_casi.py "
        f"--input_path {args.casi_path} "
        f"--output_path {os.path.join(args.data_dir, 'casi')} "
        f"--log_dir {args.log_dir}",
        "CASI processing"
    )
    
    # Step 3: Generate synthetic data
    logging.info("Step 3: Generating synthetic data")
    run_command(
        f"python preprocess/reverse_substitution.py "
        f"--input_path {args.mimic_path} "
        f"--mapping_path {args.casi_path} "
        f"--output_path {os.path.join(args.data_dir, 'synthetic')} "
        f"--log_dir {args.log_dir}",
        "Synthetic data generation"
    )
    
    # Step 4: Train LMC model
    logging.info("Step 4: Training LMC model")
    run_command(
        f"python train.py "
        f"--data_path {os.path.join(args.data_dir, 'mimic')} "
        f"--model_dir {args.model_dir} "
        f"--log_dir {args.log_dir} "
        f"--embedding_dim {args.embedding_dim} "
        f"--hidden_dim {args.hidden_dim} "
        f"--dropout {args.dropout} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.learning_rate} "
        f"--epochs {args.epochs} "
        f"--mask_prob {args.mask_prob} "
        f"--num_workers {args.num_workers}",
        "Model training"
    )
    
    # Step 5: Evaluate on test sets
    logging.info("Step 5: Evaluating model")
    
    # Evaluate on CASI
    run_command(
        f"python evaluate.py "
        f"--test_path {os.path.join(args.data_dir, 'casi')} "
        f"--model_path {os.path.join(args.model_dir, 'best_model.pt')} "
        f"--output_path {os.path.join(args.model_dir, 'results_casi')} "
        f"--log_dir {args.log_dir} "
        f"--batch_size {args.batch_size} "
        f"--num_workers {args.num_workers}",
        "CASI evaluation"
    )
    
    # Evaluate on synthetic data
    run_command(
        f"python evaluate.py "
        f"--test_path {os.path.join(args.data_dir, 'synthetic')} "
        f"--model_path {os.path.join(args.model_dir, 'best_model.pt')} "
        f"--output_path {os.path.join(args.model_dir, 'results_synthetic')} "
        f"--log_dir {args.log_dir} "
        f"--batch_size {args.batch_size} "
        f"--num_workers {args.num_workers}",
        "Synthetic data evaluation"
    )
    
    logging.info("Pipeline completed successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument('--mimic_path', type=str, required=True,
                      help='Path to MIMIC-III NOTEEVENTS.csv')
    parser.add_argument('--casi_path', type=str, required=True,
                      help='Path to CASI dataset CSV file')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory for processed data')
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory for model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for logs')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=100,
                      help='Dimension of word and metadata embeddings')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Dimension of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--mask_prob', type=float, default=0.2,
                      help='Probability of masking words during training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    main(args) 