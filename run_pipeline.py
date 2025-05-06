import argparse
import os
import subprocess
import logging
from pathlib import Path
import time

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
    logging.info(f"Command: {command}")
    
    try:
        # Use subprocess.Popen to capture output in real-time
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logging.info(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        if return_code == 0:
            logging.info(f"{description} completed successfully")
        else:
            # Get error output
            error_output = process.stderr.read()
            logging.error(f"{description} failed with error:\n{error_output}")
            raise subprocess.CalledProcessError(return_code, command)
            
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with error:\n{e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in {description}: {str(e)}")
        raise

def main(args):
    setup_logging(args.log_dir)
    logging.info("Starting LMC pipeline")
    
    # Create necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Verify input file exists
    input_path = os.path.join(args.data_dir, 'raw/mimic-iii/NOTEEVENTS.csv')
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
    
    logging.info(f"Found input file: {input_path}")
    logging.info(f"File size: {os.path.getsize(input_path) / (1024*1024*1024):.2f} GB")
    
    # Step 1: Process MIMIC-III notes
    logging.info("Step 1: Processing MIMIC-III notes")
    start_time = time.time()
    
    try:
        run_command(
            f"python preprocess/extract_sections.py "
            f"--input_path {input_path} "
            f"--output_path {os.path.join(args.data_dir, 'processed/sections')} "
            f"--log_dir {args.log_dir} "
            f"--chunk_size 10000",
            "Section extraction"
        )
        logging.info(f"Section extraction completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Section extraction failed: {str(e)}")
        return
    
    # Step 2: Create vocabulary and metadata
    logging.info("Step 2: Creating vocabulary and metadata")
    start_time = time.time()
    
    try:
        run_command(
            f"python preprocess/create_vocab.py "
            f"--input_dir {os.path.join(args.data_dir, 'processed/sections')} "
            f"--output_dir {os.path.join(args.data_dir, 'processed')} "
            f"--log_dir {args.log_dir}",
            "Vocabulary creation"
        )
        logging.info(f"Vocabulary creation completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Vocabulary creation failed: {str(e)}")
        return
    
    # Step 3: Train model
    logging.info("Step 3: Training model")
    start_time = time.time()
    
    try:
        run_command(
            f"python train.py "
            f"--data_dir {os.path.join(args.data_dir, 'processed')} "
            f"--model_dir {args.model_dir} "
            f"--log_dir {args.log_dir} "
            f"--batch_size 32 "
            f"--num_workers 4 "
            f"--window_size 10 "
            f"--embedding_dim 256 "
            f"--hidden_dim 512 "
            f"--num_layers 2 "
            f"--dropout 0.1 "
            f"--num_epochs 10 "
            f"--learning_rate 0.001 "
            f"--mask_prob 0.2",
            "Model training"
        )
        logging.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        return
    
    logging.info("Pipeline completed successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Root directory for data')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    
    args = parser.parse_args()
    
    main(args) 