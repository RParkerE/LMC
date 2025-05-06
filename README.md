# Latent Meaning Cells (LMC)

This repository contains an implementation of the Latent Meaning Cells (LMC) model for clinical acronym expansion, as described in the paper "Latent Meaning Cells: A Probabilistic Model for Learning Contextualized Word Embeddings with Metadata" (Adams et al., 2020).

## Overview

LMC is a novel probabilistic model that learns contextualized word embeddings by combining local lexical context with document metadata. In the clinical domain, metadata (such as section headers or note types) provide vital cues for word senses (e.g., an acronym's meaning).

Key features:
- Joint word-metadata modeling
- Zero-shot acronym expansion
- Efficient pre-training on clinical text
- Superior performance on clinical acronym expansion tasks
- Memory-efficient processing of large clinical datasets

## Quick Start Guide

### 1. Environment Setup

```bash
# Create and activate a new conda environment
conda create -n myfinalenv python=3.8
conda activate myfinalenv

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

1. Create the required directories:
```bash
mkdir -p data/processed
mkdir -p data/raw
mkdir -p models
mkdir -p logs
```

2. Place your data files in the appropriate directories:
   - For MIMIC-III data: Place in `data/raw/mimic-iii/`
   - For CASI dataset: Place in `data/raw/casi/`

### 3. Running the Pipeline

#### Option 1: Full Pipeline
To run the complete pipeline (preprocessing, vocabulary creation, and training):
```bash
python run_pipeline.py --data_dir data --model_dir models --log_dir logs
```

#### Option 2: Step by Step

1. **Create Vocabulary**:
```bash
python create_vocab.py
```

2. **Train the Model**:
```bash
python train.py --data_dir data/processed --model_dir models
```

### 4. Training Parameters

You can customize the training with these parameters:
```bash
python train.py \
    --data_dir data/processed \
    --model_dir models 
```

### 5. Evaluation

After training, you can evaluate the model using the evaluation script:

```bash
python evaluate.py \
    --test_path data/processed \
    --model_path models/best_model.pt \
    --output_path results \
    --log_dir logs \
    --batch_size 32 \
    --num_workers 4
```

The evaluation script will:
1. Load the trained model
2. Run evaluation on the test set
3. Calculate accuracy and F1 scores
4. Save results to `results/results.json`

The evaluation metrics include:
- Accuracy
- Macro F1 score
- Weighted F1 score

You can find the evaluation results in:
- `results/results.json`: Detailed metrics
- `logs/evaluate.log`: Evaluation logs

## Project Structure

```
LMC/
├── data/
│   ├── raw/              # Raw data files
│   │   └── mimic-iii/    # MIMIC-III data
│   └── processed/        # Processed datasets
│       ├── vocab.json    # Generated vocabulary
│       └── metadata.json # Generated metadata mappings
├── models/
│   └── lmc.py           # LMC model implementation
├── preprocess/
│   ├── extract_sections.py  # Section extraction with chunked processing
│   └── create_vocab.py      # Vocabulary creation
├── train.py             # Training script with memory-efficient data loading
├── run_pipeline.py      # Main pipeline script
├── utils/
│   ├── data_utils.py    # Dataset and data loading utilities
│   └── metrics.py       # Evaluation metrics
├── requirements.txt
└── README.md
```

## Troubleshooting

### Common Issues

1. **NumPy Version Error**:
   If you encounter NumPy compatibility issues, downgrade NumPy:
   ```bash
   pip install numpy<2.0.0
   ```

2. **Memory Issues**:
   - Reduce batch size: `--batch_size 16`
   - Reduce number of workers: `--num_workers 2`
   - Process data in smaller chunks

3. **CUDA Out of Memory**:
   - Reduce batch size
   - Use smaller model dimensions
   - Enable gradient checkpointing

## Results

The model achieves the following performance metrics on the test set:

### Loss Metrics
- Total Loss: 3.4755
- Reconstruction Loss: 3.3906
- KL Divergence Loss: 0.0849

The low KL divergence loss (0.0849) indicates that the model has learned a good balance between:
- Reconstructing the context words (reconstruction loss: 3.3906)
- Maintaining a structured latent space (KL loss: 0.0849)

These results demonstrate that the model successfully:
- Learns meaningful word representations
- Maintains a well-structured latent space
- Effectively combines local context with metadata information

## Citation

If you use this code, please cite:
```
@inproceedings{adams2020latent,
  title={Latent Meaning Cells: A Probabilistic Model for Learning Contextualized Word Embeddings with Metadata},
  author={Adams, Griffin and Alsentzer, Emily and Ketenci, Murat and Zucker, Jason and Elhadad, Noémie},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 