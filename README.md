# Latent Meaning Cells (LMC)

This repository contains an implementation of the Latent Meaning Cells (LMC) model for clinical acronym expansion, as described in the paper "Latent Meaning Cells: A Probabilistic Model for Learning Contextualized Word Embeddings with Metadata" (Adams et al., 2020).

## Overview

LMC is a novel probabilistic model that learns contextualized word embeddings by combining local lexical context with document metadata. In the clinical domain, metadata (such as section headers or note types) provide vital cues for word senses (e.g., an acronym's meaning).

Key features:
- Joint word-metadata modeling
- Zero-shot acronym expansion
- Efficient pre-training on clinical text
- Superior performance on clinical acronym expansion tasks

## Requirements

- Python 3.6+
- PyTorch >= 1.0
- NumPy
- tqdm
- scikit-learn
- pandas
- transformers (for BERT baseline)
- allennlp (for ELMo baseline)

## Installation

```bash
# Clone the repository
git clone https://github.com/RParkerE/LMC.git
cd LMC

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### MIMIC-III
1. Register for access at [PhysioNet](https://mimic.physionet.org/)
2. Download MIMIC-III v1.4
3. Extract the NOTEEVENTS table
4. Run preprocessing scripts:
```bash
python preprocess/extract_sections.py --input_path /path/to/mimic/notes --output_path data/processed/mimic
```

### CASI Dataset
1. Download from [University of Minnesota](https://hdl.handle.net/11299/137703)
2. Process using provided scripts:
```bash
python preprocess/process_casi.py --input_path /path/to/casi --output_path data/processed/casi
```

## Training

Train the LMC model:
```bash
python train.py \
    --data_path data/processed/mimic \
    --model_dir models \
    --embedding_dim 100 \
    --hidden_dim 64 \
    --dropout 0.2 \
    --learning_rate 1e-3 \
    --epochs 5 \
    --window_size 10 \
    --mask_prob 0.2
```

## Evaluation

Evaluate on test sets:
```bash
python evaluate.py \
    --model_path models/lmc_best.pt \
    --test_path data/processed/casi \
    --output_path results
```

## Project Structure

```
LMC/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed datasets
├── models/
│   ├── lmc.py           # LMC model implementation
│   ├── bsg.py           # Bayesian Skip-Gram baseline
│   └── mbsge.py         # Metadata BSG Ensemble baseline
├── preprocess/
│   ├── extract_sections.py
│   ├── process_casi.py
│   └── reverse_substitution.py
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── utils/
│   ├── data_utils.py
│   └── metrics.py
├── requirements.txt
└── README.md
```

## Results

The model achieves:
- ~74% weighted accuracy on MIMIC-RS test set
- 69% macro F1 score
- Superior performance compared to baselines (BERT, ELMo, etc.)

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