# SpectrumDETR: Representation Learning and Set Prediction for Polymer XPS Analysis

This repository contains code and analysis for a machine learning framework for interpreting polymer X-ray photoelectron spectroscopy (XPS) spectra using learned chemical representations and transformer-based set prediction.

Rather than treating XPS analysis as a discrete peak-classification problem, this project formulates spectral interpretation as a continuous set-prediction task. Each spectrum is represented as an unordered set of local chemical environment embeddings, allowing the model to predict variable numbers of chemically distinct contributors directly from spectral data. The work explores how different chemical representations influence spectral interpretation, uncertainty calibration, and generalisation to unseen chemical environments.

Polymer XPS spectra often contain heavily overlapping core-level features, making manual peak fitting subjective, chemically ambiguous, and difficult to scale. The framework developed here aims to provide a more flexible and chemically continuous approach to spectral interpretation.

The repository compares multiple representation strategies, including:

- **SkipAtom**: distributed element embeddings learned from local coordination environments
- **MatScholar**: literature-derived materials science embeddings
- **SOAP**: Smooth Overlap of Atomic Positions descriptors for local atomic environments

These representations are used as continuous targets within a DETR-inspired architecture adapted for one-dimensional spectra.

## Model

The main model combines convolutional feature extraction with transformer-based set prediction:

- A 1D convolutional encoder extracts local spectral features
- A transformer module captures long-range spectral relationships
- A fixed set of learned query slots predicts unordered local environment embeddings
- Hungarian matching is used during training to align predicted and ground-truth environments

The framework treats XPS interpretation as a permutation-invariant set prediction problem, avoiding predefined peak assignments or discrete chemical-state labels.

## Data

The models in this repository were trained using experimentally derived polymer XPS spectra together with generated local environment representations.

Due to repository size constraints, processed datasets and trained model checkpoints are not included. Small example files and processing scripts are provided to demonstrate the workflow.

Additional data or processed representations may be made available upon reasonable request.

## Requirements

Main dependencies include:

- Python
- PyTorch
- RDKit
- DScribe
- scikit-learn
- NumPy
- pandas
- matplotlib

## Repository Structure

```text
SpectrumDETR/
├── scripts/
│   ├── build_dataset.py
│   ├── build_synthetic_dataset.py
│   ├── dataset.py
│   ├── detr_model.py
│   ├── evaluate_unseen_polymers.py
│   ├── evaluate_val_accuracy.py
│   ├── plot_loss_curves.py
│   ├── plot_ood_pca.py
│   ├── plot_paper_figures.py
│   ├── plot_representation_tsne.py
│   ├── run_confidence_quality_analysis.py
│   ├── run_detr_experiments.py
│   └── run_scaling_study.py
├── *.ipynb
└── README.md
