# Human Segmentation in PyTorch for Production

End-to-end human segmentation project in PyTorch with a production-oriented structure.  
The repository covers dataset preparation, preprocessing, training, evaluation, checkpointing, inference, and prediction visualization.

## Project Goal

Build a complete computer vision pipeline for **binary human semantic segmentation**, from raw dataset preparation to model training and inference.

## Problem Definition

- **Input:** RGB image
- **Output:** binary segmentation mask
  - `1` = person
  - `0` = background

The project is based on **CIHP (Crowd Instance-level Human Parsing)**, but reformulated from fine-grained human parsing into a binary human segmentation task.

## Dataset Choice

This project uses the **CIHP (Crowd Instance-level Human Parsing)** dataset as the main training benchmark.

Although CIHP is originally designed for fine-grained human parsing, this project reformulates the task into binary semantic segmentation:

- foreground (person) = `1`
- background = `0`

All human-part labels are merged into a single human class. This keeps the dataset realism and complexity while aligning the project with a production-oriented segmentation use case.

## Implemented Pipeline

The repository currently includes:

- CIHP raw data structure validation
- CIHP preprocessing into binary masks
- PyTorch `Dataset` and `DataLoader`
- Processed sample visualization
- U-Net baseline model
- Segmentation losses and metrics
  - BCE
  - BCE + Dice
  - Dice score
  - IoU
- Tiny overfit sanity check
- Baseline training loop with validation
- Best-checkpoint saving
- Checkpoint loading and prediction visualization
- Separate local and Colab-oriented training configurations

## Current Status

The full end-to-end pipeline is already functional:

- raw CIHP data can be validated and preprocessed
- processed data can be loaded and batched
- the U-Net baseline can be trained and validated
- checkpoints can be saved and loaded
- predictions can be visualized against ground truth

A first local CPU baseline has already been validated.  
The next stage is running a longer GPU training workflow in Colab.

## Repository Structure

```text
.
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
│   ├── api/
│   ├── data/
│   ├── inference/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── requirements.txt
└── README.md