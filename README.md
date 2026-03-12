# Human Segmentation in PyTorch for Production

End-to-end human segmentation project in PyTorch, designed with a production-oriented structure. The project will cover data handling, model training, evaluation, inference, API serving, Dockerization, and CI.

## Project Goal

Build a complete computer vision pipeline for human segmentation, from dataset preparation and model training to inference and deployment.

## Planned Features

- Dataset organization and preprocessing
- Human segmentation model training in PyTorch
- Validation and evaluation with appropriate metrics
- Inference pipeline for new images
- API serving with FastAPI
- Docker support
- Basic automated tests
- CI workflow with GitHub Actions

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
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── requirements.txt
└── README.md

## Dataset Choice

This project uses the CIHP (Crowd Instance-level Human Parsing) dataset as the main training benchmark.

Although CIHP is originally designed for fine-grained human parsing, this project reformulates the task into binary semantic segmentation:
- foreground (person) = 1
- background = 0

All human-part labels are merged into a single human class. This choice keeps the dataset complexity and realism while aligning the project with a production-oriented human segmentation use case.
