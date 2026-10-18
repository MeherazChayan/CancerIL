# CancerIL-Skin

CancerIL-Skin is a GPU-accelerated research pipeline for studying continual learning in skin-lesion classification. It expands a classifier trained on HAM10000 with classes from MILK10k and evaluates generalization on the DDI dataset.

The notebook includes reproducible multi-seed experiments, class/domain-incremental and strict class-incremental protocols, established continual-learning baselines, ablation studies, statistical testing, representation-drift analysis, Grad-CAM visualizations, and paper-ready tables and figures.

## Main notebook

- [`canceril_skin_continual_learning_pipeline.ipynb`](canceril_skin_continual_learning_pipeline.ipynb) — complete training, validation, evaluation, and analysis pipeline.

## Highlights

- GPU-only training optimized for an NVIDIA RTX 4070 SUPER (12 GB)
- EfficientNet-B3, ConvNeXt-Tiny, and Swin-T backbone experiments
- HAM10000 stage-one training and MILK10k class expansion
- External validation using DDI
- CDI and strict class-incremental protocols
- LwF, EWC, Experience Replay, DER++, joint-training, and proposed CancerIL methods
- Multi-seed confidence intervals and paired significance tests
- Per-class forgetting, modality, skin-tone, replay-budget, CKA, and Grad-CAM analyses
- Resume-safe experiment outputs and checkpoints

## Requirements

- Python 3
- A CUDA-capable NVIDIA GPU (the notebook stops if CUDA is unavailable)
- PyTorch and torchvision with CUDA support
- NumPy
- pandas
- Pillow
- Matplotlib
- SciPy
- scikit-learn
- tqdm
- Jupyter Notebook or JupyterLab

Install the CUDA-compatible PyTorch build for your system first, then install the remaining packages, for example:

```bash
pip install numpy pandas pillow matplotlib scipy scikit-learn tqdm jupyter
```

## Datasets

Download and prepare these datasets separately; they are not included in this repository:

- HAM10000
- MILK10k
- DDI (Diverse Dermatology Images)

Update `HAM_ROOT`, `MILK_ROOT`, `DDI_ROOT`, and `BASE_OUTPUT` in the notebook's configuration cell to match your local directories.

## Usage

1. Open the notebook in Jupyter.
2. Configure the dataset and output paths.
3. Set `SMOKE_TEST = True` for the first run.
4. Run all cells and confirm that the end-to-end smoke test succeeds.
5. Set `SMOKE_TEST = False` to run the complete experiments.

The full experiment is computationally intensive. Completed runs are saved to disk, allowing interrupted sessions to resume without repeating finished work.

## Reproducibility notes

- The main experiment uses multiple random seeds.
- Dataset splits, predictions, checkpoints, metrics, tables, and figures are written beneath the configured output directory.
- Hyperparameters are selected using validation data only.
- Review the notebook's configuration cell before execution, especially GPU, checkpoint, modality, and dataset-policy settings.

## Data and ethics

This repository contains research code, not clinical software. It is not intended for diagnosis, treatment, or deployment in patient care. Dataset licenses, access conditions, and privacy requirements remain the responsibility of the user.
