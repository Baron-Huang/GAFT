# GAFT: Graph-Adversarial Fusion Transformer

This repository contains the research code used to develop and evaluate GAFT for three-class pathological grading. GAFT combines a Swin Transformer image branch with a ViG graph-based branch, then learns a fused prediction through the modules implemented in `models/my_net.py`.

> **Repository status.** This is an experiment-oriented research codebase. The training and evaluation scripts contain dataset- and machine-specific paths that must be configured before use. Raw pathology images, pretrained backbone weights, and trained GAFT checkpoints are not distributed in this repository.

<p align="center">
  <img src="Images/Fig_4.jpg" width="850" alt="Overview of the GAFT framework">
</p>

## Repository layout

```text
.
├── GAFT_SwinT_LSCC.py                 # GAFT: AMU-LSCC-ROI experiment
├── GAFT_SwinT_ESCC.py                 # GAFT: AMU-ESCC-ROI experiment
├── GAFT_SwinT_Greece_Larynx.py        # GAFT: Greece-Larynx experiment
├── GAFT_SwinT_Greece_Breast.py        # GAFT: Greece-Breast experiment
├── GAFT_ViT_ESCC.py                   # ViT-based GAFT variant
├── GAFT_ViT_wo_CTO_GAC_ESCC.py        # Ablation variant
├── GAFT_ViT_wo_GAC_ESCC.py            # Ablation variant
├── Fusion_method_ablat/               # Concatenation, cross-attention, gated,
│                                      # late-fusion, and weighted-summation baselines
├── SOTAs_main_func/                   # Single-backbone comparison methods and runners
├── models/                             # Swin, ViG/graph, fusion, and utility modules
├── loss_funcs/                         # Distribution and adversarial loss functions
├── utils/                              # Training, testing, metric, and logging functions
├── CAM_out.py, GNN_out.py              # CAM/feature visualisation entry points
├── CAM_module.py, GNN_cam_utils.py     # Visualisation utilities
└── *.py, *.csv, *.npz, *.png           # Experiment scripts and analysis artefacts
```

## Environment

The code requires Python, PyTorch, torchvision, and a CUDA-enabled NVIDIA GPU for the reference training workflow. No locked dependency file is included in the current repository; create an isolated environment and install packages compatible with your CUDA driver.

```bash
conda create -n gaft python=3.10 -y
conda activate gaft
# Install the PyTorch/torchvision build matching your CUDA environment first.
pip install numpy scipy pandas scikit-learn matplotlib seaborn timm einops torchinfo
```

Several comparison models under `SOTAs_main_func/` have additional, model-specific dependencies. Install these only when running the relevant script; examples include `transformers`, `torchcam`, and packages required by the original upstream implementations.

## Data layout

The GAFT entry scripts load images with `torchvision.datasets.ImageFolder`. Each dataset root must use the following structure:

```text
DATA_ROOT/
├── Train_patch/
│   ├── I/
│   ├── II/
│   └── III/
├── Val_patch/
│   ├── I/
│   ├── II/
│   └── III/
└── Test_patch/
    ├── I/
    ├── II/
    └── III/
```

The class-folder names determine the class indices. The provided GAFT scripts use 224 × 224 input images, three classes, batch size 32, 100 epochs, and random seed 0 by default. These settings are defined directly in each entry script and can be changed there.

## Pretrained backbones and checkpoints

The GAFT Swin-T + ViG scripts expect two pretrained backbone checkpoints:

- a Swin-Tiny checkpoint, loaded by `models/swin_transformer.py`;
- a ViG checkpoint, loaded by `models/gcn.py`.

Set the following variables near the beginning of the selected entry script before running it:

```python
data_root = r"/path/to/DATA_ROOT"
swinT_path = r"/path/to/swin_tiny_checkpoint.pth"
gcn_path = r"/path/to/vig_checkpoint.pth"
model_path = r"/path/to/gaft_checkpoint.pth"
model_stat = "train"  # use "test" to evaluate an existing checkpoint
```

`model_path` is the output checkpoint location in training mode and the checkpoint loaded in test mode. Create its parent directory before running a script.

## Training and evaluation

Run commands from the repository root so that the local `models`, `utils`, and `loss_funcs` packages can be imported.

```bash
# Example: configure paths and set model_stat = "train" in the script first.
python GAFT_SwinT_Greece_Larynx.py

# Example: evaluate a saved model after setting model_stat = "test".
python GAFT_SwinT_Greece_Larynx.py
```

The same pattern applies to `GAFT_SwinT_LSCC.py`, `GAFT_SwinT_ESCC.py`, and `GAFT_SwinT_Greece_Breast.py`. In test mode, `utils/train_utils.py` reports classification metrics and can write classification reports, AUC, and confusion-matrix statistics to the corresponding log file when a log path is supplied by the entry script.

## Fusion ablations and single-backbone comparisons

- `Fusion_method_ablat/` contains five fusion strategies: concatenation, cross-attention, gated fusion, late fusion, and weighted summation. `run_fusion_parallel.py` can schedule the fusion scripts.
- `SOTAs_main_func/` contains CNN, Transformer, graph, and MLP comparison scripts. `run_sotas_parallel.py` runs the configured comparison scripts while monitoring GPU memory.

Each script has its own dataset, pretrained-weight, output-checkpoint, and `mode_stat`/`model_stat` settings. Review and update those variables before execution.

## Visualisation and analysis

CAM and feature-analysis scripts are included for experimental inspection:

- `CAM_out.py`, `CAM_module.py`, and `GNN_cam_utils.py` provide CAM-related utilities;
- `GNN_out.py` and `feature_map_out.py` support graph and feature-map outputs;
- root-level analysis scripts and CSV/NPZ files record experiment-specific exploratory analyses.

These utilities also contain local paths and should be configured for the target environment.

## Data, weights, and third-party code

No raw patient-derived images, patient identifiers, pretrained weights, or trained GAFT checkpoints are included. Access to pathology data must follow the relevant ethics approvals, institutional policies, and data-use agreements. Users are responsible for obtaining third-party pretrained weights and complying with the licences of bundled or referenced upstream implementations.

## Citation

Citation information will be added when the associated manuscript is publicly available.

## Contact

Pan Huang
Email: panhuang@polyu.edu.hk
