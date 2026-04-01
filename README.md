# Fidelity-Resolved Affine Matrix Emulator (FRAME)

Code accompanying the paper: *Linking Electromagnetic Moments to Nuclear Interactions with a Global Physics-Driven Machine-Learning Emulator* ([arXiv:2603.26905](https://arxiv.org/abs/2603.26905)).

FRAME is a global, physics-constrained emulator that maps low-energy constants (LECs) of chiral effective field theory, nuclear identifiers (Z, N), and fidelity (e_max) to nuclear observables:

```
g(Z, N, f, alpha) -> {E, R_ch, mu, Q, ...}
```

The architecture combines a global latent encoder (following [BANNANE](https://arxiv.org/abs/2501.12345)) with a convergence-aware parametric operator core inspired by [Parametric Matrix Models](https://arxiv.org/abs/2401.12345). It constructs effective Hamiltonians and observable operators as low-dimensional Hermitian matrices whose entries depend affinely on the LECs, preserving the structure of the chiral interaction. Energies are obtained by diagonalizing the effective Hamiltonian; observables are computed as eigenvector bilinears. Predictions at successive fidelities (e_max in {4, 6, 8, 10}) are controlled hierarchically by increasing the matrix size, enabling controlled extrapolation across model-space truncations.

The training data consists of valence-space in-medium similarity renormalization group (VS-IMSRG) calculations using the `imsrg++` code, followed by exact diagonalization via `KSHELL`, performed for an ensemble of ~10^4 non-implausible chiral interactions derived from history matching.

## Repository Structure

```
FRAME-public/
├── README.md
├── .gitignore
├── tests/
│   └── test_model.py            # Unit tests (config, model, training, inference)
├── notebooks/
│   └── run_train.ipynb          # Interactive training notebook (oxygen example)
└── frame_public/
    ├── environment.yaml
    ├── data/
    │   ├── oxygen/              # O-12 through O-24 (E_b, R_ch)
    │   └── calcium_odd/         # Ca-37 through Ca-55, odd isotopes (E_b, R_ch, M1, E2, M1_2BC)
    ├── src/
    │   ├── config.py            # Model and training configuration
    │   ├── model.py             # Parametric matrix model (FRAME)
    │   ├── train.py             # Training loop
    │   ├── data_loader.py       # Multi-isotope CSV data loader
    │   ├── inference.py         # Prediction / inference handler
    │   └── utils.py             # Dataloaders, scalers, utilities
    └── scripts/
        ├── train.py             # CLI training entry point
        ├── eval.py              # CLI evaluation entry point
        └── slurm/
            ├── train_oxygen.sh
            └── train_calcium_odd.sh
```

## Installation

```bash
conda env create -f frame_public/environment.yaml
conda activate frame
```

## Quick Start

### Training on Oxygen (local)

```bash
python frame_public/scripts/train.py \
  --data_dir frame_public/data/oxygen \
  --file_pattern "O*.csv" \
  --target_cols "Energy ket,Rch" \
  --device cpu \
  --epochs 5000 \
  --save_root results/oxygen
```

### Training on Calcium odd isotopes (local)

```bash
python frame_public/scripts/train.py \
  --data_dir frame_public/data/calcium_odd \
  --file_pattern "Ca*.csv" \
  --device cpu \
  --epochs 5000 \
  --save_root results/calcium_odd
```

### SLURM

```bash
sbatch frame_public/scripts/slurm/train_oxygen.sh
sbatch frame_public/scripts/slurm/train_calcium_odd.sh
```

### Evaluation

After training completes, evaluate on the test split:

```bash
python frame_public/scripts/eval.py --exp_dir <path_to_run_directory>
```

## Running Tests

```bash
conda activate frame
python -m pytest tests/ -v
```

## Example Notebook

The `notebooks/run_train.ipynb` notebook walks through an interactive training run on oxygen data. To use it:

```bash
conda activate frame
python -m ipykernel install --user --name frame --display-name "frame"
jupyter notebook notebooks/run_train.ipynb
```

## Data Description

Each CSV file corresponds to one isotope and contains VS-IMSRG + KSHELL calculations at varying model-space fidelities (`emax`), computed for a subset of the non-implausible LEC ensemble from [Jiang et al. (2024)](https://arxiv.org/abs/2401.12345). Columns include:

- **Input LECs** (17 columns): the low-energy constants of the N2LO chiral interaction with Delta isobars — `Ct1S0pp`, `Ct1S0np`, `Ct1S0nn`, `Ct3S1`, `C1S0`, `C3P0`, `C1P1`, `C3P1`, `C3S1`, `CE1`, `C3P2`, `c1`, `c2`, `c3`, `c4`, `cD`, `cE`
- **emax**: model-space fidelity parameter (single-particle truncation, e = 2n + l)
- **Target observables**: `Energy ket` (binding energy, MeV), `Rch` (charge radius, fm), and for calcium odd isotopes also `M1` (magnetic dipole), `E2` (electric quadrupole), `M1_2BC` (two-body current contribution to M1)

## Customizing for Your Data

FRAME is designed to be flexible and can be adapted to different input features and target observables.

### Using Custom Input Features

By default, the model uses the 17 chiral LECs as input features. However, you can use any number of input features by specifying the `--input_cols` argument during training:

```bash
python frame_public/scripts/train.py \
  --data_dir <your_data_dir> \
  --input_cols "feature1,feature2,feature3" \
  --target_cols "observable1,observable2" \
  ...
```

The model architecture automatically adjusts its input layer size based on the number of columns provided.

### Adding New Isotopes/Data

To add data for a new isotopic chain:
1. Create a new directory in `frame_public/data/`.
2. Provide CSV files following the naming convention `ElementMass_*.csv` (e.g., `Sn132_data.csv`).
3. Ensure the CSVs contain the columns specified in your `--input_cols` and `--target_cols`, as well as a fidelity column (default `emax`).
4. Update `frame_public/src/data_loader.py` if you use a new element symbol not already in the `periodic_table` dictionary.


## Citation

```bibtex
@article{munoz2025linking,
  title   = {Linking Electromagnetic Moments to Nuclear Interactions with a Global Physics-Driven Machine-Learning Emulator},
  author  = {Munoz, Jose M. and Belley, Antoine and Ekstr{\"o}m, Andreas and Hagen, Gaute and Holt, Jason D. and Garcia Ruiz, Ronald F.},
  year    = {2026},
  eprint  = {2603.26905},
  archivePrefix = {arXiv},
  primaryClass  = {nucl-th},
}
```
If you have any questions, encounter issues, or would like to collaborate, please contact:

**Jose M. Munoz** (jmmunoz [at] mit [dot] edu)


---

EMA Lab, MIT. 2026
