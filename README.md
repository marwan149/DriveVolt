# Three-Phase IGBT Two-Level Inverter for Electrical Drives

## Overview
This repository implements a clean, modular Python package for processing inverter data, computing Clarke and Park transforms, visualizing results, and training a neural network model.

## Project structure
- `run.py` - Launcher script.
- `requirements.txt` - Python dependencies.
- `.gitignore` - Ignored files.
- `Datasetlink.txt` - Download link for the dataset.
- `src/inverter_model/` - Package source code:
  - `main.py` - Pipeline entry point and CLI.
  - `preprocessing.py` - Data loading, cleaning, and feature engineering.
  - `transforms.py` - Clarke and Park transforms.
  - `visualization.py` - Plotting utilities.
  - `modeling.py` - Model creation, training, and cross-validation.

## Installation
1. Create and activate a Python environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Dataset
Place `Inverter Data Set.csv` in a `data/` folder at the project root.
Use the URL in `Datasetlink.txt` to download the dataset if needed.

## Usage
Run the pipeline with:

```powershell
python run.py --data-path data/Inverter Data Set.csv
```

Optional CLI flags:
- `--sample-size` — number of rows used for plotting (default `5000`)
- `--outlier-threshold` — z-score limit for outlier capping (default `3.0`)
- `--noise-factor` — synthetic augmentation noise factor (default `0.01`)

## License
MIT License.
