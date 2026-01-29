# An Ensemble Machine Learning Framework for Satellite-Based Air Quality Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Enabled-green.svg)](https://earthengine.google.com/)
[![Open Access](https://img.shields.io/badge/Open%20Access-Yes-brightgreen.svg)]()

---

## Overview

This repository contains the complete code, data, and reproducible analysis pipeline for the paper:

> **"An Ensemble Machine Learning Framework for Satellite-Based Air Quality Estimation: A Comparative Study with Uncertainty Quantification"**
>
> Submitted to *Environmental Data Science* (Cambridge University Press)

The study develops and evaluates an ensemble Random Forest framework for estimating tropospheric NO₂ concentrations from multi-source satellite observations across Delhi, India (2021-2023). It provides a systematic comparison of machine learning (Random Forest, XGBoost) and deep learning (Neural Networks) approaches, with a novel focus on uncertainty quantification via ensemble variance.

---

## Key Findings

| Finding | Details |
|---------|---------|
| **Best Model** | Random Forest Ensemble (R² = 0.636, RMSE = 3.8×10⁻⁵ mol/m²) |
| **Deep Learning Performance** | Catastrophic failure (R² < 0) on dataset with <400 samples |
| **Dominant Predictor** | Temperature explains 67% of NO₂ variability |
| **Spatial Transferability** | Moderate (mean LOGO-CV R² = 0.533) |
| **Uncertainty Calibration** | 91% of observations within 2σ prediction intervals |

---

## Repository Structure
delhi-air-quality-ml/
│
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment file
│
├── notebooks/ # Jupyter notebooks (run in order)
│ ├── 01_data_acquisition.ipynb # GEE data extraction
│ ├── 02_preprocessing.ipynb # Data cleaning and merging
│ ├── 03_model_training.ipynb # ML/DL model development
│ ├── 04_evaluation.ipynb # Validation and analysis
│ └── 05_figure_generation.ipynb # Publication figures
│
├── data/
│ ├── raw/ # Raw data (downloaded via GEE)
│ │ └── .gitkeep
│ ├── processed/ # Processed datasets
│ │ ├── delhi_aq_final_dataset.csv # Main analysis dataset (376 samples)
│ │ └── results_summary.pkl # Saved results dictionary
│ └── shapefiles/ # Administrative boundaries
│ └── delhi_districts/ # Delhi district polygons
│
├── models/ # Trained models
│ ├── rf_ensemble_models.pkl # 20-member RF ensemble
│ └── scaler.pkl # StandardScaler object
│
├── figures/ # Publication-ready figures
│ ├── Figure1_Model_Performance.png
│ ├── Figure1_Model_Performance.pdf
│ ├── Figure2_Feature_Importance.png
│ ├── Figure2_Feature_Importance.pdf
│ ├── Figure3_Temporal_Patterns.png
│ ├── Figure3_Temporal_Patterns.pdf
│ ├── Figure4_Spatial_Transferability.png
│ ├── Figure4_Spatial_Transferability.pdf
│ ├── Figure5_Spatial_Maps.png
│ ├── Figure5_Spatial_Maps.pdf
│ ├── Figure6_Uncertainty_Analysis.png
│ └── Figure6_Uncertainty_Analysis.pdf
│
└── src/ # Source code modules (optional)
├── init.py
├── data_utils.py # Data processing functions
├── models.py # Model definitions
└── evaluation.py # Evaluation metrics


---

## Data Sources

All data used in this study are **freely available** from the following sources:

| Data | Source | Access | Resolution |
|------|--------|--------|------------|
| Tropospheric NO₂ | Sentinel-5P TROPOMI | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2) | ~5.5 km |
| NDVI, NDBI | Sentinel-2 MSI | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) | 10 m |
| Meteorology | ERA5-Land | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR) | ~9 km |
| District Boundaries | OpenStreetMap | [Nominatim API](https://nominatim.openstreetmap.org/) | Vector |

**Google Earth Engine Project ID:** `ee-satyamshah444`

---

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/[your-username]/delhi-air-quality-ml.git
cd delhi-air-quality-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clone the repository
git clone https://github.com/[your-username]/delhi-air-quality-ml.git
cd delhi-air-quality-ml

# Create conda environment
conda env create -f environment.yml
conda activate delhi-aq

# Core
python>=3.10
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Geospatial
earthengine-api>=0.1.370
geemap>=0.28.0
geopandas>=0.14.0
rasterio>=1.3.0
shapely>=2.0.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
requests>=2.28.0
tqdm>=4.65.0
pickle5>=0.0.11

name: delhi-aq
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scipy>=1.10.0
  - geopandas>=0.14.0
  - rasterio>=1.3.0
  - scikit-learn>=1.3.0
  - xgboost>=2.0.0
  - pytorch>=2.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - jupyter>=1.0.0
  - pip
  - pip:
    - earthengine-api>=0.1.370
    - geemap>=0.28.0


