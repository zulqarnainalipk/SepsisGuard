# SepsisGuard 🩺⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![Clinical AI](https://img.shields.io/badge/Domain-Clinical_AI-important)
![Banner](data/baneer.png)

A machine learning system for early detection of pediatric sepsis in PICUs, predicting sepsis onset 6 hours in advance using clinical time-series data.

## 📖 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data](#-data)
- [Notebooks](#-notebooks)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

## 🚀 Features
- 6-hour early sepsis prediction window
- Processes 10+ clinical data modalities
- Automated feature engineering pipeline
- Production-ready API endpoints (WIP)
- SHAP-based model interpretability

## 💻 Installation

```bash
git clone https://github.com/YOUR_USERNAME/SepsisGuard.git
cd SepsisGuard

# Create virtual environment
python -m venv sepsisenv
source sepsisenv/bin/activate  # Linux/Mac
# sepsisenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup directories
mkdir -p data/raw/{training_data,testing_data}
```

## 🛠️ Usage

### Train Model
```bash
python main.py train --config config/paths.yaml
```

### Generate Predictions
```bash
python main.py predict \
    --input data/raw/testing_data \
    --output predictions/risk_scores.csv \
    --config config/paths.yaml
```

## 📂 Project Structure
```
.
├── data/                   # Data storage
│   ├── raw/                # Original CSVs
│   └── processed/          # Cleaned data
│
├── notebooks/              # Research notebooks
│   └── competition_notebook.ipynb  # Original analysis
│
├── src/                    # Core code
│   ├── data_loader.py      # Data ingestion
│   ├── preprocessor.py     # Feature pipeline
│   ├── model.py            # Model classes
│   ├── train.py            # Training script
│   └── predict.py          # Inference script
│
├── models/                 # Saved models
├── config/                 # Configuration
├── requirements.txt        # Dependencies
└── README.md               # This file

```
## 📊 Data
**Source:** Hospital Sant Joan de Déu PICU registry  
**Format:** Hourly time-series with 50+ clinical features  

**Tables Included:**
- Patient demographics
- Medical device usage
- Drug administration
- Lab measurements
- Clinical observations
- Procedure records

**Preprocessing:**
1. Temporal alignment to hourly intervals
2. Medication history tracking
3. Vital sign anomaly detection
4. Categorical feature encoding
5. 6-hour label shift for early prediction

## 📓 Notebooks
The `notebooks/` directory contains the original competition analysis:

```bash
notebooks/
└── main.ipynb  # training and testing complete notebook
```

To launch:
```bash
jupyter lab notebooks/main.ipynb
```

## 🤝 Contributing
1. Fork the repository
2. Create feature branch:  
   `git checkout -b feature/new-feature`
3. Commit changes:  
   `git commit -m 'Add awesome feature'`
4. Push to branch:  
   `git push origin feature/new-feature`
5. Open a Pull Request

## 📜 License
MIT License - See [LICENSE](LICENSE) for details




## ❤️ Acknowledgments
- Hospital Sant Joan de Déu for data access
- Kaggle community for technical inspiration
- Open-source ML maintainers


