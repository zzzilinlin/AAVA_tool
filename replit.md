# AAVA Tool - Harmfulness Classification

## Overview
AAVA (Automated Article Vulnerability Assessment) is a machine learning project for classifying text content by harmfulness level. The tool trains and compares multiple ML models to predict whether content contains harmful biases or stereotypes related to ethnicity, sexual orientation, religion, or other sensitive attributes.

## Project Structure
```
.
├── conf/                    # Kedro configuration files
│   ├── base/               # Base configuration
│   │   ├── catalog.yml     # Data catalog definitions
│   │   └── parameters.yml  # Model and pipeline parameters
│   └── local/              # Local overrides (gitignored)
├── data/                   # Data directories (Kedro data engineering convention)
│   ├── 01_raw/            # Raw input data
│   ├── 02_intermediate/   # Preprocessed data
│   ├── 03_primary/        # Train/test splits
│   ├── 06_models/         # Trained model files
│   ├── 07_model_output/   # Model predictions/evaluations
│   └── 08_reporting/      # Reports and summaries
├── notebooks/             # Jupyter notebooks and sample data
├── scripts/               # Utility scripts
├── src/aava_tool/         # Main source code
│   └── pipelines/        # Kedro pipelines
│       ├── data_processing/  # Data loading and preprocessing
│       ├── model_training/   # Train multiple classifiers
│       └── model_evaluation/ # Evaluate and compare models
└── tests/                 # Unit tests

## Running the Pipeline

### Full Pipeline (Data Processing + Training + Evaluation)
```bash
kedro run
# or
python main.py
```

### Individual Pipelines
```bash
# Data processing only
kedro run --pipeline data_processing

# Training only
kedro run --pipeline model_training

# Evaluation only  
kedro run --pipeline model_evaluation

# Training without evaluation
kedro run --pipeline train
```

## Classification Labels
- **Not Harmful** (0): Content with no harmful biases
- **Slightly Harmful** (1): Content with minor bias indicators
- **Harmful** (2): Content with notable harmful stereotypes
- **Very Harmful** (3): Content with severe harmful biases

## Models Trained
1. **Logistic Regression** - Fast, interpretable baseline
2. **Random Forest** - Ensemble with feature importance
3. **Gradient Boosting** - High-accuracy ensemble
4. **SVM** - Good for text classification
5. **Naive Bayes** - Probabilistic baseline

## Configuration
Edit `conf/base/parameters.yml` to adjust:
- `test_size`: Train/test split ratio (default: 0.2)
- `tfidf`: Text vectorization parameters
- Model-specific hyperparameters

## Dependencies
- Python 3.11+
- Kedro 1.0.0
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

## Data Format
Input CSV should have columns:
- `text`: Article/content text
- `category`: Harmfulness label (optional, for training)
- `justification`: Annotation notes (optional)

## Adding New Data
1. Place CSV files in `data/01_raw/`
2. Update `conf/base/catalog.yml` to reference new files
3. Run the pipeline

## Environment Variables
- `GOOGLE_GENAI_API_KEY`: For LLM-based annotation (optional)
