# AAVA Tool - Dutch Opinion/Fact Classification

## Overview
AAVA is a machine learning pipeline for classifying Dutch text content on an opinion-to-fact scale. The tool uses text features (TF-IDF, SBERT embeddings) combined with demographic/political predictors to classify sentences into 6 categories from "Absoluut een mening" (absolute opinion) to "Absoluut feitelijk" (absolute fact).

## Recent Changes (December 4, 2025)
- Implemented TF-IDF + Logistic Regression with Dutch stopwords
- Implemented SBERT + Logistic Regression using distiluse-base-multilingual-cased-v2 embeddings
- Added multimodal feature fusion: text features + categorical (one-hot) + numeric (scaled)
- Added `sample_size` parameter for faster pipeline testing (default: 500 rows)
- Pipeline runs end-to-end in ~28 seconds on sampled data
- Fixed git issues: added checkpoints/ to .gitignore

## Project Structure
```
.
├── conf/                    # Kedro configuration files
│   ├── base/               # Base configuration
│   │   ├── catalog.yml     # Data catalog definitions
│   │   └── parameters.yml  # Model and pipeline parameters
│   └── local/              # Local overrides (gitignored)
├── data/                   # Data directories (Kedro convention)
│   ├── 01_raw/            # Raw input data (CSV files)
│   ├── 02_intermediate/   # Preprocessed data
│   ├── 03_primary/        # Train/test splits
│   ├── 06_models/         # Trained model files
│   ├── 07_model_output/   # Model predictions/evaluations
│   └── 08_reporting/      # Reports and summaries
├── src/aava_tool/         # Main source code
│   └── pipelines/        # Kedro pipelines
│       ├── data_processing/  # Data loading and preprocessing
│       ├── model_training/   # Train classifiers
│       └── model_evaluation/ # Evaluate and compare models
└── tests/                 # Unit tests
```

## Running the Pipeline

### Full Pipeline
```bash
kedro run
# or
python main.py
```

### Individual Pipelines
```bash
kedro run --pipeline data_processing
kedro run --pipeline model_training
kedro run --pipeline model_evaluation
```

## Classification Labels (6-class scale)
1. **Absoluut een mening** - Absolute opinion
2. **Overwegend een mening** - Predominantly opinion
3. **Gedeeltelijk een mening** - Partially opinion
4. **Gedeeltelijk feitelijk** - Partially factual
5. **Overwegend feitelijk** - Predominantly factual
6. **Absoluut feitelijk** - Absolute fact

## Models Trained
1. **TF-IDF + Logistic Regression** - Dutch stopwords, bigrams (1,2), combined with categorical/numeric features (600-10,000 features)
2. **SBERT + Logistic Regression** - distiluse-base-multilingual-cased-v2 embeddings (512-dim) combined with categorical/numeric features (536 features)

## Feature Set
- **Text column**: `sentence` (Dutch text)
- **Categorical predictors**: `geslacht`, `opleiding`, `politiek_int`, `politiek_pos` (one-hot encoded)
- **Numeric predictor**: `leeftijd.jaar` (scaled)
- **Target**: `value` (6-class opinion/fact scale)

## Configuration
Edit `conf/base/parameters.yml` to adjust:
- `sample_size`: Number of rows to sample for testing (null = full dataset ~84k rows)
- `test_size`: Train/test split ratio (default: 0.2)
- `tfidf`: max_features, ngram_range, min_df
- `sbert`: model_name, batch_size
- `logistic_regression`: max_iter, C, class_weight

## Dependencies
- Python 3.11+
- Kedro
- scikit-learn
- pandas, numpy
- sentence-transformers (for SBERT model)

## Data Format
Input CSV files in `data/01_raw/`:
- `sentences.csv`: `id`, `sentence`
- `raw_annotation_data.csv`: `coder`, `unit_id`, `value`
- `participant_data.csv`: `coder`, `geslacht`, `leeftijd.jaar`, `opleiding`, `politiek_int`, `politiek_pos`

## Current Performance (500 sample)
- TF-IDF + LogReg: 22% accuracy, 0.22 F1 (weighted)
- SBERT + LogReg: 19% accuracy, 0.20 F1 (weighted)
- Random baseline: ~16.7% (6 classes)

Performance expected to improve significantly with full dataset training.
