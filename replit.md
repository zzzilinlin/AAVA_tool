# AAVA Tool - Dutch Harmfulness Classification

## Overview
AAVA is a machine learning pipeline for classifying Dutch text content by harmfulness level. The tool trains two types of models:
1. **General Models** - Classify harmfulness based solely on text content (sentence)
2. **Feature-Aware Models** - Classify harmfulness using text + demographic features to predict how different groups perceive harmfulness

## Recent Changes (December 11, 2025)
- Implemented dual model approach: General (text-only) vs Feature-Aware (text + demographics)
- Updated for 3-class harmfulness classification: "low", "medium", "high"
- New data format with `sentence` and `article` columns
- Demographic features: gender, age, education_level, ethnicity, nationality
- TF-IDF General model achieves 73% accuracy on dummy data (best performer)
- SBERT using paraphrase-multilingual-mpnet-base-v2 (768-dim embeddings)

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

## Classification Labels (3-class harmfulness)
- **low** - Not harmful or minimally harmful content
- **medium** - Moderately harmful content
- **high** - Highly harmful content

## Models Trained (4 total)

### General Models (Text Only)
1. **TF-IDF General** - TF-IDF + Logistic Regression using only sentence text
2. **SBERT General** - Sentence-BERT embeddings + Logistic Regression using only sentence text

### Feature-Aware Models (Text + Demographics)
3. **TF-IDF Feature-Aware** - TF-IDF + demographics (gender, age, education, ethnicity, nationality)
4. **SBERT Feature-Aware** - SBERT embeddings + demographics

## Feature Set
- **Text column**: `sentence` (Dutch text to classify)
- **Article context**: `article` (full article context, available for future use)
- **Categorical predictors**: `gender`, `education_level`, `ethnicity`, `nationality` (one-hot encoded)
- **Numeric predictor**: `age` (scaled)
- **Target**: `harmfulness_level` (3-class: low/medium/high)

## Configuration
Edit `conf/base/parameters.yml` to adjust:
- `sample_size`: Number of rows to sample for testing (null = full dataset)
- `test_size`: Train/test split ratio (default: 0.2)
- `text_column`: Column containing text to classify (default: "sentence")
- `label_column`: Target column (default: "harmfulness_level")
- `cat_columns`: List of categorical feature columns
- `num_columns`: List of numeric feature columns
- `tfidf`: max_features, ngram_range, min_df
- `sbert`: model_name, batch_size
- `logistic_regression`: max_iter, C, class_weight

## Dependencies
- Python 3.11+
- Kedro
- scikit-learn
- pandas, numpy
- sentence-transformers (for SBERT models)

## Data Format
Input CSV file in `data/01_raw/`:
- `dummy_data_preprocessed.csv`: Contains columns:
  - `sentence`: Text to classify
  - `article`: Full article context
  - `harmfulness_level`: Target label (low/medium/high)
  - `gender`: Annotator gender
  - `age`: Annotator age
  - `education_level`: Annotator education
  - `ethnicity`: Annotator ethnicity
  - `nationality`: Annotator nationality

## Current Performance (1000 sample)

### General Models (Text Only)
- **TF-IDF General**: 73% accuracy, 0.72 F1 (weighted) ⭐ Best
- **SBERT General**: 57.5% accuracy, 0.62 F1 (weighted)

### Feature-Aware Models (Text + Demographics)
- **TF-IDF Feature-Aware**: 66% accuracy, 0.68 F1 (weighted)
- **SBERT Feature-Aware**: 60% accuracy, 0.64 F1 (weighted)

Note: On the dummy data, general (text-only) models outperform feature-aware models, suggesting the harmfulness labels in the dummy data are primarily text-driven rather than demographic-driven.
