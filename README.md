# 'Women's Clothing E-Commerce Reviews' dataset quality test

## Dataset 
'Women's Clothing E-Commerce Reviews' (https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) is a Kaggle dataset which revolves around E-commerce reviews written by customers of a specific online store. This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review and contains variables of varying types eg. 'Age' (int), 'Review Text' (String), Rating (int / ordered categorical), 'Recommended IND' (binary categorical).

## Pipeline Overview

The pipeline is orchestrated using Prefect and consists of three main steps:
1. **Data Quality Tests**: Runs data validation scripts to check for missing values and distribution expectations.
2. **Model Training**: Trains a logistic regression model to predict `Recommended IND` using selected features, with configuration and validation checks.
3. **Model Robustness Validation**: Validates the trained model's robustness to small input perturbations.

All models and metadata are versioned and tracked using MLflow (local file store). Configuration for required columns and training parameters is managed via `model/config.yaml`.

## Features Used for Training
- **Numeric:**
  - Age
  - Rating
  - Positive Feedback Count
- **Categorical:**
  - Division Name
  - Department Name
  - Class Name
- **Target:**
  - Recommended IND

## Orchestration and Flow
- The main pipeline is defined in `flows/training_flow.py` using Prefect.
- Data quality tests are run via `pre_training_tests/main.py` as a subprocess.
- Model training is handled by `model/train_model.py`, which loads configuration, validates data, and logs the model to MLflow.
- After training, the latest model is loaded and validated for robustness by perturbing numeric features and checking for excessive sensitivity.

## Model Versioning and Metadata
- Models are logged to MLflow under the experiment `recommendation-models`.
- Each run includes:
  - The trained model (with input signature and example)
  - Model parameters
  - `requirements.txt` as an artifact
- MLflow's local file store is used for experiment tracking and model registry.

## Error Handling
- The pipeline checks for:
  - Presence of all required columns (as specified in `model/config.yaml`)
  - Sufficient dataset size (minimum configurable in `model/config.yaml`)
  - File errors and unexpected exceptions
- Clear error messages are printed and surfaced in the Prefect flow if any validation fails.

## Model Robustness Validation
- After training, the pipeline perturbs numeric features (±5% noise) and checks the change in predicted probabilities.
- If the root mean squared error (RMSE) of probability drift exceeds 0.1, the model is flagged as too sensitive.
- This helps ensure the model is robust to small input changes and not overfitted.

## Configuration
- All key parameters (required columns, minimum training size, error handling) are set in `model/config.yaml`.

## How to Run the ML Pipeline

1. **Build the Docker image:**
   ```PS
   docker build -t pre_deployment_test .
   ```
2. **Run the pipeline:**
   ```PS
   docker run -v "${PWD}/mlruns:/app/mlruns" pre-deployment-test
   ```
   This will:
   - Run Task 1 data tests
   - Train and serialize a model (if enough data)
   - Version the model and save metadata
   - Validate model robustness on edge cases

### Local Development
- You can run the flow locally with:
  ```bash
  python flows/training_flow.py
  ```
- Model training can be run directly with:
  ```bash
  python model/train_model.py
  ```

### Requirements
- All dependencies are listed in `requirements.txt`.
- Prefect, joblib, scikit-learn, pandas, mlflow, and pyyaml are required for orchestration, model handling, and configuration.

### Directory Structure
- `model/train_model.py`: Model training and MLflow logging
- `flows/training_flow.py`: Prefect pipeline
- `pre_training_tests/`: Data quality tests
- `model/config.yaml`: Configuration for required columns and training parameters
- `mlruns/`: MLflow experiment tracking and model registry
- `data/`: Input data

---

This setup ensures your pipeline is robust, versioned, locally reproducible, and meets all requirements for your MLOps course.

## Expectation definitions

### Missing values in 'Review' and 'Title' columns
'Review' and 'Title' are qualitative fields provided by users. It's common to see some missing entries, especially for optional fields. In industry and academic standards, a column with >30% missing data is often considered unreliable for direct use without imputation or deeper inspection.

However, the nature of this dataset is such that 'Review' and 'Title' columns tend to be sparse as the users of online platforms don't have to write a review or a title in order to be able to leave a a rating (1-5 stars) - therefore it is natural to see such datasets with a lot of missing values in those columns. To approximate the reasonable expectation for how many missing values are permissible in the 'Title' and 'Review' columns, we can take the percentage of all reviews on Google with a star rating but no text - latest figure being 54.2% is 2022 according to https://www.soci.ai/insights/state-of-google-reviews/. This means that 45.8% of reviews DO NOT have a text. However, this statistic is too general; applying to all kinds of businesses that are listed on Google - most of which are likely to have less loyal and engaging customers (eg. supermarkets, coffeshops or chain restaurants like McDonalds) than a specific brand of women's clothing (which this dataset represents). We don't have data to back this up, but we can estimate that due to this factor, we can reasonably expect that the percentage of reviews with no text (i.e. missing values in 'Review' and 'Title' columns) would be around 35%. If we assume the reviews without text to be distributed according to the binomial distribution with p=35% and N being a large number, we can say it is VERY unlikely that the data would have > 40% of missing values in the 'Rating' and 'Title' columns. At confidence level of 5% and small N=246, the critical upper bound for p is 40%. In reality this statistic is likely to be based on a much higher N, therefore the 40% upper bound would correspond to a MUCH lower confidence level. Therefore the figure of 40% is chosen for the threshold.

### Expectation of distribution for 'Age' and 'Rating' attributes

Assuming the company is in the US, it is reasonable to also assume that a person shopping (and leaving Reviews) online is no younger than 13. This is because of the Children's Online Privacy Protection Act (COPPA) which is a federal law restricting the collection of personal information from children under 13 without verifiable parental consent. As a result, many online platforms prohibit users under 13 from creating accounts. Oldest living person (until October 2024) in the US was Elizabeth Francis according to CNBC (https://www.cnbc.com/2024/07/31/the-oldest-living-person-in-the-us-just-turned-115.html). Due to these facts it is reasonable to expect that the values of the 'Age' attribute lie in the interval [13, 115].

Since the nature of the 'Rating' attribute is such that it is between 1 and 5 (inclusive) by definition, it is certain that the values in this column should be exactly between (and including) 1 and 5.

