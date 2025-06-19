# 'Women's Clothing E-Commerce Reviews' dataset ML flow

## Dataset 
'Women's Clothing E-Commerce Reviews' (https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) is a Kaggle dataset which revolves around E-commerce reviews written by customers of a specific online store. This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review and contains variables of varying types eg. 'Age' (int), 'Review Text' (String), Rating (int / ordered categorical), 'Recommended IND' (binary categorical).

## Pipeline Overview

The pipeline is orchestrated using Prefect and consists of three main steps:
1. **Data Quality Tests**: Runs data validation scripts to check for missing values and distribution expectations.
2. **Model Training**: Trains a logistic regression model to predict `Recommended IND` using selected features, with configuration and validation checks.
3. **Model Robustness Validation**: Validates the trained model's robustness to small input perturbations.

All models and metadata are versioned and tracked using MLflow (local file store). Configuration for required columns and training parameters is managed via `config.yaml` in the root directory.

## Data Splitting Strategy

### Unseen Data Segment for Testing
The project implements a deterministic data splitting strategy based on the `Clothing ID` attribute:

- **Training Data**: All rows with **even** `Clothing ID` values
- **Test Data**: All rows with **odd** `Clothing ID` values

### Rationale for This Approach
1. **Deterministic and Reproducible**: The split is based on a stable identifier, ensuring consistent train/test partitions across all runs
2. **No Data Leakage**: Training and test sets are completely separate, preventing any overlap
3. **Balanced Distribution**: Provides roughly equal train/test sizes depending on Clothing ID distribution
4. **Consistent Across Components**: All components (training, A/B testing, monitoring) use the same data splitting logic

This approach ensures that drift monitoring and A/B testing are performed on truly unseen data that was never exposed during model training.

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
- The main pipeline is defined in `./training_flow.py` using Prefect. Each step is dockerized separately with its own requirements.txt and Dockerfile. Custom DockerContainer class is created in training_flow.py which is used to execute the steps (tasks) as separate containers.
- Data quality tests are run via `./pre_training_tests/main.py` as a subprocess and .ctl files are written which indicate whether the data has passed these tests.
- Model training is handled by `model/train/train.py`, which checks whether data has passed the data-quality tests from the previous step, loads configuration, trains the model, and logs the model to MLflow.
- After training, the latest model is loaded and validated for robustness by perturbing numeric features and checking for excessive sensitivity. This happens in `./model/validate/validate_robustness.py`

## Flow Versioning with MLflow

### Implementation
The training workflow is versioned using MLflow to track flow-level configurations and parameters:

- **Workflow Tracking**: Uses separate `flow_runs` directory for workflow-level metadata
- **Model Tracking**: Uses `mlruns` directory for model-specific artifacts and parameters
- **Configuration Logging**: All workflow parameters (min_training_size, max_iter, random_state) are logged as MLflow parameters
- **Artifact Tracking**: Configuration files and workflow metadata are logged as artifacts

### Benefits of Flow Versioning
1. **Reproducibility**: Complete workflow configurations are tracked and can be reproduced
2. **Experiment Tracking**: Different hyperparameter combinations can be compared
3. **Audit Trail**: Full history of workflow changes and their impact on model performance
4. **Configuration Management**: All important parameters are captured at the flow level, not just in the training code

### Flow Configuration Parameters
- `min_training_size`: Minimum required dataset size for training
- `max_iter`: Maximum iterations for logistic regression
- `random_state`: Random seed for reproducibility
- `training_data_split`: Method used for data splitting (even_clothing_id)
- `training_samples`: Actual number of samples used for training

## Model Versioning and Metadata
- Models are logged to MLflow under the experiment `recommendation-models`.
- All information regarding runs and versions is inside the `./mlruns` folder
- Metadata for each run is in `mlruns/models/recommendation_model/version-{MODEL_VERSION}`
- Models runs are represented in `./mlruns/{random-18digit-number}/{RUN_ID}`
- Each run includes:
  - The trained model (with input signature and example)
  - Model parameters
  - `requirements.txt` as an artifact
  - Training data split information
  - Model-specific features and configurations
- MLflow's local file store is used for experiment tracking and model registry.
- Model weights are in `./mlruns/{random-18digit-number}/{RUN_ID}/artifacts/model/model.pkl`
- Model requirements are in `./mlruns/{random-18digit-number}/{RUN_ID}/artifacts/requirements.txt`

## Drift Monitoring

### Implementation
A separate monitoring flow (`monitoring/monitoring_flow.py`) performs drift detection on unseen test data:

1. **Data Loading**: Loads training data (even Clothing IDs) and test data (odd Clothing IDs)
2. **KL Divergence Computation**: Calculates distribution drift between training and test data

### KL Divergence Choice and Reasoning

**Why KL Divergence?**
- **Distribution Comparison**: KL divergence measures how one probability distribution diverges from a reference distribution
- **Sensitive to Changes**: Detects subtle shifts in feature distributions that might indicate data drift
- **Interpretable**: Provides a single metric that can be thresholded for alerting
- **Widely Used**: Industry standard for detecting distributional changes in ML systems

**What It Measures:**
- **Feature Distribution Drift**: How the distribution of individual features (e.g., Age, Rating) changes between training and test data
- **Statistical Significance**: Whether observed differences are likely due to random variation or actual drift
- **Magnitude of Change**: Quantifies the extent of distributional shift

**Expected Behavior and Sourcing:**

**Expected Behavior:**
- **Low KL Divergence (< 0.1)**: Indicates stable data distributions, suggesting the model should perform consistently
- **High KL Divergence (> 0.1)**: Indicates significant distributional changes, suggesting potential model performance degradation

**Reasoning for Expected Behavior:**
1. **Domain Stability**: E-commerce clothing reviews should have relatively stable user demographics and rating patterns over time
2. **Feature Consistency**: Age distributions, rating patterns, and categorical feature distributions should remain consistent for a stable business
3. **Model Assumptions**: The model assumes training and test data come from similar distributions; significant drift violates this assumption

**Threshold Justification:**
- **0.1 threshold**: Based on empirical observations that KL divergence values below 0.1 typically indicate acceptable distributional similarity
- **Conservative Approach**: Lower threshold ensures early detection of potential issues
- **Actionable Alerts**: Values above threshold trigger investigation and potential model retraining

## A/B Testing Framework

### Implementation
The A/B testing framework (`ab_test/ab_test_flow.py`) compares different model versions:

1. **Data Loading**: Loads test data (odd Clothing IDs) for evaluation
2. **Model Retrieval**: Gets the latest two model run IDs (arbitrary choice) from MLflow
3. **Prediction Generation**: Runs predictions using each model version
4. **Performance Comparison**: Evaluates and compares model performance

### A/B Testing Process
1. **Model Selection**: Automatically retrieves the two most recent model versions
2. **Configuration Loading**: Loads model-specific configurations (parameters, features) from MLflow runs
3. **Prediction Execution**: Generates predictions using each model on the same test dataset
4. **Performance Evaluation**: Compares accuracy and other metrics between model versions
5. **Comprehensive Reporting**: Provides detailed comparison including model configurations and performance statistics

### Model-Specific Configuration Loading
The A/B testing framework loads model-specific configurations directly from MLflow runs:
- **Parameters**: Retrieved from `./mlruns/experiment_id/run_id/params`
- **Features**: Retrieved from `./mlruns/experiment_id/run_id/artifacts/model/input_example.json`
- **Model Artifacts**: Loaded using MLflow's model registry

This ensures that each model is evaluated using its exact training configuration, providing accurate and fair comparisons.

### Multiple A/B Tests Handling

**Hypothetical Approach for Multiple Concurrent/Subsequent Tests:**

1. **Test Identification**:
   - **Test IDs**: Assign unique identifiers to each A/B test (e.g., `ab_test_2024_01`, `ab_test_2024_02`)
   - **Timestamp-based Naming**: Use timestamps to distinguish between tests
   - **Purpose-based Naming**: Include test purpose in identifier (e.g., `feature_comparison`, `hyperparameter_tuning`)

2. **Model Version Management**:
   - **Explicit Version Selection**: Instead of "latest two", specify exact model versions for each test
   - **Version Tagging**: Tag models with test-specific labels in MLflow
   - **Test Metadata**: Store test configuration and model versions in test-specific artifacts

3. **Result Isolation**:
   - **Separate Artifact Directories**: Create test-specific output directories
   - **Test-specific Metrics**: Store results with test identifiers
   - **Independent Evaluation**: Ensure tests don't interfere with each other

4. **Operational Considerations**:
   - **Test Scheduling**: Implement test queuing and scheduling mechanisms
   - **Resource Management**: Ensure sufficient computational resources for concurrent tests
   - **Result Aggregation**: Maintain a central registry of all A/B test results

**Implementation Strategy:**
```python
# Example approach for multiple tests
def ab_test_flow(test_id: str, model_version_a: str, model_version_b: str):
    # Use test-specific directories
    test_artifact_dir = f"artifacts/ab_test/{test_id}"
    # Store test metadata
    test_config = {
        "test_id": test_id,
        "model_a": model_version_a,
        "model_b": model_version_b,
        "timestamp": datetime.now().isoformat()
    }
    # Execute test with isolated resources
```

## Error Handling
- The pipeline (step 2: `model/train/train.py`) checks for:
  - Presence of all required columns (as specified in `config.yaml`)
  - Sufficient dataset size (minimum configurable in `config.yaml`)
  - File errors and unexpected exceptions
- Clear error messages are printed and surfaced in the Prefect flow if any validation fails.

## Robustness Expectation and Rationale

We define robustness as the model's ability to maintain stable output probabilities under small, realistic perturbations to numeric input features. Specifically, we add ±5% Gaussian noise to features like `Age`, `Rating`, and `Positive Feedback Count`, and compute the RMSE between the model's predicted probabilities on the original and perturbed inputs.

A threshold of **0.1 RMSE** is used to flag excessive sensitivity. This captures undesirable behavior where minor input variations cause disproportionate output shifts—common in overfitted or unstable models.

This approach aligns with robustness evaluation practices in real-world ML systems, where performance drift under noise is a key indicator of generalization.

## Configuration
- All key parameters (required columns, minimum training size, error handling) are set in `./config.yaml` in the root directory.
- Flow-level configurations are tracked in MLflow for versioning and reproducibility.

## How to Run the ML Pipeline

### Main Training Pipeline
- run "chmod +x run.sh" in bash to give permission for the run.sh file that is provided
- run "./run.sh" script

./run.sh
- builds the Dockerfiles for each step in the flow (3) and r
- installs minimalist requirements.txt which are needed to run the flow script (`./training_flow.py`)
- runs the flow (`./training_flow.py`)

### Drift Monitoring
```bash
# Build monitoring images
docker build -t monitoring-load-data:latest monitoring/tasks/load_data/
docker build -t monitoring-kl-div:latest monitoring/tasks/compute_kl_divergence/

# Run monitoring flow
python monitoring/monitoring_flow.py
```

### A/B Testing
```bash
# Build A/B testing images
docker build -t ab-load_test_data:latest ab_test/tasks/load_test_data/
docker build -t ab-get_latest_two_run_ids:latest ab_test/tasks/get_latest_two_run_ids/
docker build -t ab-run_predictions:latest ab_test/tasks/run_predictions/
docker build -t ab-evaluate_ab:latest ab_test/tasks/evaluate_ab/

# Run A/B test flow
python ab_test/ab_test_flow.py
```

## Project Structure
- `pre_training_tests/`: Data quality tests
- `model/train/train.py`: Model training and MLflow logging
- `model/validate/validate_robustness.py`: Model validation
- `training_flow.py`: Prefect pipeline with MLflow versioning
- `monitoring/monitoring_flow.py`: Drift monitoring pipeline
- `ab_test/ab_test_flow.py`: A/B testing pipeline
- `config.yaml`: Configuration for required columns and training parameters
- `mlruns/`: MLflow experiment tracking and model registry
- `flow_runs/`: MLflow workflow versioning and tracking
- `data/`: Input data
- `artifacts/`: Output artifacts from monitoring and A/B testing

## Expectation definitions

### Missing values in 'Review' and 'Title' columns
'Review' and 'Title' are qualitative fields provided by users. It's common to see some missing entries, especially for optional fields. In industry and academic standards, a column with >30% missing data is often considered unreliable for direct use without imputation or deeper inspection.

However, the nature of this dataset is such that 'Review' and 'Title' columns tend to be sparse as the users of online platforms don't have to write a review or a title in order to be able to leave a a rating (1-5 stars) - therefore it is natural to see such datasets with a lot of missing values in those columns. To approximate the reasonable expectation for how many missing values are permissible in the 'Title' and 'Review' columns, we can take the percentage of all reviews on Google with a star rating but no text - latest figure being 54.2% is 2022 according to https://www.soci.ai/insights/state-of-google-reviews/. This means that 45.8% of reviews DO NOT have a text. However, this statistic is too general; applying to all kinds of businesses that are listed on Google - most of which are likely to have less loyal and engaging customers (eg. supermarkets, coffeshops or chain restaurants like McDonalds) than a specific brand of women's clothing (which this dataset represents). We don't have data to back this up, but we can estimate that due to this factor, we can reasonably expect that the percentage of reviews with no text (i.e. missing values in 'Review' and 'Title' columns) would be around 35%. If we assume the reviews without text to be distributed according to the binomial distribution with p=35% and N being a large number, we can say it is VERY unlikely that the data would have > 40% of missing values in the 'Rating' and 'Title' columns. At confidence level of 5% and small N=246, the critical upper bound for p is 40%. In reality this statistic is likely to be based on a much higher N, therefore the 40% upper bound would correspond to a MUCH lower confidence level. Therefore the figure of 40% is chosen for the threshold.

### Expectation of distribution for 'Age' and 'Rating' attributes

Assuming the company is in the US, it is reasonable to also assume that a person shopping (and leaving Reviews) online is no younger than 13. This is because of the Children's Online Privacy Protection Act (COPPA) which is a federal law restricting the collection of personal information from children under 13 without verifiable parental consent. As a result, many online platforms prohibit users under 13 from creating accounts. Oldest living person (until October 2024) in the US was Elizabeth Francis according to CNBC (https://www.cnbc.com/2024/07/31/the-oldest-living-person-in-the-us-just-turned-115.html). Due to these facts it is reasonable to expect that the values of the 'Age' attribute lie in the interval [13, 115].

Since the nature of the 'Rating' attribute is such that it is between 1 and 5 (inclusive) by definition, it is certain that the values in this column should be exactly between (and including) 1 and 5.

