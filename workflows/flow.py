import os
from prefect import flow, task, get_run_logger
from src.model.train import train_and_serialize_model
from src.model.versioning import list_models, load_model_by_id
from src.model.robustness import check_model_robustness

MODEL_DIR = os.path.join(os.getcwd(), 'model')
DATA_PATH = os.path.join(os.getcwd(), 'data', 'Womens Clothing E-Commerce Reviews.csv')

@task
def data_tests():
    # Run the Task 1 data tests
    from expectations.test_missing_values import run_missing_values_test
    from expectations.test_distribution import run_distribution_test
    run_missing_values_test()
    run_distribution_test()

@task
def train_model_task():
    return train_and_serialize_model(DATA_PATH)

@task
def robustness_check_task(model_id, model_path, metadata):
    meta_path = os.path.join(model_path, "metadata.json")
    model_file = os.path.join(model_path, "model.joblib")
    return check_model_robustness(model_file, meta_path)

@flow
def main_flow():
    logger = get_run_logger()
    logger.info("Step 1: Running data tests...")
    data_tests()
    logger.info("Step 2: Training model...")
    try:
        model_id, model_path, metadata = train_model_task()
    except ValueError as e:
        logger.error(f"Model training failed: {e}")
        return
    logger.info(f"Model trained and saved at {model_path}")
    logger.info("Step 3: Checking model robustness...")
    robustness = robustness_check_task(model_id, model_path, metadata)
    logger.info(f"Robustness check: {robustness}")

if __name__ == "__main__":
    main_flow()
