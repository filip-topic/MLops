import pandas as pd
import numpy as np
#import mlflow
from prefect import flow, task
from scipy.stats import entropy

@task
def load_data(n_test=2000):
    """
    Loads the dataset and splits it into training and held-out test sets.
    The last n_test rows are used as the test set to simulate unseen, post-deployment data.
    """
    df = pd.read_csv('data/Womens Clothing E-Commerce Reviews.csv')
    test_df = df.tail(n_test)
    train_df = df.iloc[:-n_test]
    return train_df, test_df

@task
def compute_kl_divergence(train_df, test_df, feature='Age', bins=20):
    """
    Computes the KL divergence between the distributions of a feature in the training and test sets.
    KL divergence is a standard measure of distributional drift.
    """
    train_hist, bin_edges = np.histogram(train_df[feature].dropna(), bins=bins, density=True)
    test_hist, _ = np.histogram(test_df[feature].dropna(), bins=bin_edges, density=True)
    # Add small value to avoid division by zero
    train_hist += 1e-8
    test_hist += 1e-8
    kl_div = entropy(test_hist, train_hist)
    print(f"KL divergence for '{feature}' between training and test: {kl_div:.4f}")
    # Document expectation: KL close to 0 means no drift; higher values indicate drift.
    return kl_div

@flow(name="Monitoring Drift Detection Flow")
def monitoring_flow(n_test: int = 2000):
    """
    Monitoring flow for post-deployment drift detection.
    - Uses the last n_test rows as unseen data.
    - Performs KL divergence drift test on 'Age'.
    """
    train_df, test_df = load_data(n_test)
    kl_div = compute_kl_divergence(train_df, test_df, feature='Age')
    # You can add more features or model prediction drift here as needed.

if __name__ == "__main__":
    monitoring_flow() 