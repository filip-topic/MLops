from great_expectations.dataset import PandasDataset
import pandas as pd


class ReviewDataset(PandasDataset):
    def expect_column_to_have_missing_values_less_than_threshold(self, column, threshold):
        missing_count = self[column].isnull().sum()
        total_count = len(self)
        missing_percentage = missing_count / total_count
        return {
            "success": missing_percentage < threshold,
            "result": {"missing_percentage": missing_percentage}
        }


def run_missing_values_test():
    df = pd.read_csv('data/Womens Clothing E-Commerce Reviews.csv')
    dataset = ReviewDataset(df)

    pct = 0.4

    assert dataset.expect_column_to_have_missing_values_less_than_threshold("Review Text", pct)["success"]
    print(f"Dataset has less than {pct * 100}% of missing values in the 'Review' column")
    assert dataset.expect_column_to_have_missing_values_less_than_threshold("Title", pct)["success"]
    print(f"Dataset has less than {pct * 100}% of missing values in the 'Title' column")

    with open("missing_values_flag.txt", "w") as f:
        pass  # This creates the file and closes it without writing anything
