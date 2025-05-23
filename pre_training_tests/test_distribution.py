from great_expectations.dataset import PandasDataset
import pandas as pd



class ReviewDataset(PandasDataset):
    def expect_column_values_to_be_between_with_reasonable_range(self, column, min_value, max_value):
        return self.expect_column_values_to_be_between(column, min_value=min_value, max_value=max_value)


def run_distribution_test():
    df = pd.read_csv('data/Womens Clothing E-Commerce Reviews.csv')
    dataset = ReviewDataset(df)

    # testing Age
    assert dataset.expect_column_values_to_be_between_with_reasonable_range("Age", 13, 115)["success"]
    print(f"All values in the 'Age' column are within reasonable bounds")

    # Test Rating Distribution: Ratings should be between 1 and 5
    assert dataset.expect_column_values_to_be_between_with_reasonable_range("Rating", 1, 5)["success"]
    print(f"All values in the 'Rating' column are within reasonable bounds")

    with open("distribution_flag.txt", "w") as f:
        pass  # This creates the file and closes it without writing anything
