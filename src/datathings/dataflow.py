from pydantic import BaseModel
import os
from typing import List
import json
import pandas as pd
from functools import wraps


class StepData(BaseModel):
    step_name: str
    initial_count: int
    final_count: int
    removed_count: int


class DataFlowTrackerModel(BaseModel):
    steps: List[StepData]

    def to_json(self, filename: str):
        """Save the data flow tracking information to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @staticmethod
    def from_json(filepath: str):
        """Load the data flow tracking information from a JSON file. Might not need."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return DataFlowTrackerModel.model_validate(data)


class DataFlowTracker:
    def __init__(self):
        self.steps = []

    def track(self, step_name: str):
        """Decorator to track steps in the data processing pipeline."""

        def decorator(func):
            @wraps(func)
            def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
                initial_count = len(df)
                result_df = func(df, *args, **kwargs)
                final_count = len(result_df)
                removed = initial_count - final_count

                # Record step data
                self.steps.append(
                    StepData(
                        step_name=step_name,
                        initial_count=initial_count,
                        final_count=final_count,
                        removed_count=removed,
                    )
                )
                return result_df

            return wrapper

        return decorator

    def export_to_model(self) -> DataFlowTrackerModel:
        """Export the tracked steps as a pydantic model."""
        return DataFlowTrackerModel(steps=self.steps)


if __name__ == "__main__":
    # Example usage
    tracker = DataFlowTracker()

    @tracker.track("Remove Duplicates")
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    @tracker.track("Remove Nulls")
    def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    @tracker.track("Filter Outliers")
    def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["value"] < 100]

    data = {
        "name": ["Alice", "Bob", "Alice", "Alice", "Charlie", "Bob"],
        "value": [10, 20, 10, 30, 100, 20],
    }
    df = pd.DataFrame(data)

    df_processed = remove_duplicates(df)
    df_processed = remove_nulls(df_processed)
    df_processed = filter_outliers(df_processed)

    data_flow_model = tracker.export_to_model()
    data_flow_model.to_json("data_flow_tracking.json")

    loaded_model = DataFlowTrackerModel.from_json("data_flow_tracking.json")
    print(loaded_model)
