from functools import wraps
import pandas as pd
from typing import Callable
import matplotlib.pyplot as plt
import networkx as nx


class DataFlowTracker:
    def __init__(self):
        self.steps = []
        self.counts = {}

    def track(self, step_name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
                initial_count = len(df)
                result_df = func(df, *args, **kwargs)
                final_count = len(result_df)
                removed = initial_count - final_count

                self.steps.append(step_name)
                self.counts[step_name] = {
                    "initial": initial_count,
                    "final": final_count,
                    "removed": removed,
                }
                return result_df

            return wrapper

        return decorator

    def generate_flowchart(self, title: str = "Data Processing Flow"):
        G = nx.DiGraph()
        pos = {}
        labels = {}

        # Create nodes and edges
        for i, step in enumerate(self.steps):
            node_current = f"step_{i}"
            counts = self.counts[step]

            # Create node labels with counts
            labels[node_current] = (
                f"{step}\nInitial: {counts['initial']}\nRemoved: {counts['removed']}"
            )
            pos[node_current] = (0, -i)  # Vertical layout

            G.add_node(node_current)
            if i > 0:
                G.add_edge(f"step_{i-1}", node_current)

        # Plot
        plt.figure(figsize=(10, 2 * len(self.steps)))
        nx.draw(
            G,
            pos,
            labels=labels,
            node_color="lightblue",
            node_size=3000,
            font_size=8,
            font_weight="bold",
            arrows=True,
            edge_color="gray",
        )
        plt.title(title)
        return plt
