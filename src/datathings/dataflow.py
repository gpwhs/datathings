import functools
import pandas as pd
from graphviz import Digraph
from typing import Tuple, List, Optional


class DataTracker:
    """
    A 'branching' data tracker that:
      • Creates a single Graphviz diagram.
      • Uses a dictionary to map each DataFrame object to its node ID.
      • Can handle a function returning multiple DataFrames (branching).
      • Allows customization of node/edge styling.
      • Can skip certain steps if track=False.
    """

    def __init__(
        self,
        name="DataFlow",
        default_node_style: Optional[dict] = None,
        default_edge_style: Optional[dict] = None,
    ):
        """
        Initialize a DataTracker with optional default styles for nodes and edges.
          - default_node_style might be something like: {"shape": "box", "style": "filled", "fillcolor": "lightblue"}
          - default_edge_style might be something like: {"style": "dotted", "color": "gray"}
        """
        self.graph = Digraph(name=name)
        self.node_count = 0
        # Map from "id of a DataFrame object" -> "node ID in the graph"
        self.df_to_node = {}

        # Store the default node & edge attributes
        self.default_node_style = default_node_style or {}
        self.default_edge_style = default_edge_style or {}

    def _new_node_id(self) -> str:
        """Generates a simple unique ID for a node."""
        self.node_count += 1
        return f"node_{self.node_count}"

    def _get_or_create_node_for_df(
        self, df: pd.DataFrame, label_if_new: str, node_style: Optional[dict] = None
    ) -> str:
        """
        If we already have a node for this DataFrame, return it.
        Otherwise, create a new node with the specified label and styling.
        """
        df_id = id(df)
        if df_id not in self.df_to_node:
            new_id = self._new_node_id()
            self.df_to_node[df_id] = new_id

            # Combine default node style with any function-specific node style
            merged_node_style = dict(self.default_node_style)
            if node_style:
                merged_node_style.update(node_style)

            # Graphviz's .node() method can take style params as keyword args.
            # e.g. .node("id", label="...", shape="box", color="blue", style="filled")
            self.graph.node(new_id, label=label_if_new, **merged_node_style)

        return self.df_to_node[df_id]

    def track(
        self,
        stage_label: str = None,
        reason: str = None,
        output_labels: Optional[List[str]] = None,
        show_edge_label: bool = True,
        track: bool = True,  # <--- track boolean, if false, do not add to diagram
        node_style: Optional[dict] = None,  # <--- custom node styling for this step
        edge_style: Optional[dict] = None,  # <--- custom edge styling for this step
    ):
        """
        Decorator that can handle:
          • a function returning ONE DataFrame,
          • or multiple DataFrames (e.g. train/test split).

        Arguments:
          - stage_label:    A name for this transformation (e.g. "Train/Test Split" or "Remove duplicates").
          - reason:         If rows are removed, show the reason in the edge label (unless show_edge_label=False).
          - output_labels:  If multiple DataFrames are returned, optionally provide labels (["Train", "Test"] etc.).
          - show_edge_label: If False, we do not put any label on the edges.
          - track:          If False, do not add anything to the diagram (skip nodes/edges).
          - node_style:     Optional dict of styling attributes for the *output nodes* of this stage.
          - edge_style:     Optional dict of styling attributes for the edges (like {"color": "red", "style": "dotted"}).
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # If track=False, we just skip diagramming but still call the function.
                if not track:
                    return func(*args, **kwargs)

                if not args:
                    raise ValueError(
                        "The tracked function must receive at least one DataFrame as its first argument."
                    )
                if not isinstance(args[0], pd.DataFrame):
                    raise ValueError("The first argument must be a pd.DataFrame.")

                df_in = args[0]
                n_in = len(df_in)

                # Find (or create) a node for the INPUT DataFrame
                input_label = (
                    f"Initial Data\nN={n_in}"
                    if not stage_label
                    else f"{stage_label} Input\nN={n_in}"
                )
                input_node_id = self._get_or_create_node_for_df(
                    df_in, label_if_new=input_label
                )

                # Call the actual function
                result = func(*args, **kwargs)

                # We'll allow either a single DataFrame or a tuple/list of DataFrames
                if isinstance(result, pd.DataFrame):
                    # Single-output scenario
                    df_out_list = [result]
                elif isinstance(result, (list, tuple)) and all(
                    isinstance(x, pd.DataFrame) for x in result
                ):
                    # Multi-output scenario
                    df_out_list = list(result)
                else:
                    raise ValueError(
                        f"Tracked function {func.__name__} returned something that's not a DataFrame "
                        f"or a list/tuple of DataFrames."
                    )

                # For each output DataFrame, create a new node (or re-use if we already tracked that df)
                for i, df_out in enumerate(df_out_list):
                    n_out = len(df_out)

                    # Node label
                    if output_labels and i < len(output_labels):
                        out_label = f"{output_labels[i]}\nN={n_out}"
                    else:
                        # e.g. "Output 1\nN=3" if multiple outputs
                        out_label_main = stage_label or func.__name__
                        if len(df_out_list) > 1:
                            out_label = f"Output {i+1} of {out_label_main}\nN={n_out}"
                        else:
                            out_label = f"{out_label_main}\nN={n_out}"

                    out_node_id = self._get_or_create_node_for_df(
                        df_out, label_if_new=out_label, node_style=node_style
                    )

                    # Build an edge label if we want to show it
                    edge_kwargs = dict(
                        self.default_edge_style
                    )  # Start with global default
                    if edge_style:
                        edge_kwargs.update(
                            edge_style
                        )  # Override with user‐provided style

                    if show_edge_label:
                        removed_count = n_in - n_out
                        if removed_count > 0:
                            if reason:
                                edge_label = (
                                    f"Removed {removed_count}\nReason: {reason}"
                                )
                            else:
                                edge_label = f"Removed {removed_count}"
                        else:
                            edge_label = "No changes"

                        # If multiple outputs & user provided output_labels,
                        # we might want the edge label to show "Train" / "Test" explicitly
                        if output_labels and i < len(output_labels):
                            # E.g. "Train | Removed 10"
                            if removed_count != 0:
                                edge_label = f"{output_labels[i]} | {edge_label}"
                            else:
                                # If no removal, at least label the edge with "Train"/"Test" etc
                                edge_label = output_labels[i]

                        edge_kwargs["label"] = edge_label
                    # If show_edge_label=False, we do not set any label.

                    self.graph.edge(input_node_id, out_node_id, **edge_kwargs)

                return result

            return wrapper

        return decorator


# -------------
# EXAMPLE USAGE
# -------------
def main():
    # Create a DataTracker with some default styles (optional)
    # Can use things like default_node_style, default_edge_style, edge_style, node_style to override
    datatracker = DataTracker(
        default_node_style={
            "shape": "box",
            "style": "filled",
            "fillcolor": "lightgray",
        },
        default_edge_style={"style": "dashed", "color": "blue"},
    )

    @datatracker.track(
        stage_label="Remove Duplicates",
        reason="Duplicate Rows",
        show_edge_label=True,
        node_style={"fillcolor": "yellow"},  # Override node fill for this step
    )
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    @datatracker.track(
        stage_label="Train/Test Split",
        output_labels=["Train Set", "Test Set"],
        show_edge_label=False,  # Hide the edges for the train/test split
        node_style={"fillcolor": "lightgreen"},
    )
    def split_into_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Very simplistic "split": everything with value < 40 is "train," else "test."
        train_df = df[df["value"] < 40]
        test_df = df[df["value"] >= 40]
        return train_df, test_df

    @datatracker.track(
        stage_label="Remove Low Values < 20",
        reason="value < 20",
        show_edge_label=True,
        edge_style={
            "color": "red",
            "style": "solid",
            "fontsize": "8",
        },  # Override edge style
    )
    def remove_values_less_than_20(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["value"] >= 20]

    # Example of a step you do NOT want to visualize
    @datatracker.track(track=False)  # Skips creating nodes/edges
    def some_auxiliary_step(df: pd.DataFrame) -> pd.DataFrame:
        df["value_plus_one"] = df["value"] + 1
        return df

    # ----------------
    # RUN THE PIPELINE
    # ----------------
    data = {
        "name": ["Alice", "Bob", "Alice", "Alice", "Charlie", "Bob"],
        "value": [10, 20, 10, 30, 100, 20],
    }
    df = pd.DataFrame(data)

    # 1) Remove duplicates
    df_no_dup = remove_duplicates(df)

    # 2) (Untracked) Some random step you don't need in the diagram
    df_aux = some_auxiliary_step(df_no_dup)

    # 3) Train/test split
    df_train, df_test = split_into_train_test(df_aux)

    # 4) Remove super-low values from the training set
    df_train_clean = remove_values_less_than_20(df_train)

    # Render the final diagram
    datatracker.graph.render("styled_data_flow", format="png", cleanup=True)
    print("Done. Check out styled_data_flow.png for the visualization.")


if __name__ == "__main__":
    main()
