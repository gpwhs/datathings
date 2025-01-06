import functools
import pandas as pd
from graphviz import Digraph
from typing import List, Optional


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
        self.graph = Digraph(name=name)
        self.node_count = 0

        # Map "id of a DataFrame object" -> "node ID in the graph"
        self.df_to_node = {}

        # Store default node & edge attributes
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
        Otherwise, create a new node with the specified label & styling.
        """
        df_id = id(df)
        if df_id not in self.df_to_node:
            new_id = self._new_node_id()
            self.df_to_node[df_id] = new_id

            # Merge default + any user-specified node style
            merged_node_style = dict(self.default_node_style)
            if node_style:
                merged_node_style.update(node_style)

            self.graph.node(new_id, label=label_if_new, **merged_node_style)

        return self.df_to_node[df_id]

    def track(
        self,
        stage_label: str = None,
        reason: str = None,
        output_labels: Optional[List[str]] = None,
        show_edge_label: bool = True,
        track: bool = True,
        node_style: Optional[dict] = None,
        edge_style: Optional[dict] = None,
    ):
        """
        Decorator that can handle:
          • a function returning ONE DataFrame,
          • or multiple DataFrames (e.g. train/test split).

        Arguments:
          - stage_label:    A name for this transformation (e.g. "Train/Test Split", "Remove duplicates").
          - reason:         If rows are removed, show the reason in the edge label (unless show_edge_label=False).
          - output_labels:  If multiple DataFrames are returned, optionally provide labels (["Train", "Test"], etc.).
          - show_edge_label: If False, do not label the edges at all.
          - track:          If False, skip adding nodes/edges to the diagram (but still run the function).
          - node_style:     Optional dict of styling attributes for the output nodes of *this* stage.
          - edge_style:     Optional dict of styling attributes for edges from this stage.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not track:
                    # Skip adding to the diagram; just run the function
                    return func(*args, **kwargs)

                if not args:
                    raise ValueError(
                        "The tracked function must receive a DataFrame as its first argument."
                    )
                if not isinstance(args[0], pd.DataFrame):
                    raise ValueError("The first argument must be a pd.DataFrame.")

                df_in = args[0]
                n_in = len(df_in)

                # -- 1) If we've never seen this DataFrame, label it as "Initial Dataset"
                df_in_id = id(df_in)
                if df_in_id not in self.df_to_node:
                    input_label = f"Initial Dataset\nN={n_in}"
                else:
                    # If it's already in the tracker, we're presumably doing a new transformation.
                    # So we could label it something like:
                    #   e.g. "<stage_label> Input\nN={n_in}"
                    # But often you won't need to rewrite the label for an *existing* node.
                    # We'll keep it minimal and just reuse the existing node.
                    input_label = f"Initial Dataset\nN={n_in}"

                input_node_id = self._get_or_create_node_for_df(
                    df_in, label_if_new=input_label
                )

                # -- 2) Call the transformation
                result = func(*args, **kwargs)

                # We'll allow a single DF or list/tuple of DFs
                if isinstance(result, pd.DataFrame):
                    df_out_list = [result]
                elif isinstance(result, (list, tuple)) and all(
                    isinstance(x, pd.DataFrame) for x in result
                ):
                    df_out_list = list(result)
                else:
                    raise ValueError(
                        f"Tracked function {func.__name__} returned something that's not a DataFrame "
                        f"or a list/tuple of DataFrames."
                    )

                # -- 3) Create or re-use nodes for each output DataFrame
                edge_kwargs = dict(self.default_edge_style)
                if edge_style:
                    edge_kwargs.update(edge_style)

                for i, df_out in enumerate(df_out_list):
                    n_out = len(df_out)

                    # Construct the label for the *output* node
                    if output_labels and i < len(output_labels):
                        out_label = f"{output_labels[i]}\nN={n_out}"
                    else:
                        out_label_main = stage_label or func.__name__
                        if len(df_out_list) > 1:
                            out_label = f"Output {i+1} of {out_label_main}\nN={n_out}"
                        else:
                            out_label = f"{out_label_main}\nN={n_out}"

                    out_node_id = self._get_or_create_node_for_df(
                        df_out, label_if_new=out_label, node_style=node_style
                    )

                    # -- 4) Possibly add an edge label
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

                        # If multiple outputs & the user provided output_labels, we might refine
                        if output_labels and i < len(output_labels):
                            if removed_count != 0:
                                edge_label = f"{output_labels[i]} | {edge_label}"
                            else:
                                edge_label = output_labels[i]

                        edge_kwargs["label"] = edge_label

                    self.graph.edge(input_node_id, out_node_id, **edge_kwargs)

                # Return the single or multiple DataFrames
                return result

            return wrapper

        return decorator


def main():
    """
    Sample usage:
    - The first time a DataFrame is seen, the node is labeled "Initial Dataset\nN=..."
    - Subsequent transformations get new nodes labeled "<stage_label>\nN=..."
    """
    datatracker = DataTracker()

    @datatracker.track(
        stage_label="Remove Duplicates", reason="Duplicates", show_edge_label=True
    )
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    @datatracker.track(
        stage_label="Train/Test Split",
        output_labels=["Train", "Test"],
        show_edge_label=False,
        edge_style={"color": "blue"},
    )
    def split_into_train_test(df: pd.DataFrame):
        train_df = df[df["value"] < 40]
        test_df = df[df["value"] >= 40]
        return train_df, test_df

    data = {
        "name": ["Alice", "Bob", "Alice", "Alice", "Charlie", "Bob"],
        "value": [10, 20, 10, 30, 100, 20],
    }
    df = pd.DataFrame(data)

    # 1) The first time df is used, node labeled "Initial Dataset\nN=6"
    df_no_dup = remove_duplicates(df)

    # 2) Split into train/test
    df_train, df_test = split_into_train_test(df_no_dup)

    datatracker.graph.render("my_flow_diagram", format="png", cleanup=True)
    print("Diagram saved: my_flow_diagram.png")


if __name__ == "__main__":
    main()
