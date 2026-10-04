"""Tree generation for intersectional gating.

Generates combined columns (depth-2) and manages the parent-child relationships.
"""
import itertools
import pandas as pd

def build_intersections(df: pd.DataFrame, sensitive_cols: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build depth-2 intersections of sensitive attributes.

    A row missing either parent value is missing in the intersection too: an
    unknown race is not a group called "nan", and the parent analyzers drop it
    for the same reason.

    Returns:
        A tuple of (modified_df, tree) where tree maps parent attributes to their
        intersection children. e.g. "race" -> ["race_AND_sex", "race_AND_age"].
    """
    df_out = df.copy()
    tree: dict[str, list[str]] = {col: [] for col in sensitive_cols}

    # Cap depth at 2
    for col1, col2 in itertools.combinations(sensitive_cols, 2):
        if col1 not in df.columns or col2 not in df.columns:
            continue

        combo_name = f"{col1}_AND_{col2}"
        # Combine the columns into a single categorical column
        combined = df_out[col1].astype(str) + " + " + df_out[col2].astype(str)
        df_out[combo_name] = combined.where(df_out[col1].notna() & df_out[col2].notna())

        tree[col1].append(combo_name)
        tree[col2].append(combo_name)

    return df_out, tree


def parents_of(tree: dict[str, list[str]]) -> dict[str, list[str]]:
    """Invert `tree` into child -> parents, in the order children were built.

    Gating looks parents up here rather than by parsing the child's name,
    which breaks for any user column whose name already contains "_AND_".
    """
    parents: dict[str, list[str]] = {}
    for parent, children in tree.items():
        for child in children:
            parents.setdefault(child, []).append(parent)
    return parents
