"""One-hot estable, con categoría de referencia y grupos por feature."""
import numpy as np
import pandas as pd


def fit_encoding(frame):
    state = {"columns": list(frame), "categories": {}, "groups": {}, "output_columns": []}
    for position, col in enumerate(frame):
        if pd.api.types.is_numeric_dtype(frame[col]):
            outputs = [col]
        else:
            categories = sorted(frame[col].astype(str).unique())
            state["categories"][col] = categories
            # Nombres únicos incluso si una categoría contiene separadores.
            outputs = [f"{col}__oh{position}_{i}={category}" for i, category in enumerate(categories[1:], 1)]
        state["groups"][col] = outputs
        state["output_columns"].extend(outputs)
    if len(set(state["output_columns"])) != len(state["output_columns"]):
        raise ValueError("Colisión de nombres de columnas después del encoding")
    return state


def apply_encoding(frame, state, handle_unknown="ignore"):
    """ignore codifica categorías desconocidas como la referencia (todo cero)."""
    if handle_unknown not in ("ignore", "error"):
        raise ValueError("handle_unknown debe ser ignore o error")
    result = {}
    for col in state["columns"]:
        if col not in state["categories"]:
            result[col] = frame[col].astype(float)
            continue
        values = frame[col].astype(str)
        categories = state["categories"][col]
        if handle_unknown == "error" and not values.isin(categories).all():
            raise ValueError(f"Categorías desconocidas en {col}")
        for name, category in zip(state["groups"][col], categories[1:]):
            result[name] = (values == category).astype(float)
    result = pd.DataFrame(result, index=frame.index).reindex(columns=state["output_columns"])
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("La matriz codificada contiene valores no finitos")
    return result
