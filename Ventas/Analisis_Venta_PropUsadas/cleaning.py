"""Imputación aprendida exclusivamente en entrenamiento."""
import numpy as np
import pandas as pd


def validate_target(frame, target="price_rub"):
    values = pd.to_numeric(frame[target], errors="raise")
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{target} debe contener precios positivos, finitos y no nulos")
    return values.astype(float)


def fit_cleaning(frame, numeric_strategy="median", categorical_fill="__MISSING__"):
    if numeric_strategy not in ("median", "mean"):
        raise ValueError("numeric_strategy debe ser median o mean")
    state = {"columns": list(frame), "numeric": [], "fill": {},
             "numeric_strategy": numeric_strategy, "categorical_fill": categorical_fill}
    for col in frame:
        if pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col]):
            state["numeric"].append(col)
            values = pd.to_numeric(frame[col]).replace([np.inf, -np.inf], np.nan)
            value = getattr(values, numeric_strategy)()
            state["fill"][col] = float(value) if pd.notna(value) else 0.0
        else:
            state["fill"][col] = categorical_fill
    return state


def apply_cleaning(frame, state):
    result = frame.loc[:, state["columns"]].copy()
    for col in result:
        if col in state["numeric"]:
            result[col] = pd.to_numeric(result[col], errors="raise").replace([np.inf, -np.inf], np.nan).fillna(state["fill"][col]).astype(float)
        else:
            result[col] = result[col].astype("string").fillna(state["fill"][col])
    return result
