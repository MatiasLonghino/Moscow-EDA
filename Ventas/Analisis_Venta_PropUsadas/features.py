"""Transformaciones deterministas y exclusión de información inadmisible."""
import numpy as np
import pandas as pd


FORBIDDEN = {"price_rub", "price_per_sqm", "id", "date_posted", "year_month"}


def build_features(frame, include=None, exclude=None, derive=True):
    result = frame.copy()
    date = pd.to_datetime(result["date_posted"], errors="raise")
    if derive:
        result["listing_year"] = date.dt.year.astype(float)
        result["listing_month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
        result["listing_month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
        if "building_year" in result:
            age = date.dt.year - pd.to_numeric(result["building_year"], errors="coerce")
            result["building_age"] = age.where(age >= 0)
        if {"floor", "total_floors"}.issubset(result):
            result["relative_floor"] = result["floor"] / result["total_floors"].replace(0, np.nan)
    # Una estación todavía no inaugurada no aporta atributos operativos.
    if "metro_year_opened" in result:
        future = result["metro_year_opened"] > date.dt.year
        result.loc[future, [c for c in result if c.startswith("metro_") and c not in
                           ("metro_station", "metro_line", "metro_distance_min", "metro_distance_type")]] = np.nan
    if include is not None:
        if set(include) & FORBIDDEN:
            raise ValueError(f"Features no admisibles: {sorted(set(include) & FORBIDDEN)}")
        missing = set(include) - set(result)
        if missing:
            raise ValueError(f"Features inexistentes: {sorted(missing)}")
        result = result.loc[:, list(include)]
    else:
        result = result.drop(columns=list(FORBIDDEN), errors="ignore")
    result = result.drop(columns=list(exclude or []), errors="ignore")
    if result.empty:
        raise ValueError("No hay variables explicativas")
    return result
