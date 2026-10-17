"""Carga y joins muchos-a-uno, sin escribir datasets en disco."""
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "Data"


def _unique_keys(frame, keys, name):
    if frame[keys].isna().any().any():
        raise ValueError(f"{name}: claves nulas: {keys}")
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: claves duplicadas: {keys}")


def load_data(secondary_path=None, monthly_path=None, metro_path=None,
              monthly_lag_months=1, require_matches=False):
    """El rezago expresa cuándo se supone disponible un agregado mensual.

    Debe ser al menos un mes: no se admiten agregados del mes en curso.
    La disponibilidad real debe comprobarse con el proveedor; aumentar el
    rezago si la publicación ocurre más tarde. Se exige coincidencia exacta
    del mes rezagado, sin rellenar desde meses futuros.
    """
    if not isinstance(monthly_lag_months, int) or monthly_lag_months < 1:
        raise ValueError("monthly_lag_months debe ser un entero >= 1")
    sales = pd.read_csv(secondary_path or DEFAULT_DATA_DIR / "secondary_market.csv")
    monthly = pd.read_csv(monthly_path or DEFAULT_DATA_DIR / "district_prices_monthly.csv")
    metro = pd.read_csv(metro_path or DEFAULT_DATA_DIR / "metro_stations.csv")
    for col in ["date_posted", "district", "okrug", "metro_station", "metro_line", "price_rub"]:
        if col not in sales:
            raise ValueError(f"Falta columna de propiedades: {col}")
    if "id" in sales:
        _unique_keys(sales, ["id"], "propiedades")
    elif sales.duplicated().any():
        raise ValueError("Propiedades: filas duplicadas sin identificador")
    sales["date_posted"] = pd.to_datetime(sales["date_posted"], errors="raise")
    if sales[["date_posted", "district", "okrug", "metro_station", "metro_line"]].isna().any().any():
        raise ValueError("Propiedades: claves de integración nulas")
    monthly["year_month"] = pd.to_datetime(monthly["year_month"], errors="raise").dt.to_period("M").astype(str)
    _unique_keys(monthly, ["year_month", "district", "okrug"], "mensual")
    _unique_keys(metro, ["station_name", "line"], "metro")
    # Prefijos evitan colisiones y permiten auditar el origen de las features.
    monthly = monthly.rename(columns={c: f"monthly_{c}" for c in monthly
                                     if c not in ["year_month", "district", "okrug"]})
    sales["year_month"] = (sales["date_posted"].dt.to_period("M") - monthly_lag_months).astype(str)
    result = sales.merge(monthly, on=["year_month", "district", "okrug"], how="left",
                         validate="many_to_one", indicator="_monthly_join", sort=False)
    metro = metro.rename(columns={c: f"metro_{c}" for c in metro
                                 if c not in ["station_name", "line"]})
    result = result.merge(metro, left_on=["metro_station", "metro_line"],
                          right_on=["station_name", "line"], how="left",
                          validate="many_to_one", indicator="_metro_join", sort=False)
    if len(result) != len(sales):
        raise ValueError("Los joins alteraron la cantidad de propiedades")
    audit = {"rows": len(result), "monthly_unmatched": int((result._monthly_join == "left_only").sum()),
             "metro_unmatched": int((result._metro_join == "left_only").sum()),
             "monthly_lag_months": monthly_lag_months,
             "missing_by_column": result.isna().sum().to_dict()}
    if require_matches and (audit["monthly_unmatched"] or audit["metro_unmatched"]):
        raise ValueError(f"Joins incompletos: {audit}")
    result = result.drop(columns=["_monthly_join", "_metro_join", "station_name", "line"])
    result.attrs["join_audit"] = audit
    return result
