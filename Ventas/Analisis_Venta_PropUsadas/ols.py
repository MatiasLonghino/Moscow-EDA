"""OLS, predicciones y evaluación independiente del selector."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


def design_matrix(frame):
    if "const" in frame:
        raise ValueError("El nombre const está reservado para el intercepto")
    result = frame.astype(float).copy()
    result.insert(0, "const", 1.0)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("OLS requiere datos numéricos finitos")
    return result


def fit_ols(X, y):
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X e y deben tener las mismas filas e índices")
    if len(y) < 3 or not np.isfinite(np.asarray(y, dtype=float)).all():
        raise ValueError("OLS requiere al menos tres objetivos finitos")
    model = sm.OLS(y.astype(float), design_matrix(X), missing="raise").fit()
    if model.df_resid <= 0:
        raise ValueError("OLS sin grados de libertad residuales: reducir las features")
    return model


def predict_ols(model, X):
    columns = [c for c in model.model.exog_names if c != "const"]
    return model.predict(design_matrix(X.loc[:, columns]))


def evaluate_ols(model, X, y, split="validation"):
    """AIC/BIC son propiedades del ajuste y se informan sólo en train.

    R² ajustado fuera de train usa el rango efectivo del modelo de train.
    """
    prediction = predict_ols(model, X)
    residual = np.asarray(y) - np.asarray(prediction)
    n = len(y)
    sse = float(residual @ residual)
    sst = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    rank = int(model.model.rank)
    values = {"n": n, "r2": r2,
              "r2_adjusted": 1 - (1 - r2) * (n - 1) / (n - rank) if n > rank else float("nan"),
              "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(sse / n))}
    if split == "train":
        values.update(aic=float(model.aic), bic=float(model.bic))
    return {f"{split}_{key}": value for key, value in values.items()}


def diagnose_ols(model, X):
    residual = np.asarray(model.resid)
    jb = jarque_bera(residual)
    result = {"residual_mean": float(residual.mean()), "residual_std": float(residual.std()),
              "jarque_bera": float(jb.statistic), "jarque_bera_pvalue": float(jb.pvalue),
              "durbin_watson": float(durbin_watson(residual)),
              "condition_number": float(model.condition_number),
              "design_rank": int(model.model.rank),
              "design_columns": len(model.model.exog_names),
              "rank_deficient": bool(model.model.rank < len(model.model.exog_names))}
    if X.shape[1] and model.model.rank > 1:
        lm, pvalue, _, _ = het_breuschpagan(residual, model.model.exog)
        result.update(breusch_pagan=float(lm), breusch_pagan_pvalue=float(pvalue))
    # VIF de todas las columnas mediante una sola descomposición.
    variable = X.loc[:, X.std(ddof=0) > 0]
    if variable.shape[1]:
        z = (variable - variable.mean()) / variable.std(ddof=0)
        correlation = z.to_numpy().T @ z.to_numpy() / len(z)
        eigenvalues, vectors = np.linalg.eigh(correlation)
        nonzero = eigenvalues > max(eigenvalues.max(), 1) * 1e-10
        invdiag = (vectors[:, nonzero] ** 2 / eigenvalues[nonzero]).sum(axis=1)
        dependent = (vectors[:, ~nonzero] ** 2).sum(axis=1) > 1e-10
        result["vif"] = {col: float("inf") if dependent[i] else float(invdiag[i])
                         for i, col in enumerate(variable)}
    else:
        result["vif"] = {}
    return result


def diagnostic_figure(model):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(model.fittedvalues, model.resid, alpha=0.25, s=5)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Precio ajustado (RUB)", ylabel="Residuo (RUB)", title="Entrenamiento")
    sm.qqplot(model.resid, line="s", ax=axes[1])
    axes[2].hist(model.resid, bins=40)
    axes[2].set(xlabel="Residuo (RUB)", title="Distribución de residuos")
    fig.tight_layout()
    return fig
