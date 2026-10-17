"""Selección sobre entrenamiento; las categorías se evalúan como bloques."""
from itertools import combinations
from math import comb

import numpy as np
from scipy.stats import f as f_distribution

if __package__:
    from .ols import fit_ols
else:
    from ols import fit_ols


def _setup(X, y, groups, criterion):
    if criterion not in ("aic", "bic", "pvalue"):
        raise ValueError("criterion debe ser aic, bic o pvalue")
    groups = {name: list(cols) for name, cols in (groups if groups is not None else
              {c: [c] for c in X}).items() if cols}
    flattened = [c for cols in groups.values() for c in cols]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(X):
        raise ValueError("Los grupos deben particionar exactamente las columnas de X")
    cache = {}

    def fit(names):
        key = tuple(g for g in groups if g in names)
        if key not in cache:
            model = fit_ols(X[[c for g in key for c in groups[g]]], y)
            # No retener cientos de matrices de diseño/modelos en las búsquedas.
            cache[key] = {"aic": float(model.aic), "bic": float(model.bic),
                          "ssr": float(model.ssr), "df_resid": float(model.df_resid)}
        return cache[key]

    return groups, fit, cache


def _group_pvalue(full, reduced):
    if full["df_resid"] >= reduced["df_resid"]:
        return 1.0  # bloque redundante, sin nuevos grados de libertad
    difference = max(0., reduced["ssr"] - full["ssr"])
    if full["ssr"] == 0:
        return 0.0 if difference > 0 else 1.0
    statistic = difference / (reduced["df_resid"] - full["df_resid"]) / (full["ssr"] / full["df_resid"])
    pvalue = float(f_distribution.sf(statistic, reduced["df_resid"] - full["df_resid"], full["df_resid"]))
    return pvalue if np.isfinite(pvalue) else 1.0


def _result(groups, selected, history, cache, converged=True):
    selected = [g for g in groups if g in selected]
    return {"selected_groups": selected,
            "selected_features": [c for g in selected for c in groups[g]],
            "history": history, "n_models": len(cache), "converged": converged}


def _search(X, y, groups=None, criterion="aic", mode="forward", max_steps=100,
            min_improvement=1e-6, entry_threshold=0.05, exit_threshold=0.10):
    if max_steps < 1 or min_improvement < 0:
        raise ValueError("max_steps >= 1 y min_improvement >= 0 requeridos")
    if not 0 < entry_threshold < exit_threshold < 1:
        raise ValueError("Se requiere 0 < entry_threshold < exit_threshold < 1")
    groups, fit, cache = _setup(X, y, groups, criterion)
    selected = list(groups) if mode == "backward" else []
    current = fit(selected)
    history = []
    seen = {tuple(selected)}
    for _ in range(max_steps):
        changes = []
        if mode in ("forward", "stepwise"):
            for g in groups:
                if g not in selected:
                    names = [n for n in groups if n in selected or n == g]
                    candidate = fit(names)
                    score = _group_pvalue(candidate, current) if criterion == "pvalue" else candidate[criterion]
                    if (score < entry_threshold if criterion == "pvalue" else score < current[criterion] - min_improvement):
                        changes.append((score, "add", g, names, candidate))
        if mode in ("backward", "stepwise"):
            for g in selected:
                names = [n for n in selected if n != g]
                candidate = fit(names)
                pvalue = _group_pvalue(current, candidate)
                score = -pvalue if criterion == "pvalue" else candidate[criterion]
                if (pvalue > exit_threshold if criterion == "pvalue" else score < current[criterion] - min_improvement):
                    changes.append((score, "drop", g, names, candidate))
        if not changes:
            return _result(groups, selected, history, cache)
        score, action, group, names, current = min(changes, key=lambda item: (item[0], item[1], item[2]))
        if tuple(names) in seen:
            return _result(groups, selected, history, cache, converged=False)
        selected = names
        seen.add(tuple(names))
        history.append({"action": action, "group": group, "criterion": criterion,
                        "value": abs(score) if criterion == "pvalue" else score,
                        "selected_groups": list(selected)})
    return _result(groups, selected, history, cache, converged=False)


def forward_selection(X, y, groups=None, criterion="aic", **kwargs):
    return _search(X, y, groups, criterion, mode="forward", **kwargs)


def backward_elimination(X, y, groups=None, criterion="aic", **kwargs):
    return _search(X, y, groups, criterion, mode="backward", **kwargs)


def stepwise_selection(X, y, groups=None, criterion="aic", **kwargs):
    return _search(X, y, groups, criterion, mode="stepwise", **kwargs)


def best_subset_selection(X, y, groups=None, criterion="bic", max_features=5,
                          max_groups=15, max_models=4096):
    """max_features limita grupos, incluyendo siempre el modelo con intercepto."""
    if criterion not in ("aic", "bic"):
        raise ValueError("Best subset requiere aic o bic")
    groups, fit, cache = _setup(X, y, groups, criterion)
    if max_features < 0 or max_groups < 0 or max_models < 1:
        raise ValueError("Límites de búsqueda inválidos")
    size = min(max_features, len(groups))
    budget = sum(comb(len(groups), k) for k in range(size + 1))
    if len(groups) > max_groups or budget > max_models:
        raise ValueError(f"Best subset excede límites: {len(groups)} grupos, {budget} modelos. Reducir candidatas o max_features.")
    best, best_score, history = [], float("inf"), []
    for k in range(size + 1):
        for names in combinations(groups, k):
            score = fit(names)[criterion]
            history.append({"groups": list(names), "criterion": criterion, "value": score})
            if score < best_score:
                best, best_score = list(names), score
    return _result(groups, best, history, cache)


def no_selection(X, y, groups=None, criterion="aic"):
    groups, _, cache = _setup(X, y, groups, criterion)
    return _result(groups, list(groups), [], cache)


SELECTORS = {"none": no_selection, "forward": forward_selection,
             "backward": backward_elimination, "stepwise": stepwise_selection,
             "best_subset": best_subset_selection}
