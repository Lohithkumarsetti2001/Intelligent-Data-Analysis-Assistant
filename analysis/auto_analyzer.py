
import numpy as np
import pandas as pd
from scipy import stats

def detect_types(df: pd.DataFrame) -> dict:
    types = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            types[c] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            types[c] = "datetime"
        else:
            types[c] = "categorical"
    return types

def quick_summary(df: pd.DataFrame) -> dict:
    out = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing_pct": float(df.isna().mean().mean())*100,
        "numeric_cols": df.select_dtypes(include="number").shape[1],
        "categorical_cols": df.select_dtypes(exclude="number").shape[1]
    }
    return out

def simple_tests(df: pd.DataFrame, target: str) -> list[str]:
    """Attempt quick significance checks against a numeric target."""
    insights = []
    if target not in df.columns:
        return insights
    y = df[target].dropna()
    if len(y) < 10:
        return insights
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in num_cols:
        if c == target: 
            continue
        x = df[c].dropna()
        if len(x) >= 10:
            r, p = stats.pearsonr(y.loc[x.index].fillna(y.mean()), x)
            if np.isfinite(p) and p < 0.05:
                insights.append(f"{c} is correlated with {target} (r={r:.2f}, p={p:.3g}).")
    return insights[:5]
