
# preprocess.py
"""
Utilities to load and clean the Crime vs No-Crime articles dataset.
Expects a CSV with:
  - text column: "title"
  - label column: "is_crime_report" (0/1, False/True, "yes"/"no" supported)
"""

import re
import pandas as pd
from typing import Tuple, List

TEXT_COL = "title"
LABEL_COL = "is_crime_report"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    x = text.lower()
    x = _URL_RE.sub(" ", x)
    x = _HTML_RE.sub(" ", x)
    x = _NONALNUM_RE.sub(" ", x)
    x = _MULTISPACE_RE.sub(" ", x).strip()
    return x

def _normalize_label(val):
    # Map common truthy/falsey values to 0/1
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return int(val > 0)
    s = str(val).strip().lower()
    if s in {"1","true","t","yes","y"}:
        return 1
    if s in {"0","false","f","no","n"}:
        return 0
    # Fallback: try to cast to int
    try:
        return int(s)
    except Exception:
        return None

def load_data(csv_path: str, text_col: str = TEXT_COL, label_col: str = LABEL_COL) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must have columns '{text_col}' and '{label_col}'. Found: {list(df.columns)}")
    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].apply(clean_text)
    df[label_col] = df[label_col].apply(_normalize_label)
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)
    return df

def get_X_y(df: pd.DataFrame, text_col: str = TEXT_COL, label_col: str = LABEL_COL) -> Tuple[List[str], List[int]]:
    X = df[text_col].tolist()
    y = df[label_col].tolist()
    return X, y
