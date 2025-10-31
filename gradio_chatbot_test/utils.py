import os
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def get_env(name: str, default=None, cast=None):
    """환경 변수 읽기 (타입 변환 지원)"""
    value = os.getenv(name, default)
    if cast and value is not None:
        try:
            return cast(value)
        except Exception:
            return default
    return value

TOKEN_PATTERN = re.compile(r"[A-Za-z가-힣0-9]+")

def simple_tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())

def normalize_scores(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
