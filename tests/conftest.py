"""app.py는 import 시 Streamlit UI와 모델 학습이 실행되므로 직접 import할 수 없다.

app.py를 수정하지 않고, AST로 순수 함수 정의만 추출해 격리된 네임스페이스에서 실행한다.
(캐시 데코레이터는 제거한다.) 함수를 별도 모듈로 분리하면 이 로더는 일반 import로 대체한다.
"""
import ast
import datetime
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

PURE_FUNCS = {
    "_truthy", "_sheet_safe_title", "_number_or_blank", "_safe_float",
    "_ecos_period_bounds", "parse_corpcode_xml", "add_features", "make_sequences",
    "refresh_actual_prices", "build_prediction_sheet_rows", "get_dart_fundamentals",
}


def load_app_functions(extra_globals=None):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in PURE_FUNCS:
            node.decorator_list = []
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ns = {
        "np": np, "pd": pd, "datetime": datetime.datetime, "timedelta": datetime.timedelta,
        "ET": __import__("xml.etree.ElementTree", fromlist=["x"]),
        "requests": types.SimpleNamespace(get=None),
    }
    ns.update(extra_globals or {})
    exec(compile(module, str(APP_PATH), "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in PURE_FUNCS})


@pytest.fixture(scope="session")
def app_src():
    return APP_PATH.read_text(encoding="utf-8")


@pytest.fixture()
def app():
    return load_app_functions()


@pytest.fixture()
def ohlcv():
    """결정적 합성 OHLCV (300 영업일). 네트워크 호출 없음."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2023-01-02", periods=300)
    close = 70000 + np.cumsum(rng.normal(0, 500, len(idx)))
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, len(idx)),
    }, index=idx)
