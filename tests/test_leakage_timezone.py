"""리뷰에서 발견된 결함을 재현하는 테스트. xfail(strict) — 수정되면 XPASS로 실패해 마커 제거를 강제한다."""
import ast
import types

import pandas as pd
import pytest

from conftest import load_app_functions


def _fake_dart_requests():
    """연도·보고서별로 구분 가능한 영업이익률을 돌려주는 가짜 DART 응답."""
    codes = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}

    class R:
        def __init__(self, year, rc):
            self.year, self.rc = int(year), rc

        def json(self):
            margin_bp = (self.year - 2000) * 10 + codes[self.rc]  # 예: FY2023 연간 = 234
            return {"list": [
                {"account_nm": "매출액", "thstrm_amount": "1,000,000"},
                {"account_nm": "영업이익", "thstrm_amount": str(margin_bp * 100)},
            ]}

    return types.SimpleNamespace(get=lambda url, params, timeout: R(params["bsns_year"], params["reprt_code"]))


@pytest.mark.xfail(reason="🟠 연간 사업보고서 가용일이 12/28+60일(=다음해 2/26)로 법정기한(3/31) 이전")
def test_dart_annual_not_available_before_march_31():
    app = load_app_functions({"requests": _fake_dart_requests()})
    df = app.get_dart_fundamentals("X", "2022-01-03", "2024-12-31", "KEY")
    fy2023_annual = (23 * 10 + 4) * 100 / 1_000_000  # 영업이익률 0.0234
    first = df.index[(df["영업이익률"] - fy2023_annual).abs() < 1e-12].min()
    assert first >= pd.Timestamp("2024-03-31")


def _calls(src, attr):
    hits = []
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
                and not n.args and not n.keywords):
            hits.append(n.lineno)
    return hits


@pytest.mark.xfail(reason="🟠 naive datetime.now()/today() — Asia/Seoul 미지정 (app.py:350,503,1442)")
def test_no_naive_now_or_today(app_src):
    assert _calls(app_src, "now") == [] and _calls(app_src, "today") == []


@pytest.mark.xfail(reason="🟠 예측일을 freq='B'로 생성 — KRX 휴장일 미반영 (app.py:1379)")
def test_forecast_dates_use_krx_calendar(app_src):
    assert "freq='B'" not in app_src and 'freq="B"' not in app_src


@pytest.mark.xfail(reason="🟠 torch 미설치 시 nn 참조 NameError — fallback 도달 불가 (app.py:795)")
def test_torch_wrapper_is_guarded(app_src):
    tree = ast.parse(app_src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PyTorchLSTMWrapper")
    inner = next(n for n in cls.body if isinstance(n, ast.ClassDef))
    bases = [ast.unparse(b) for b in inner.bases]
    assert "nn.Module" not in bases  # 모듈 최상위에서 무조건 nn을 참조하면 안 됨


@pytest.mark.skip(reason="TODO: 사이드 블록(app.py:1343-1348)을 함수로 분리 후 나이브 기준선·방향정확도 정의 테스트 추가")
def test_direction_accuracy_uses_previous_close_baseline():
    ...
