import pandas as pd
import pytest


def test_number_or_blank(app):
    f = app._number_or_blank
    assert f(None) == "" and f("") == "" and f(float("nan")) == ""
    assert f("1,234") == 1234
    assert f("12.3456", digits=2) == 12.35
    assert f("abc") == ""


def test_sheet_safe_title(app):
    assert app._sheet_safe_title("LSTM_[삼성]:전자*?/\\") == "LSTM_삼성전자"
    assert app._sheet_safe_title("[]") == "LSTM_주가예측"
    assert len(app._sheet_safe_title("가" * 200)) == 80


def test_truthy(app):
    assert app._truthy("True") and app._truthy(" on ") and not app._truthy("")


def test_refresh_actual_prices_fills_error(app):
    headers = ["종목코드", "예측일", "예측종가", "실제종가", "오차", "오차율(%)"]
    values = [headers, ["005930", "2026-01-05", "70000", "", "", ""],
              ["000660", "2026-01-05", "100", "", "", ""]]
    out, changed = app.refresh_actual_prices(values, "005930", {"2026-01-05": 71400})
    assert changed == 1
    assert out[1][3:] == ["71400", "1400", "2.00"]
    assert out[2][3:] == ["", "", ""]  # 다른 종목은 미변경


def test_refresh_actual_prices_missing_columns_noop(app):
    values = [["a", "b"], ["1", "2"]]
    assert app.refresh_actual_prices(values, "005930", {}) == (values, 0)


def test_build_rows_width_matches_headers(app):
    # PREDICTION_HEADERS는 app.py 상수라 여기서는 열 수(22)만 고정
    pred = pd.DataFrame({"날짜": ["2026-01-05"], "예측 종가": [70000], "전일대비": [100], "등락률(%)": [0.14]})
    rows = app.build_prediction_sheet_rows(
        pred, "삼성전자", "005930", "2026-01-02", {"2026-01-05": 71000},
        {"rmse": 1.0, "mae": 1.0, "mape": 1.0, "direction_accuracy": 50.0},
        {"lookback": 60, "pred_days": 30, "epochs": 100, "batch": 32}, ["close", "rsi"],
    )
    assert len(rows) == 1 and len(rows[0]) == 22
    assert rows[0][9] == 71000 and rows[0][10] == 1000  # 실제종가, 오차
