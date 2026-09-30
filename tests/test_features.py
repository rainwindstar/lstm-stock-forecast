import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler


def test_add_features_columns_and_no_nan(app, ohlcv):
    out = app.add_features(ohlcv)
    for col in ["ma5", "ma20", "ma60", "rsi", "macd", "macd_signal", "vol_log"]:
        assert col in out.columns
    assert not out.isna().any().any()
    assert out["rsi"].between(0, 100).all()


def test_add_features_is_causal(app, ohlcv):
    """미래 행을 잘라내도 과거 행의 피처 값이 같아야 한다 (rolling/ewm causality)."""
    full = app.add_features(ohlcv)
    cut = app.add_features(ohlcv.iloc[:250])
    common = cut.index
    pd.testing.assert_frame_equal(full.loc[common], cut, check_exact=False, rtol=1e-9)


def test_make_sequences_shape_and_target_excludes_window(app):
    data = np.arange(40, dtype=float).reshape(20, 2)
    X, y = app.make_sequences(data, seq_len=5)
    assert X.shape == (15, 5, 2) and y.shape == (15,)
    # 타깃은 윈도우 직후 시점의 0번 컬럼이어야 한다 (당일 정보 미포함)
    np.testing.assert_array_equal(X[0][:, 0], data[0:5, 0])
    assert y[0] == data[5, 0]
    assert y[0] not in X[0][:, 0]


def test_train_only_scaler_split_indices(app):
    """app.py:1265-1281의 분할 산식을 재현: 검증 타깃은 split_idx 이후여야 한다.
    TODO: 해당 블록을 함수로 분리하면 실제 함수를 호출하도록 교체."""
    n, lookback, val_split = 700, 60, 0.2
    n_val = max(1, int((n - lookback) * val_split))
    split_idx = n - n_val
    train_count = split_idx - lookback
    feat = np.random.default_rng(0).normal(size=(n, 3))
    scaler = MinMaxScaler().fit(feat[:split_idx])
    X, y = app.make_sequences(scaler.transform(feat), lookback)
    target_pos = np.arange(lookback, n)
    assert target_pos[:train_count].max() < split_idx <= target_pos[train_count:].min()
    assert len(y[train_count:]) == n_val


@pytest.mark.xfail(reason="🟠 bfill이 초기 구간을 미래값으로 채움 (app.py:1033, 1046)")
def test_no_bfill_in_external_feature_alignment(app_src):
    assert ".bfill()" not in app_src
