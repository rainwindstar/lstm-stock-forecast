# LSTM 주가 예측 대시보드 v4.1 — 전면 UI 리디자인
# 가격 기술지표 + 거시경제(환율·VIX·KOSPI) + DART 재무 통합 멀티피처 모델

import warnings, os, json
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib
try:
    import koreanize_matplotlib  # Streamlit Cloud/Linux에서 Matplotlib 한글 폰트 보강
except Exception:
    koreanize_matplotlib = None
matplotlib.rcParams['font.family'] = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import FinanceDataReader as fdr
import zipfile, io, requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = APP_DIR.parents[1]
LOCAL_CORPCODE_XML = APP_DIR / "CORPCODE.xml"
LOCAL_ENV_PATHS = [
    APP_DIR / ".env",
    WORKSPACE_DIR / ".env",
    WORKSPACE_DIR / "00_영웅문" / ".env",
]
DEFAULT_STOCKS = pd.DataFrame([
    {"회사명": "삼성전자", "종목코드": "005930", "corp_code": ""},
    {"회사명": "SK하이닉스", "종목코드": "000660", "corp_code": ""},
    {"회사명": "NAVER", "종목코드": "035420", "corp_code": ""},
    {"회사명": "카카오", "종목코드": "035720", "corp_code": ""},
    {"회사명": "현대차", "종목코드": "005380", "corp_code": ""},
    {"회사명": "씨젠", "종목코드": "096530", "corp_code": ""},
])


def load_local_env():
    for env_path in LOCAL_ENV_PATHS:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass


def _secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


load_local_env()

DART_API_KEY = (
    os.getenv("DART_API_KEY", "").strip()
    or os.getenv("OPENDART_API_KEY", "").strip()
    or _secret("DART_API_KEY")
    or _secret("OPENDART_API_KEY")
)
ECOS_API_KEY = os.getenv("ECOS_API_KEY", "").strip() or _secret("ECOS_API_KEY")
KRX_API_KEY = os.getenv("KRX_API_KEY", "").strip() or _secret("KRX_API_KEY")
PUBLIC_DATA_API_KEY = os.getenv("PUBLIC_DATA_API_KEY", "").strip() or _secret("PUBLIC_DATA_API_KEY")
GOOGLE_SHEET_FOLDER_ID = os.getenv("GOOGLE_SHEET_FOLDER_ID", "").strip() or _secret("GOOGLE_SHEET_FOLDER_ID")
GOOGLE_SHEET_SHARE_EMAIL = os.getenv("GOOGLE_SHEET_SHARE_EMAIL", "").strip() or _secret("GOOGLE_SHEET_SHARE_EMAIL")
GOOGLE_SHEETS_AUTO_SAVE_DEFAULT = _truthy(os.getenv("GOOGLE_SHEETS_AUTO_SAVE", _secret("GOOGLE_SHEETS_AUTO_SAVE")))

# ══════════════════════════════════════════════════════════
#  페이지 설정 & CSS
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="LSTM 주가 예측", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, .stDataFrame, .stSelectbox, .stTextInput, .stButton {
    font-family: 'Noto Sans KR', 'NanumGothic', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
}
/* ── 섹션 헤더 ───────────────────── */
.sec-header {
    font-size: 13px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #7890ab;
    border-bottom: 1px solid #2a3a50; padding-bottom: 6px;
    margin: 18px 0 10px;
}

/* ── 기본 KPI 카드 ────────────────── */
.kpi {
    background: #161f2e;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    padding: 14px 18px 12px;
    min-height: 90px;
}
.kpi-label {
    font-size: 11px; color: #7890ab;
    text-transform: uppercase; letter-spacing: .9px;
    white-space: nowrap;
}
.kpi-val {
    font-size: 24px; font-weight: 800; color: #f0f4ff;
    margin: 5px 0 3px; line-height: 1.15;
    white-space: nowrap;
}
.kpi-up   { font-size: 13px; color: #f87171; font-weight: 600; }
.kpi-down { font-size: 13px; color: #60a5fa; font-weight: 600; }
.kpi-neu  { font-size: 13px; color: #94a3b8; }

/* ── 거시경제 카드 ────────────────── */
.macro {
    background: #12163a;
    border-radius: 12px;
    border-left: 4px solid #818cf8;
    padding: 14px 18px 12px;
    min-height: 90px;
}
.macro .kpi-label { color: #a5b4fc; }
.macro .kpi-val   { font-size: 22px; }

/* ── DART 카드 ───────────────────── */
.dart {
    background: #0f2318;
    border-radius: 12px;
    border-left: 4px solid #34d399;
    padding: 14px 18px 12px;
    min-height: 90px;
}
.dart .kpi-label { color: #6ee7b7; }
.dart .kpi-val   { font-size: 22px; }

/* ── 정확도 카드 ─────────────────── */
.acc {
    background: #1a1a2e;
    border-radius: 12px;
    border-left: 4px solid #f59e0b;
    padding: 14px 18px 12px;
    text-align: center;
    min-height: 90px;
}
.acc .kpi-label { color: #fbbf24; }
.acc .kpi-val   { font-size: 28px; }

/* ── 종목 타이틀 바 ──────────────── */
.stock-title {
    background: linear-gradient(90deg, #1e3a5f 0%, #162032 100%);
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 6px;
    border: 1px solid #2d4a7a;
}
.stock-name { font-size: 28px; font-weight: 800; color: #e0eeff; }
.stock-code { font-size: 14px; color: #7890ab; margin-left: 8px; }
.stock-sub  { font-size: 12px; color: #7890ab; margin-top: 4px; }

/* ── 피처 뱃지 ───────────────────── */
.badge {
    display: inline-block;
    background: #1e3a5f; color: #93c5fd;
    border-radius: 20px; padding: 3px 10px;
    font-size: 11px; margin: 2px 2px;
    border: 1px solid #2d5a9e;
}
.badge.macro { background: #1e1b4b; color: #a5b4fc; border-color: #4338ca; }
.badge.dart  { background: #052e16; color: #6ee7b7; border-color: #065f46; }

/* ── 구분선 ──────────────────────── */
hr { border-color: #2a3a50 !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  카드 렌더 헬퍼
# ──────────────────────────────────────────────────────────
def kpi_card(label, value, delta=None, card="kpi", help_txt=""):
    if delta is not None:
        txt = str(delta)
        is_up  = "▲" in txt or (txt.startswith("+") and not txt.startswith("+-"))
        is_dn  = "▼" in txt or txt.startswith("-")
        cls    = "kpi-up" if is_up else ("kpi-down" if is_dn else "kpi-neu")
        d_html = f'<div class="{cls}">{delta}</div>'
    else:
        d_html = ""
    tip = f'title="{help_txt}"' if help_txt else ""
    return f"""
<div class="{card}" {tip}>
  <div class="kpi-label">{label}</div>
  <div class="kpi-val">{value}</div>
  {d_html}
</div>"""


def section(title, icon=""):
    st.markdown(f'<div class="sec-header">{icon} {title}</div>', unsafe_allow_html=True)


class GoogleSheetsConfigError(Exception):
    pass


PREDICTION_HEADERS = [
    "검색일시", "검색일자", "종목명", "종목코드", "기준일", "예측일",
    "예측종가", "예측전일대비", "예측등락률(%)", "실제종가", "오차", "오차율(%)",
    "lookback", "예측일수", "epochs", "batch", "검증RMSE", "검증MAE",
    "검증MAPE", "방향정확도(%)", "피처수", "피처목록",
]


def _sheet_safe_title(value, max_len=80):
    invalid_chars = '[]:*?/\\'
    cleaned = "".join(ch for ch in str(value) if ch not in invalid_chars).strip()
    return cleaned[:max_len] or "LSTM_주가예측"


def _number_or_blank(value, digits=None):
    if value is None or pd.isna(value):
        return ""
    num = float(value)
    if digits is not None:
        num = round(num, digits)
    return int(num) if float(num).is_integer() else num


def _load_service_account_info():
    for secret_name in ("gcp_service_account", "google_service_account"):
        try:
            value = st.secrets.get(secret_name)
            if value:
                return dict(value)
        except Exception:
            pass
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip() or _secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        if raw.startswith("{"):
            return json.loads(raw)
        json_path = Path(raw)
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
    raise GoogleSheetsConfigError(
        "Google 서비스 계정 정보가 없습니다. Streamlit secrets의 [gcp_service_account] 또는 "
        "GOOGLE_SERVICE_ACCOUNT_JSON 환경변수를 설정해 주세요."
    )


@st.cache_resource(show_spinner=False)
def get_google_sheets_client():
    if gspread is None or Credentials is None:
        raise GoogleSheetsConfigError(
            "gspread/google-auth 패키지가 설치되어 있지 않습니다. requirements.txt 반영 후 재배포해 주세요."
        )
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(_load_service_account_info(), scopes=scopes)
    return gspread.authorize(creds)


def open_or_create_prediction_spreadsheet(client, title):
    try:
        return client.open(title), False
    except Exception:
        try:
            spreadsheet = client.create(title, folder_id=GOOGLE_SHEET_FOLDER_ID or None)
        except TypeError:
            spreadsheet = client.create(title)
        if GOOGLE_SHEET_SHARE_EMAIL:
            spreadsheet.share(GOOGLE_SHEET_SHARE_EMAIL, perm_type="user", role="writer")
        return spreadsheet, True


def get_or_create_prediction_worksheet(spreadsheet):
    try:
        ws = spreadsheet.worksheet("예측기록")
    except Exception:
        ws = spreadsheet.add_worksheet(title="예측기록", rows=1000, cols=len(PREDICTION_HEADERS))
        ws.update(values=[PREDICTION_HEADERS], range_name="A1")
        ws.freeze(rows=1)
    values = ws.get_all_values()
    if not values or values[0] != PREDICTION_HEADERS:
        ws.update(values=[PREDICTION_HEADERS], range_name="A1")
    return ws


def build_prediction_sheet_rows(pred_table, stock_name, stock_code, base_date, close_by_date, metrics, model_settings, features):
    searched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    searched_date = searched_at[:10]
    rows = []
    for _, row in pred_table.iterrows():
        forecast_date = str(row["날짜"])
        predicted_close = _number_or_blank(row["예측 종가"])
        actual_close = close_by_date.get(forecast_date, "")
        error = ""
        error_pct = ""
        if actual_close != "" and predicted_close != "":
            error = float(actual_close) - float(predicted_close)
            error_pct = error / float(predicted_close) * 100 if float(predicted_close) else ""
        rows.append([
            searched_at, searched_date, stock_name, stock_code, base_date, forecast_date,
            predicted_close, _number_or_blank(row["전일대비"]), _number_or_blank(row["등락률(%)"], 2),
            _number_or_blank(actual_close), _number_or_blank(error), _number_or_blank(error_pct, 2),
            model_settings["lookback"], model_settings["pred_days"], model_settings["epochs"], model_settings["batch"],
            _number_or_blank(metrics["rmse"]), _number_or_blank(metrics["mae"]), _number_or_blank(metrics["mape"], 2),
            _number_or_blank(metrics["direction_accuracy"], 2), len(features), ", ".join(features),
        ])
    return rows


def refresh_actual_prices(values, stock_code, close_by_date):
    if not values:
        return values, 0
    headers = values[0]
    rows = values[1:]
    idx = {name: i for i, name in enumerate(headers)}
    required = ["종목코드", "예측일", "예측종가", "실제종가", "오차", "오차율(%)"]
    if any(col not in idx for col in required):
        return values, 0
    changed = 0
    for row in rows:
        while len(row) < len(headers):
            row.append("")
        if row[idx["종목코드"]] != stock_code:
            continue
        forecast_date = row[idx["예측일"]]
        actual = close_by_date.get(forecast_date)
        if actual in (None, ""):
            continue
        predicted = row[idx["예측종가"]]
        try:
            predicted_num = float(str(predicted).replace(",", ""))
            actual_num = float(actual)
        except Exception:
            continue
        error = actual_num - predicted_num
        error_pct = error / predicted_num * 100 if predicted_num else ""
        new_values = [str(int(actual_num)), str(int(error)), f"{error_pct:.2f}" if error_pct != "" else ""]
        target_cols = ["실제종가", "오차", "오차율(%)"]
        if any(row[idx[col]] != val for col, val in zip(target_cols, new_values)):
            for col, val in zip(target_cols, new_values):
                row[idx[col]] = val
            changed += 1
    return [headers] + rows, changed


def save_prediction_to_google_sheets(title, rows, stock_code, close_by_date):
    client = get_google_sheets_client()
    spreadsheet, created = open_or_create_prediction_spreadsheet(client, title)
    ws = get_or_create_prediction_worksheet(spreadsheet)
    values = ws.get_all_values()
    if not values:
        values = [PREDICTION_HEADERS]
    if values[0] != PREDICTION_HEADERS:
        values[0] = PREDICTION_HEADERS
    values, refreshed = refresh_actual_prices(values, stock_code, close_by_date)
    existing_rows = values[1:]
    key_cols = ["검색일자", "종목코드", "예측일"]
    idx = {name: i for i, name in enumerate(PREDICTION_HEADERS)}
    row_map = {tuple(row[idx[col]] for col in key_cols): i for i, row in enumerate(existing_rows) if len(row) >= len(PREDICTION_HEADERS)}
    upserted = 0
    for row in rows:
        row = ["" if value is None else value for value in row]
        key = tuple(str(row[idx[col]]) for col in key_cols)
        if key in row_map:
            existing_rows[row_map[key]] = row
        else:
            row_map[key] = len(existing_rows)
            existing_rows.append(row)
        upserted += 1
    ws.clear()
    ws.update(values=[PREDICTION_HEADERS] + existing_rows, range_name="A1", value_input_option="USER_ENTERED")
    ws.freeze(rows=1)
    return spreadsheet.url, upserted, refreshed, created


# ══════════════════════════════════════════════════════════
#  데이터 함수
# ══════════════════════════════════════════════════════════
def parse_corpcode_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    rows = [{'회사명':    c.findtext('corp_name',''),
             '종목코드':  c.findtext('stock_code','').zfill(6),
             'corp_code': c.findtext('corp_code','')}
            for c in root.findall('list')
            if len(c.findtext('stock_code','')) == 6]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="종목 목록 로딩 중...", ttl=86400)
def load_stock_list(dart_api_key):
    frames = []
    try:
        if dart_api_key:
            resp = requests.get(
                "https://opendart.fss.or.kr/api/corpCode.xml",
                params={"crtfc_key": dart_api_key},
                timeout=20,
            )
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                frames.append(parse_corpcode_xml(zf.read(zf.namelist()[0])))
    except Exception as e:
        st.warning(f"DART 종목 목록 갱신 실패 → 로컬/대체 목록 사용 ({e})")

    try:
        if not frames and LOCAL_CORPCODE_XML.exists():
            frames.append(parse_corpcode_xml(LOCAL_CORPCODE_XML.read_bytes()))
    except Exception as e:
        st.warning(f"로컬 CORPCODE.xml 로딩 실패 → KRX 대체 ({e})")

    try:
        if not frames:
            krx = fdr.StockListing('KRX')[['Code','Name']].dropna()
            krx.columns = ['종목코드','회사명']
            krx['corp_code'] = ''
            frames.append(krx[['회사명','종목코드','corp_code']])
    except Exception as e:
        st.warning(f"KRX 목록 실패 → 기본 종목만 사용 ({e})")

    frames.append(DEFAULT_STOCKS)
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=['회사명','종목코드'])
        .drop_duplicates('종목코드')
        .sort_values('회사명')
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="주가 데이터 수집 중...", ttl=3600)
def get_price_data(code, years=3):
    end = datetime.today(); start = end - timedelta(days=years*365)
    df = fdr.DataReader(code, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna().sort_index()


def _safe_float(value):
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return np.nan


def _ecos_period_bounds(period, s, e):
    start = pd.Timestamp(s)
    end = pd.Timestamp(e)
    if period == "M":
        return start.strftime("%Y%m"), end.strftime("%Y%m")
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


@st.cache_data(show_spinner=False, ttl=3600)
def get_ecos_series(stat, period, item, s, e, api_key):
    if not api_key:
        return pd.Series(dtype="float64")
    start, end = _ecos_period_bounds(period, s, e)
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/{stat}/{period}/{start}/{end}/{item}"
    try:
        data = requests.get(url, timeout=15).json()
        rows = data.get("StatisticSearch", {}).get("row", [])
        if not rows:
            return pd.Series(dtype="float64")
        values = []
        for row in rows:
            raw_date = str(row.get("TIME", ""))
            date_format = "%Y%m" if period == "M" else "%Y%m%d"
            date = pd.to_datetime(raw_date, format=date_format, errors="coerce")
            value = _safe_float(row.get("DATA_VALUE"))
            if pd.notna(date) and pd.notna(value):
                values.append((date, value))
        if not values:
            return pd.Series(dtype="float64")
        return pd.Series(dict(values)).sort_index().astype(float)
    except Exception:
        return pd.Series(dtype="float64")


def _fdr_close(ticker, s, e):
    try:
        df = fdr.DataReader(ticker, s, e)
        if df.empty:
            return pd.Series(dtype="float64")
        close_col = "Close" if "Close" in df.columns else df.columns[0]
        series = df[close_col].copy()
        series.index = pd.to_datetime(series.index)
        return pd.to_numeric(series, errors="coerce").dropna().sort_index()
    except Exception:
        return pd.Series(dtype="float64")


@st.cache_data(show_spinner=False, ttl=3600)
def get_naver_index_close(index_code, s, e, pages=80):
    frames = []
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://finance.naver.com/sise/sise_index_day.naver"
    start = pd.Timestamp(s)
    end = pd.Timestamp(e)
    try:
        for page in range(1, pages + 1):
            response = requests.get(
                url,
                params={"code": index_code, "page": page},
                headers=headers,
                timeout=12,
            )
            response.raise_for_status()
            table = pd.read_html(io.StringIO(response.text))[0].dropna()
            if table.empty:
                continue
            table = table.iloc[:, :6]
            table.columns = ["date", "close", "change", "open", "high", "low"]
            frames.append(table[["date", "close"]])

        if not frames:
            return pd.Series(dtype="float64")
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(
            df["close"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        df = (
            df.dropna(subset=["date", "close"])
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")
        )
        df = df[(df.index >= start) & (df.index <= end)]
        return df["close"].astype(float)
    except Exception:
        return pd.Series(dtype="float64")


@st.cache_data(show_spinner=False, ttl=3600)
def get_macro_bundle(s, e, ecos_api_key):
    series_map = {}
    source_map = {}

    usdkrw_ecos = get_ecos_series("731Y001", "D", "0000001", s, e, ecos_api_key)
    if not usdkrw_ecos.empty:
        series_map["usdkrw"] = usdkrw_ecos
        source_map["usdkrw"] = "ECOS"
    else:
        usdkrw_fdr = _fdr_close("USD/KRW", s, e)
        if not usdkrw_fdr.empty:
            series_map["usdkrw"] = usdkrw_fdr
            source_map["usdkrw"] = "FDR"

    vix = _fdr_close("^VIX", s, e)
    if not vix.empty:
        series_map["vix"] = vix
        source_map["vix"] = "FDR"

    kospi = get_naver_index_close("KOSPI", s, e)
    if not kospi.empty:
        series_map["kospi"] = kospi
        source_map["kospi"] = "Naver"
    else:
        kospi = _fdr_close("KS11", s, e)
        if not kospi.empty:
            series_map["kospi"] = kospi
            source_map["kospi"] = "FDR"

    kosdaq = get_naver_index_close("KOSDAQ", s, e)
    if not kosdaq.empty:
        series_map["kosdaq"] = kosdaq
        source_map["kosdaq"] = "Naver"

    ecos_extra_specs = {
        "base_rate": ("722Y001", "M", "0101000", "기준금리", "%"),
        "bond_3y": ("817Y002", "D", "010200000", "국채 3년", "%"),
        "bond_10y": ("817Y002", "D", "010210000", "국채 10년", "%"),
        "cpi": ("901Y009", "M", "0", "소비자물가", "%"),
    }
    ecos_extra = {}
    for key, (stat, period, item, label, unit) in ecos_extra_specs.items():
        series = get_ecos_series(stat, period, item, s, e, ecos_api_key)
        if not series.empty:
            ecos_extra[key] = {
                "label": label,
                "unit": unit,
                "series": series,
                "source": "ECOS",
            }

    if not series_map:
        return pd.DataFrame(), {}, {}

    macro = pd.concat(series_map, axis=1).ffill().bfill()
    macro.index = pd.to_datetime(macro.index)
    res = pd.DataFrame(index=macro.index)
    if 'usdkrw' in macro: res['환율변화']   = macro['usdkrw'].pct_change()
    if 'vix'    in macro:
        r = macro['vix'].rolling(60, min_periods=10)
        res['VIX정규화']  = (macro['vix'] - r.mean()) / (r.std() + 1e-9)
    if 'kospi'  in macro: res['KOSPI수익률'] = macro['kospi'].pct_change()
    if 'kosdaq' in macro: res['KOSDAQ수익률'] = macro['kosdaq'].pct_change()

    display = {}
    if "usdkrw" in macro:
        srs = macro["usdkrw"].dropna()
        if len(srs) >= 2:
            display["환율"] = {
                "label": "USD/KRW 환율",
                "value": float(srs.iloc[-1]),
                "delta": float(srs.iloc[-1] - srs.iloc[-2]),
                "rate": float((srs.iloc[-1] / srs.iloc[-2] - 1) * 100),
                "unit": "원",
                "source": source_map.get("usdkrw", ""),
            }
    if "vix" in macro:
        srs = macro["vix"].dropna()
        z = res["VIX정규화"].dropna() if "VIX정규화" in res else pd.Series(dtype=float)
        if len(srs) >= 2 and not z.empty:
            display["VIX"] = {
                "label": "VIX 공포지수",
                "value": float(srs.iloc[-1]),
                "delta": float(srs.iloc[-1] - srs.iloc[-2]),
                "zscore": float(z.iloc[-1]),
                "unit": "",
                "source": source_map.get("vix", ""),
            }
    if "kospi" in macro:
        srs = macro["kospi"].dropna()
        if len(srs) >= 2:
            display["KOSPI"] = {
                "label": "KOSPI",
                "value": float(srs.iloc[-1]),
                "delta": float(srs.iloc[-1] - srs.iloc[-2]),
                "rate": float((srs.iloc[-1] / srs.iloc[-2] - 1) * 100),
                "unit": "",
                "source": source_map.get("kospi", ""),
            }
    if "kosdaq" in macro:
        srs = macro["kosdaq"].dropna()
        if len(srs) >= 2:
            display["KOSDAQ"] = {
                "label": "KOSDAQ",
                "value": float(srs.iloc[-1]),
                "delta": float(srs.iloc[-1] - srs.iloc[-2]),
                "rate": float((srs.iloc[-1] / srs.iloc[-2] - 1) * 100),
                "unit": "",
                "source": source_map.get("kosdaq", ""),
            }
    for item in ecos_extra.values():
        srs = item["series"].dropna()
        if len(srs) >= 1:
            key = item["label"]
            prev = srs.iloc[-2] if len(srs) >= 2 else srs.iloc[-1]
            display[key] = {
                "label": item["label"],
                "value": float(srs.iloc[-1]),
                "delta": float(srs.iloc[-1] - prev),
                "unit": item["unit"],
                "source": item["source"],
            }

    return res.ffill().bfill(), display, source_map


@st.cache_data(show_spinner=False, ttl=86400)
def get_dart_fundamentals(corp_code, s, e, dart_api_key):
    if not corp_code or not dart_api_key:
        return pd.DataFrame()
    records = []
    for year in range(int(s[:4])-2, int(e[:4])+1):
        for rc, mo in [('11013',3),('11012',6),('11014',9),('11011',12)]:
            try:
                r = requests.get("https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
                                  params={'crtfc_key':dart_api_key,'corp_code':corp_code,
                                          'bsns_year':str(year),'reprt_code':rc,'fs_div':'CFS'},
                                  timeout=8)
                rev = op = None
                for it in r.json().get('list',[]):
                    try: v = float(it.get('thstrm_amount','').replace(',',''))
                    except: continue
                    if it.get('account_nm') == '매출액':   rev = v
                    elif it.get('account_nm') == '영업이익': op  = v
                if rev and rev != 0 and op is not None:
                    rd = pd.Timestamp(year=year, month=mo, day=28) + pd.Timedelta(days=60)
                    records.append({'date': rd, 'revenue': rev, 'op': op})
            except Exception:
                pass
    if len(records) < 3:
        return pd.DataFrame()
    df_q = pd.DataFrame(records).dropna().sort_values('date').set_index('date')
    df_q['영업이익률']   = (df_q['op'] / df_q['revenue']).clip(-1, 1)
    df_q['YoY매출성장'] = df_q['revenue'].pct_change(4).clip(-2, 2)
    idx = pd.bdate_range(start=s, end=e)
    return df_q[['영업이익률','YoY매출성장']].reindex(idx).ffill().dropna()


def add_features(df):
    d = df.copy()
    d['ma5']       = d['close'].rolling(5).mean()
    d['ma20']      = d['close'].rolling(20).mean()
    d['ma60']      = d['close'].rolling(60).mean()
    d['bb_mid']    = d['close'].rolling(20).mean()
    d['bb_std']    = d['close'].rolling(20).std()
    d['bb_upper']  = d['bb_mid'] + 2*d['bb_std']
    d['bb_lower']  = d['bb_mid'] - 2*d['bb_std']
    delta = d['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi']         = 100 - 100/(1 + gain/loss.replace(0,1e-9))
    ema12 = d['close'].ewm(span=12,adjust=False).mean()
    ema26 = d['close'].ewm(span=26,adjust=False).mean()
    d['macd']        = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9,adjust=False).mean()
    d['macd_hist']   = d['macd'] - d['macd_signal']
    d['vol_log']     = np.log1p(d['volume'])
    return d.dropna()


def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i]); y.append(data[i,0])
    return np.array(X), np.array(y)


def build_model(seq_len, n_feat):
    m = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_len, n_feat)),
        Dropout(0.2), Bidirectional(LSTM(32)), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return m


# ══════════════════════════════════════════════════════════
#  사이드바
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 모델 설정")
    lookback   = st.slider("📅 과거 입력일 수",    20, 120,  60, 10)
    pred_days  = st.slider("🔮 미래 예측일 수",    10,  90,  30,  5)
    epochs     = st.slider("🔁 학습 에포크(최대)", 20, 200, 100, 10)
    batch_sz   = st.slider("📦 배치 크기",         16, 128,  32, 16)
    val_split  = st.slider("✂️ 검증 비율 (%)",     10,  30,  20,  5) / 100
    data_years = st.slider("🗓️ 데이터 기간(년)",    1,   5,   3,  1)

    st.markdown("---")
    st.subheader("🧩 추가 피처")
    use_macro = st.toggle("🌐 거시경제 (ECOS·VIX·KOSPI·KOSDAQ)", value=True)
    use_dart  = st.toggle("📋 DART 재무 (영업이익률·YoY)", value=True)

    st.markdown("---")
    st.subheader("🔌 로컬 API")
    api_rows = [
        ("ECOS", ECOS_API_KEY),
        ("DART", DART_API_KEY),
        ("KRX", KRX_API_KEY),
        ("공공데이터", PUBLIC_DATA_API_KEY),
    ]
    for api_name, api_key in api_rows:
        st.caption(f"{'✅' if api_key else '⚪'} {api_name}: {'사용 가능' if api_key else '키 없음'}")
    st.markdown("---")
    st.subheader("Google Sheets 저장")
    auto_save_sheets = st.toggle(
        "예측 결과 자동 저장",
        value=GOOGLE_SHEETS_AUTO_SAVE_DEFAULT,
        help="설정된 Google 서비스 계정으로 LSTM_종목명_주가예측 시트에 검색일자 기준 저장/갱신합니다.",
    )
    save_sheets_now = st.button("현재 결과 저장/갱신", use_container_width=True)

    st.markdown("---")
    st.caption("EarlyStopping patience=15")
    st.caption("ReduceLR patience=7 · factor=0.5")
    if st.button("🔄 캐시 초기화", use_container_width=True):
        st.cache_data.clear(); st.rerun()


# ══════════════════════════════════════════════════════════
#  헤더 & 종목 검색
# ══════════════════════════════════════════════════════════
st.markdown('<h1 style="font-size:32px;margin-bottom:4px;">📈 LSTM 주가 예측 대시보드</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#7890ab;font-size:13px;margin-bottom:20px;">가격 기술지표 + 거시경제 + DART 재무 통합 멀티피처 모델</p>', unsafe_allow_html=True)

stock_df = load_stock_list(DART_API_KEY)

col_q, col_sel = st.columns([1, 2])
with col_q:
    query = st.text_input("🔍 종목 검색", value="삼성전자", label_visibility="collapsed",
                          placeholder="종목명 또는 코드 입력...")
    direct_code = st.text_input("종목코드 직접 입력", value="", placeholder="예: 005930")
matched = stock_df[
    stock_df['회사명'].str.contains(query, na=False) |
    stock_df['종목코드'].str.contains(query, na=False)
].head(50)

if matched.empty:
    clean_code = ''.join(ch for ch in direct_code if ch.isdigit())[:6]
    if len(clean_code) == 6:
        sel_name, code, corp_code = "직접 입력", clean_code, ""
    else:
        st.warning("일치하는 종목이 없습니다. 6자리 종목코드를 직접 입력해 주세요."); st.stop()
else:
    with col_sel:
        options  = matched.apply(lambda r: f"{r['회사명']}  ({r['종목코드']})", axis=1).tolist()
        selected = st.selectbox("종목 선택", options, label_visibility="collapsed")

    sel_name  = selected.split('  (')[0]
    code      = selected.split('(')[1].rstrip(')')
    clean_code = ''.join(ch for ch in direct_code if ch.isdigit())[:6]
    if len(clean_code) == 6:
        sel_name, code = "직접 입력", clean_code

    corp_row  = stock_df[stock_df['종목코드'] == code]
    corp_code = corp_row['corp_code'].iloc[0] if not corp_row.empty and 'corp_code' in corp_row.columns else ''


# ══════════════════════════════════════════════════════════
#  데이터 수집
# ══════════════════════════════════════════════════════════
with st.spinner(f"📡 {sel_name} 데이터 수집 중..."):
    try:
        raw = get_price_data(code, years=data_years)
    except Exception as e:
        st.error(f"주가 데이터 수집 실패: {e}"); st.stop()

if raw.empty or len(raw) < lookback + 40:
    st.error(f"데이터 부족 ({len(raw)}일). 캐시 초기화 후 재시도하세요."); st.stop()

df = add_features(raw)
ds, de = df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')
BASE_FEATURES = ['close','vol_log','ma5','ma20','rsi']
extra_cols = []
macro_status = {}; macro_display = {}; dart_status = {}

if use_macro:
    with st.spinner("🌐 거시경제 데이터 수집 중..."):
        mdf, macro_display, macro_sources = get_macro_bundle(ds, de, ECOS_API_KEY)
    if not mdf.empty:
        al    = mdf.reindex(df.index, method='ffill').ffill().bfill()
        valid = [c for c in al.columns if al[c].notna().sum() > lookback*2]
        if valid:
            df = pd.concat([df, al[valid]], axis=1)
            extra_cols.extend(valid)
            macro_status = {c: float(al[c].iloc[-1]) for c in valid}
    else:
        st.warning("거시경제 데이터를 불러오지 못했습니다. ECOS/FDR 연결 상태를 확인해 주세요.")

if use_dart and corp_code and DART_API_KEY:
    with st.spinner("📋 DART 재무 데이터 수집 중..."):
        ddf = get_dart_fundamentals(corp_code, ds, de, DART_API_KEY)
    if not ddf.empty:
        al    = ddf.reindex(df.index, method='ffill').ffill().bfill()
        valid = [c for c in al.columns if al[c].notna().sum() > lookback*2]
        if valid:
            df = pd.concat([df, al[valid]], axis=1)
            extra_cols.extend(valid)
            dart_status = {c: float(al[c].iloc[-1]) for c in valid}
elif use_dart and corp_code and not DART_API_KEY:
    st.info("DART API 키가 없어 재무 피처를 건너뜁니다. 환경변수 DART_API_KEY 또는 Streamlit secrets에 키를 넣으면 활성화됩니다.")
elif use_dart and not corp_code:
    st.info("이 종목은 DART corp_code를 찾을 수 없어 재무 피처를 건너뜁니다.")

FEATURES = BASE_FEATURES + extra_cols
df       = df.dropna(subset=FEATURES)

latest = df.iloc[-1]; prev = df.iloc[-2]
pchg  = latest['close'] - prev['close']
prate = pchg / prev['close'] * 100
arr   = "▲" if pchg > 0 else "▼"
pchg_str = f"{arr} {abs(pchg):,.0f}원 ({prate:+.2f}%)"

# ══════════════════════════════════════════════════════════
#  종목 타이틀 바
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="stock-title">
  <span class="stock-name">{sel_name}</span>
  <span class="stock-code">{code}</span>
  <div class="stock-sub">
    데이터 기간: {ds} ~ {de} &nbsp;|&nbsp; {len(df)}거래일
    &nbsp;|&nbsp; 활성 피처 {len(FEATURES)}개
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  KPI 패널 1: 주가 현황
# ══════════════════════════════════════════════════════════
section("주가 현황", "💹")
c1, c2, c3, c4, c5 = st.columns(5)

c1.markdown(kpi_card("현재가 (최근 종가)",
                     f"{int(latest['close']):,}원", pchg_str), unsafe_allow_html=True)
c2.markdown(kpi_card("52주 최고가",
                     f"{int(df['close'].tail(252).max()):,}원"), unsafe_allow_html=True)
c3.markdown(kpi_card("52주 최저가",
                     f"{int(df['close'].tail(252).min()):,}원"), unsafe_allow_html=True)

rsi_v  = latest['rsi']
rsi_st = "⚠️ 과매수" if rsi_v>70 else ("📉 과매도" if rsi_v<30 else "✅ 중립")
c4.markdown(kpi_card("RSI (14일)", f"{rsi_v:.1f}", rsi_st), unsafe_allow_html=True)

macd_dir = "▲ 상향" if latest['macd'] > latest['macd_signal'] else "▼ 하향"
c5.markdown(kpi_card("MACD 방향", macd_dir,
                     f"MACD: {latest['macd']:+.1f}"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  KPI 패널 2: 거시경제 지표
# ══════════════════════════════════════════════════════════
if macro_display:
    section("거시경제 지표", "🌐")
    icon_map = {
        "환율": "💱",
        "VIX": "😱",
        "KOSPI": "📊",
        "KOSDAQ": "📊",
        "기준금리": "🏦",
        "국채 3년": "📉",
        "국채 10년": "📈",
        "소비자물가": "🧾",
    }
    help_map = {
        "환율": "ECOS 우선, 실패 시 FDR. 원/달러 환율 원값과 전일 대비입니다.",
        "VIX": "FDR. 글로벌 공포지수 원값과 최근 60거래일 기준 Z-score입니다.",
        "KOSPI": "네이버 금융 우선, 실패 시 FDR. KOSPI 지수 원값과 전일 등락률입니다.",
        "KOSDAQ": "네이버 금융. KOSDAQ 지수 원값과 전일 등락률입니다.",
        "기준금리": "ECOS. 한국은행 기준금리 최신 월 자료입니다.",
        "국채 3년": "ECOS. 국고채 3년 금리 최신 일 자료입니다.",
        "국채 10년": "ECOS. 국고채 10년 금리 최신 일 자료입니다.",
        "소비자물가": "ECOS. 소비자물가 지표 최신 월 자료입니다.",
    }
    ordered_keys = ["환율", "VIX", "KOSPI", "KOSDAQ", "기준금리", "국채 3년", "국채 10년", "소비자물가"]
    display_items = [(k, macro_display[k]) for k in ordered_keys if k in macro_display]
    for start in range(0, len(display_items), 4):
        mcols = st.columns(min(4, len(display_items) - start))
        for col, (k, item) in zip(mcols, display_items[start:start+4]):
            source = item.get("source", "")
            label = f"{icon_map.get(k, '📌')} {item.get('label', k)}"
            if source:
                label = f"{label} · {source}"
            value = item.get("value", np.nan)
            delta = item.get("delta", np.nan)
            if k == "환율":
                rate = item.get("rate", 0.0)
                sign = "▲" if delta > 0 else ("▼" if delta < 0 else "━")
                value_str = f"{value:,.1f}원"
                delta_str = f"{sign} {abs(delta):,.1f}원 ({rate:+.2f}%)"
            elif k == "VIX":
                zscore = item.get("zscore", 0.0)
                sign = "▲" if delta > 0 else ("▼" if delta < 0 else "━")
                value_str = f"{value:,.2f}"
                delta_str = f"{sign} {abs(delta):,.2f} · Z {zscore:+.2f}"
            elif k in ["KOSPI", "KOSDAQ"]:
                rate = item.get("rate", 0.0)
                sign = "▲" if delta > 0 else ("▼" if delta < 0 else "━")
                value_str = f"{value:,.2f}"
                delta_str = f"{sign} {abs(delta):,.2f}p ({rate:+.2f}%)"
            else:
                unit = item.get("unit", "")
                sign = "▲" if delta > 0 else ("▼" if delta < 0 else "━")
                value_str = f"{value:,.2f}{unit}"
                delta_str = f"{sign} {abs(delta):,.2f}{unit}"
            col.markdown(
                kpi_card(label, value_str, delta_str, card="macro", help_txt=help_map.get(k, "")),
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════
#  KPI 패널 3: DART 재무지표
# ══════════════════════════════════════════════════════════
if dart_status:
    section("DART 재무지표", "📋")
    helps_d = {
        '영업이익률':  '영업이익/매출액. 10%↑=양호, 20%↑=우수 (공시 60일 지연 반영)',
        'YoY매출성장': '전년 동기 대비 매출 성장률. 양수=성장세',
    }
    dcols = st.columns(max(2, len(dart_status)))
    for col, (k, v) in zip(dcols, dart_status.items()):
        pct_str = f"{v:+.2%}"
        sign = "▲" if v > 0 else "▼"
        col.markdown(kpi_card(f"📋 {k}", pct_str,
                               f"{sign} {abs(v):.2%}", card="dart",
                               help_txt=helps_d.get(k, '')), unsafe_allow_html=True)

# 피처 뱃지
st.markdown('<br>', unsafe_allow_html=True)
badges = ""
for f in FEATURES:
    cls = "macro" if f in macro_status else ("dart" if f in dart_status else "")
    badges += f'<span class="badge {cls}">{f}</span>'
st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  데이터 탭
# ══════════════════════════════════════════════════════════
with st.expander("📊 데이터 미리보기 & 피처 상관관계"):
    t1, t2, t3 = st.tabs(["최근 OHLCV", "기술 지표", "피처 상관관계 히트맵"])
    with t1:
        s_ = df[['close','open','high','low','volume']].tail(20).copy()
        s_.index = s_.index.strftime('%Y-%m-%d')
        st.dataframe(s_.style.format({c:'{:,.0f}' for c in s_.columns}), use_container_width=True)
    with t2:
        c2_ = [c for c in ['ma5','ma20','ma60','rsi','macd','bb_upper','bb_lower']+extra_cols if c in df.columns]
        s2_ = df[c2_].tail(20).copy(); s2_.index = s2_.index.strftime('%Y-%m-%d')
        st.dataframe(s2_.style.format({c:('{:+.4f}' if c in extra_cols else '{:,.2f}') for c in c2_}), use_container_width=True)
    with t3:
        cc_ = [c for c in FEATURES if c in df.columns]
        corr = df[cc_].tail(252).corr()
        fig_c, ax_c = plt.subplots(figsize=(max(6,len(cc_)), max(5,len(cc_)-1)))
        im = ax_c.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        ax_c.set_xticks(range(len(cc_))); ax_c.set_xticklabels(cc_, rotation=45, ha='right', fontsize=9)
        ax_c.set_yticks(range(len(cc_))); ax_c.set_yticklabels(cc_, fontsize=9)
        for i in range(len(cc_)):
            for j in range(len(cc_)):
                ax_c.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=7,
                          color='black' if abs(corr.values[i,j]) < 0.7 else 'white')
        plt.colorbar(im, ax=ax_c); ax_c.set_title('피처 상관관계 (최근 252거래일)', fontsize=11)
        plt.tight_layout(); st.pyplot(fig_c)

# ══════════════════════════════════════════════════════════
#  캔들 차트
# ══════════════════════════════════════════════════════════
section("최근 120거래일 차트", "📉")
cdf = df.tail(120)
fig1, (ap, av, am) = plt.subplots(3, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios':[4,1,1.5]}, sharex=True)
for dt, row in cdf.iterrows():
    up  = row['close'] >= row['open']
    clr = '#f87171' if up else '#60a5fa'
    ap.plot([dt,dt],[row['low'],row['high']], color=clr, lw=0.8)
    ap.bar(dt, abs(row['close']-row['open']), bottom=min(row['open'],row['close']),
           color=clr, width=0.6, alpha=0.9)
ap.plot(cdf.index, cdf['ma5'],     label='MA5',  color='#fbbf24', lw=1, ls='--')
ap.plot(cdf.index, cdf['ma20'],    label='MA20', color='#a78bfa', lw=1, ls='--')
ap.plot(cdf.index, cdf['ma60'],    label='MA60', color='#34d399', lw=1, ls='--')
ap.fill_between(cdf.index, cdf['bb_upper'], cdf['bb_lower'], alpha=0.07, color='#94a3b8')
ap.plot(cdf.index, cdf['bb_upper'], color='#94a3b8', lw=0.7, ls=':')
ap.plot(cdf.index, cdf['bb_lower'], color='#94a3b8', lw=0.7, ls=':', label='BB')
ap.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
ap.set_title(f'{sel_name} ({code})', fontsize=13, fontweight='bold', color='#e0eeff')
ap.legend(loc='upper left', fontsize=8, framealpha=0.3); ap.grid(True, alpha=0.15)
ap.set_facecolor('#0d1117'); ap.tick_params(colors='#7890ab')

vc = ['#f87171' if c>=o else '#60a5fa' for c,o in zip(cdf['close'],cdf['open'])]
av.bar(cdf.index, cdf['volume'], color=vc, alpha=0.8, width=0.6)
av.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x/1e4):,}만'))
av.set_ylabel('거래량', fontsize=8, color='#7890ab')
av.grid(True, alpha=0.15); av.set_facecolor('#0d1117'); av.tick_params(colors='#7890ab')

am.plot(cdf.index, cdf['macd'],        label='MACD',   color='#60a5fa', lw=1.2)
am.plot(cdf.index, cdf['macd_signal'], label='Signal', color='#f87171', lw=1.2)
mc_ = ['#f87171' if v>=0 else '#60a5fa' for v in cdf['macd_hist']]
am.bar(cdf.index, cdf['macd_hist'], color=mc_, alpha=0.5, width=0.6)
am.axhline(0, color='#475569', lw=0.5)
am.set_ylabel('MACD', fontsize=8, color='#7890ab')
am.legend(loc='upper left', fontsize=8, framealpha=0.3)
am.grid(True, alpha=0.15); am.set_facecolor('#0d1117'); am.tick_params(colors='#7890ab')
am.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
am.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
fig1.patch.set_facecolor('#0d1117')
plt.xticks(rotation=45, fontsize=8); plt.tight_layout()
st.pyplot(fig1)


# ══════════════════════════════════════════════════════════
#  LSTM 학습
# ══════════════════════════════════════════════════════════
section("LSTM 모델 학습", "🤖")
feat_data = df[FEATURES].values.astype("float32")
n_total_samples = len(feat_data) - lookback
n_val  = max(1, int(n_total_samples * val_split))
split_idx = len(feat_data) - n_val
if split_idx <= lookback:
    st.error("검증 비율이 너무 커서 학습 샘플이 부족합니다. 검증 비율을 낮추거나 데이터 기간을 늘려 주세요.")
    st.stop()

scaler = MinMaxScaler()
scaler.fit(feat_data[:split_idx])
scaled = scaler.transform(feat_data)
X, y = make_sequences(scaled, lookback)
train_count = split_idx - lookback
X_tr, X_v = X[:train_count], X[train_count:]
y_tr, y_v = y[:train_count], y[train_count:]
target_dates = df.index[lookback:]
val_dates = target_dates[train_count:]

close_scaler = MinMaxScaler()
close_scaler.fit(df[['close']].values[:split_idx])

info_cols = st.columns(4)
info_cols[0].info(f"**피처 수**: {len(FEATURES)}개")
info_cols[1].info(f"**학습 샘플**: {len(X_tr):,}개")
info_cols[2].info(f"**검증 샘플**: {len(X_v):,}개")
info_cols[3].info(f"**시퀀스 길이**: {lookback}일")

bar = st.progress(0, text="모델 초기화 중...")

with st.spinner("학습 중..."):
    model = build_model(lookback, len(FEATURES))

    class _CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            pct = min(int((ep+1)/epochs*100), 100)
            bar.progress(pct, text=f"에포크 {ep+1}/{epochs}  |  val_loss: {logs.get('val_loss',0):.6f}")

    history = model.fit(
        X_tr, y_tr, validation_data=(X_v, y_v),
        epochs=epochs, batch_size=batch_sz,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
            _CB(),
        ], verbose=0)

ae = len(history.history['loss'])
bar.progress(100, text=f"✅ 학습 완료  {ae}/{epochs} 에포크 (EarlyStopping 적용)")

with st.expander("📉 학습 곡선 보기"):
    fl, al = plt.subplots(figsize=(10, 3))
    al.plot(history.history['loss'],     label='Train Loss', color='#60a5fa')
    al.plot(history.history['val_loss'], label='Val Loss',   color='#f87171')
    al.set_xlabel('Epoch'); al.set_ylabel('MSE Loss')
    al.legend(framealpha=0.3); al.grid(True, alpha=0.2)
    fl.patch.set_facecolor('#0d1117'); al.set_facecolor('#0d1117')
    al.tick_params(colors='#94a3b8'); plt.tight_layout()
    st.pyplot(fl)


# ══════════════════════════════════════════════════════════
#  정확도 패널
# ══════════════════════════════════════════════════════════
section("모델 정확도", "📊")

def inv(a): return close_scaler.inverse_transform(np.asarray(a).reshape(-1,1))

vp = inv(model.predict(X_v, verbose=0).flatten())
va = inv(y_v)
rmse   = np.sqrt(mean_squared_error(va, vp))
mae    = mean_absolute_error(va, vp)
mape   = np.mean(np.abs((va - vp)/(va + 1e-9))) * 100
da     = np.mean(np.sign(np.diff(va.flatten())) == np.sign(np.diff(vp.flatten()))) * 100

a1, a2, a3, a4, a5 = st.columns(5)
a1.markdown(kpi_card("RMSE",        f"{rmse:,.0f}원",   card="acc"), unsafe_allow_html=True)
a2.markdown(kpi_card("MAE",         f"{mae:,.0f}원",    card="acc"), unsafe_allow_html=True)
a3.markdown(kpi_card("MAPE",        f"{mape:.2f}%",     card="acc"), unsafe_allow_html=True)
a4.markdown(kpi_card("가격 정확도", f"{max(0,100-mape):.1f}%", card="acc"), unsafe_allow_html=True)
a5.markdown(kpi_card("🎯 방향 정확도", f"{da:.1f}%",
                     "✅ 우수" if da>=60 else ("📈 유의미" if da>=55 else "⚠️ 낮음"),
                     card="acc"), unsafe_allow_html=True)

if da >= 60:
    st.success(f"🎯 방향 예측 **{da:.1f}%** — 우수한 수준! (피처 {len(FEATURES)}개 적용)")
elif da >= 55:
    st.info(f"📈 방향 예측 **{da:.1f}%** — 통계적으로 유의미합니다.")
else:
    st.warning(f"⚠️ 방향 예측 **{da:.1f}%** — lookback 조정 또는 피처 추가를 권장합니다.")


# ══════════════════════════════════════════════════════════
#  미래 예측 & 결과 차트
# ══════════════════════════════════════════════════════════
section("미래 예측", "🔮")
cur = scaled[-lookback:].copy()
fsc = []
for _ in range(pred_days):
    p = model.predict(cur.reshape(1, lookback, len(FEATURES)), verbose=0)[0,0]
    fsc.append(p); nxt = cur[-1].copy(); nxt[0] = p
    cur = np.vstack([cur[1:], nxt])

fc = inv(np.array(fsc))
fd = pd.date_range(start=df.index[-1]+pd.tseries.offsets.BDay(1), periods=pred_days, freq='B')

vd = val_dates
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(df.index[-180:], df['close'].values[-180:],
         label='실제 주가', color='#60a5fa', lw=1.5)
ax2.plot(vd, va.flatten(), label='검증 실제', color='#60a5fa', lw=1.5, alpha=0.4)
ax2.plot(vd, vp.flatten(), label='검증 예측', color='#fbbf24', lw=1.5, ls='--')
ax2.plot(fd, fc.flatten(), label=f'미래 예측({pred_days}일)', color='#34d399', lw=2.5)
residual_pct = np.abs((va.flatten() - vp.flatten()) / (va.flatten() + 1e-9))
band_pct = float(np.clip(np.nanmedian(residual_pct), 0.02, 0.15))
ax2.fill_between(fd, fc.flatten()*(1-band_pct), fc.flatten()*(1+band_pct),
                 alpha=0.15, color='#34d399', label=f'±{band_pct*100:.1f}% 검증 오차 구간')
ax2.axvline(df.index[-1], color='#475569', lw=1.2, ls='--', alpha=0.7)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
ax2.set_title(f'{sel_name}  |  LSTM 예측 (피처 {len(FEATURES)}개)',
              fontsize=13, fontweight='bold', color='#e0eeff')
ax2.legend(loc='upper left', framealpha=0.3); ax2.grid(True, alpha=0.15)
ax2.set_facecolor('#0d1117'); fig2.patch.set_facecolor('#0d1117')
ax2.tick_params(colors='#94a3b8')
plt.xticks(rotation=45); plt.tight_layout()
st.pyplot(fig2)

# ── 요약 배너 ────────────────────────────────────────────
last_close  = int(df['close'].iloc[-1])
final_price = int(fc.flatten()[-1])
chg_pct     = (final_price - last_close) / last_close * 100
arr2 = "▲" if chg_pct > 0 else "▼"
clr2 = "#f87171" if chg_pct > 0 else "#60a5fa"
st.markdown(f"""
<div style="background:#1c2536; border-radius:12px; padding:20px 28px;
            border-left:5px solid {clr2}; margin:12px 0;">
  <span style="font-size:15px; color:#94a3b8;">현재가</span>
  <span style="font-size:22px; font-weight:800; color:#f0f4ff; margin:0 12px;">{last_close:,}원</span>
  <span style="font-size:15px; color:#94a3b8;">→  {pred_days}일 후 예측가</span>
  <span style="font-size:26px; font-weight:800; color:{clr2}; margin:0 12px;">{final_price:,}원</span>
  <span style="font-size:18px; color:{clr2};">{arr2} {abs(chg_pct):.1f}%</span>
</div>
""", unsafe_allow_html=True)

# ── 예측 테이블 ───────────────────────────────────────────
pred_df = pd.DataFrame({
    '날짜':      fd.strftime('%Y-%m-%d'),
    '예측 종가': fc.flatten().astype(int),
})
pred_df['전일대비'] = pred_df['예측 종가'].diff().fillna(
    pred_df['예측 종가'].iloc[0]-last_close).astype(int)
pred_df['등락률(%)'] = (
    pred_df['전일대비']/pred_df['예측 종가'].shift(1).fillna(last_close)*100).round(2)

st.dataframe(
    pred_df.style
        .format({'예측 종가':'{:,}','전일대비':'{:+,}','등락률(%)':'{:+.2f}'})
        .map(lambda v: 'color:#f87171' if isinstance(v,(int,float)) and v>0 else
                       ('color:#60a5fa' if isinstance(v,(int,float)) and v<0 else ''),
             subset=['전일대비','등락률(%)']),
    use_container_width=True,
    height=min(400, 36*(len(pred_df)+1)),
)

csv = pred_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
st.download_button("⬇️ 예측 결과 CSV 다운로드", data=csv,
                   file_name=f"{code}_{sel_name}_{datetime.today().strftime('%Y%m%d')}.csv",
                   mime="text/csv", use_container_width=True)

close_by_date = {idx.strftime('%Y-%m-%d'): int(value) for idx, value in df['close'].items()}
metrics = {"rmse": rmse, "mae": mae, "mape": mape, "direction_accuracy": da}
model_settings = {"lookback": lookback, "pred_days": pred_days, "epochs": epochs, "batch": batch_sz}
sheet_title = _sheet_safe_title(f"LSTM_{sel_name}_{code}_주가예측")
prediction_rows = build_prediction_sheet_rows(
    pred_df, sel_name, code, df.index[-1].strftime('%Y-%m-%d'), close_by_date, metrics, model_settings, FEATURES
)

if auto_save_sheets or save_sheets_now:
    try:
        sheet_url, saved_count, refreshed_count, created_sheet = save_prediction_to_google_sheets(sheet_title, prediction_rows, code, close_by_date)
        action = "생성" if created_sheet else "갱신"
        st.success(f"Google Sheets {action} 완료: 예측 {saved_count}행 저장, 실제 종가 {refreshed_count}행 갱신")
        st.link_button("Google Sheets 열기", sheet_url, use_container_width=True)
    except GoogleSheetsConfigError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Google Sheets 저장 실패: {e}")


# ══════════════════════════════════════════════════════════
#  피처 설명
# ══════════════════════════════════════════════════════════
desc_map = {
    'close':     ('🔵 기술', '종가 (예측 대상)'),
    'vol_log':   ('🔵 기술', '거래량 log — 비정상 급등 포착'),
    'ma5':       ('🔵 기술', '5일 이동평균 — 단기 추세'),
    'ma20':      ('🔵 기술', '20일 이동평균 — 중기 추세'),
    'rsi':       ('🔵 기술', 'RSI(14) — 과매수/과매도 판단'),
    '환율변화':   ('🟣 거시', 'USD/KRW 일변화율 — 외국인 수급 선행'),
    'VIX정규화':  ('🟣 거시', 'VIX 정규화 — 글로벌 리스크'),
    'KOSPI수익률':('🟣 거시', 'KOSPI 수익률 — 시장 전체 흐름'),
    'KOSDAQ수익률':('🟣 거시', 'KOSDAQ 수익률 — 성장주/중소형주 분위기'),
    '영업이익률':  ('🟢 DART', '영업이익률 — 기업 수익성 (공시 60일 지연)'),
    'YoY매출성장': ('🟢 DART', 'YoY 매출성장률 — 4분기 대비 성장세'),
}
with st.expander("ℹ️ 활성 피처 설명"):
    rows = [{'구분': desc_map.get(f,('⚪',''))[0],
             '피처명': f,
             '현재값': f"{float(df[f].iloc[-1]):+.4f}" if f in df.columns else '-',
             '설명': desc_map.get(f,('','?'))[1]}
            for f in FEATURES]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
