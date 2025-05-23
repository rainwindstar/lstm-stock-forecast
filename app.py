# Streamlit 기반 LSTM 주가 예측 대시보드 (OpenDART 기업코드 API 적용 + 네이버 코드 변환)

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import requests
import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# 시드 고정
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

st.title("📈 LSTM 기반 주가 예측 대시보드")

# 🔐 OpenDART API KEY
DART_API_KEY = 'ea6f080a6e93838fee1467220df5cbdccce35ecc'

@st.cache_data(show_spinner=False)
def load_stock_list():
    zip_url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    zip_path = "CORPCODE.zip"
    try:
        with open(zip_path, "wb") as f:
            f.write(requests.get(zip_url).content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        tree = ET.parse("CORPCODE.xml")
        root = tree.getroot()
        rows = []
        for child in root:
            name = child.find('corp_name').text
            code = child.find('stock_code').text
            if code and len(code) == 6:
                rows.append({'회사명': name, '종목코드': code.zfill(6)})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame(columns=['회사명', '종목코드'])

stock_df = load_stock_list()
if stock_df.empty:
    st.error("❌ 종목 리스트를 불러올 수 없습니다. OpenDART API 키 또는 네트워크 확인 필요")
    st.stop()

# 종목 검색
query = st.text_input("🔍 종목명 또는 코드 검색:", value="씨젠")
matched = stock_df[stock_df['회사명'].str.contains(query) | stock_df['종목코드'].astype(str).str.contains(query)]

if matched.empty:
    st.warning("일치하는 종목이 없습니다.")
    st.stop()

selected_row = st.selectbox("📌 예측할 종목을 선택하세요:", matched.apply(lambda row: f"{row['회사명']} ({row['종목코드']})", axis=1))
selected_name, code = selected_row.split(' (')
code = code.rstrip(')')

# 네이버용 코드 (종목코드 + .KS or .KQ)
if code.startswith('0') or code.startswith('1'):
    naver_code = code + ""
else:
    naver_code = code

# LSTM 설정
st.sidebar.header("⚙️ LSTM 모델 설정")
lookback_days = st.sidebar.slider("과거 입력일 수 (lookback days)", 30, 180, 120, 10)
predict_days = st.sidebar.slider("미래 예측일 수 (predict days)", 30, 180, 120, 10)
compare_days = st.sidebar.slider("과거 예측 정확도 비교일 수", 30, 180, 90, 10)
epochs = st.sidebar.slider("에포크 수 (학습 반복 수)", 5, 100, 10, 5)
batch_size = st.sidebar.slider("배치 크기", 8, 128, 32, 8)

# 네이버 금융 크롤링
@st.cache_data(show_spinner=False)
def get_naver_data(code, max_pages=30):
    url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    df = pd.DataFrame()
    for page in range(1, max_pages + 1):
        try:
            res = requests.get(f"{url}&page={page}", headers=headers)
            soup = BeautifulSoup(res.text, 'lxml')
            table = soup.select_one("table.type2")
            temp_df = pd.read_html(str(table))[0]
            df = pd.concat([df, temp_df])
        except:
            continue
    df = df.dropna()
    df.columns = ['date', 'close', 'diff', 'open', 'high', 'low', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'close']].dropna()
    df.set_index('date', inplace=True)
    df = df.astype({'close': 'float'})
    df.sort_index(inplace=True)
    return df

st.write("### 📅 네이버 금융에서 데이터 로딩 중...")
try:
    data = get_naver_data(naver_code)
    if data.empty or len(data) < 150:
        raise ValueError("❌ 수집된 데이터가 충분하지 않습니다. (150일 이상 필요)")
except Exception as e:
    st.error(f"❌ 종목 데이터를 불러올 수 없습니다. 선택한 코드: {code}\n\n오류 내용: {e}")
    st.stop()

# 전처리 및 시퀀스 구성
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback_days)

# 모델
model = Sequential([
    LSTM(50, input_shape=(lookback_days, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

# 예측
x_recent = X[-compare_days:]
past_pred = scaler.inverse_transform(model.predict(x_recent))
actual_recent = scaler.inverse_transform(y[-compare_days:])
dates_past = data.index[-compare_days:]

rmse = np.sqrt(mean_squared_error(actual_recent, past_pred))
mae = mean_absolute_error(actual_recent, past_pred)

st.subheader("📊 예측 정확도 (과거 예측 비교 기준)")
st.markdown(f"- **RMSE**: `{rmse:,.2f}`  |  **MAE**: `{mae:,.2f}`")

# 미래 예측
future_input = scaled_data[-lookback_days:]
future_pred_scaled = []
current_input = future_input.copy()
for _ in range(predict_days):
    pred = model.predict(current_input.reshape(1, lookback_days, 1), verbose=0)
    pred_val = pred[0][0]
    current_input = np.append(current_input[1:], [[pred_val]], axis=0)
    future_pred_scaled.append([pred_val])

future_pred = scaler.inverse_transform(future_pred_scaled)
dates_future = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=predict_days)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates_past, actual_recent, label='Actual (Past)', color='blue')
ax.plot(dates_past, past_pred, label='Predicted (Past)', color='orange')
ax.plot(dates_future, future_pred, label='Forecast (Future)', color='green', linestyle='dashed')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.set_title(f'Stock Price Forecast with LSTM: {selected_name} ({code})')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.success("✔️ 예측이 완료되었습니다. 그래프와 지표를 확인하세요!")
