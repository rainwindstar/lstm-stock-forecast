# LSTM 주가 예측 대시보드

Streamlit으로 실행하는 국내 주식 LSTM 예측 대시보드입니다. 가격 기술지표, 선택형 거시경제 지표, 선택형 DART 재무 지표를 함께 사용해 검증 성능과 미래 영업일 예측값을 확인합니다.

## 주요 업그레이드
- 학습/검증 데이터를 시간순으로 분리하고, 스케일러도 학습 구간에만 맞춰 데이터 누수를 줄였습니다.
- DART API 키를 코드에 저장하지 않고 환경변수 또는 Streamlit secrets에서 읽습니다.
- 로컬 `CORPCODE.xml`, KRX 목록, 기본 종목 목록 순서로 종목 검색을 대체합니다.
- RMSE, MAE, MAPE, 방향 정확도와 검증 오차 기반 예측 구간을 표시합니다.
- CSV 다운로드와 ngrok 공유 스크립트를 제공합니다.

## 설치
```bash
python -m pip install -r requirements.txt
```

## 실행
```bash
python -m streamlit run app.py
```

또는 Windows에서:
```bash
python start_share.py
```

`start_share.py`는 기본적으로 로컬 서버를 열고, `NGROK_AUTHTOKEN` 또는 `NGROK_TOKEN` 환경변수가 있으면 외부 공유 주소까지 생성합니다.

## 선택 설정
DART 재무 피처를 사용하려면 환경변수를 설정합니다.

```bash
set DART_API_KEY=발급받은_OPEN_DART_API_KEY
```

Streamlit secrets를 쓰는 경우 `.streamlit/secrets.toml`에 아래처럼 넣습니다.

```toml
DART_API_KEY = "발급받은_OPEN_DART_API_KEY"
```

예측 결과는 과거 데이터 기반 실험용이며 투자 판단의 단독 근거로 사용하면 안 됩니다.
