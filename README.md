# LSTM 주가 예측 대시보드

Streamlit으로 실행하는 국내 주식 LSTM 예측 대시보드입니다. 가격 기술지표, 선택형 거시경제 지표, 선택형 DART 재무 지표를 함께 사용해 검증 성능과 미래 영업일 예측값을 확인합니다.

## 주요 업그레이드
- 학습/검증 데이터를 시간순으로 분리하고, 스케일러도 학습 구간에만 맞춰 데이터 누수를 줄였습니다.
- DART API 키를 코드에 저장하지 않고 환경변수 또는 Streamlit secrets에서 읽습니다.
- 로컬 `.env`의 ECOS/DART/KRX/공공데이터 키를 자동 인식합니다.
- 거시경제 지표는 ECOS 환율/기준금리/국채/CPI, 네이버 KOSPI/KOSDAQ, FDR VIX를 조합해 표시합니다.
- 모델 입력 피처와 화면 표시값을 분리해 환율/KOSPI/VIX가 사람이 읽기 쉬운 원값·등락률·Z-score로 보입니다.
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

카톡 공유용 공개 주소까지 만들려면 Windows에서 먼저 Cloudflare CLI를 설치하고 새 터미널을 여세요:

```powershell
winget install --id Cloudflare.cloudflared -e
```

그런 다음 실행합니다:

```bash
publish_kakao.bat
```

실행 후 콘솔에 표시되는 `https://...trycloudflare.com` 주소를 카톡에 공유합니다. 같은 주소는 `public_url.txt`에도 저장되고 클립보드에도 복사됩니다. 공유 중에는 콘솔 창을 닫지 마세요. 종료는 `Ctrl+C`이며 서버와 터널을 정리하고 만료된 `public_url.txt`를 삭제합니다. 기본 포트는 8502이며 `STREAMLIT_PORT`로 변경할 수 있습니다. 이미 사용 중인 포트는 오류로 알리고 기존 프로세스를 종료하지 않습니다.

ngrok을 사용하려면 ngrok 계정의 인증 토큰을 환경변수로 설정합니다 (Windows cmd):
```bat
set NGROK_AUTHTOKEN=발급받은_ngrok_토큰
```

실행:
```bash
python start_share.py
```

`start_share.py`는 기본적으로 로컬 서버를 열고, `NGROK_AUTHTOKEN` 또는 `NGROK_TOKEN` 환경변수가 있으면 HTTPS 외부 공유 주소를 생성하고 동일하게 `public_url.txt` 저장·Windows 클립보드 복사를 수행합니다. `pyngrok`는 `requirements.txt`에 포함됩니다. Cloudflare는 계정 토큰 없이 임시 URL을 생성하며 실행할 때마다 주소가 바뀔 수 있습니다.

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


## Google Sheets 자동 저장

사이드바에서 `예측 결과 자동 저장`을 켜면 예측 결과가 `LSTM_종목명_종목코드_주가예측` Google Sheets의 `예측기록` 탭에 저장됩니다. 이후 같은 종목을 다시 검색하면 저장된 예측일의 실제 종가가 들어온 경우 `실제종가`, `오차`, `오차율(%)`이 자동 갱신됩니다.

필요한 설정과 운영 계획은 `GOOGLE_SHEETS_ACTUAL_PRICE_PLAN.md`를 참고하세요.
