# Google Sheets 실제 종가 자동 입력 및 오차 관리 계획서

## 목표
Streamlit에서 특정 종목을 검색하고 LSTM 예측을 수행하면 `LSTM_종목명_종목코드_주가예측` Google Sheets를 생성 또는 갱신한다. 같은 검색일자, 종목코드, 예측일 조합은 중복 추가하지 않고 기존 행을 덮어쓴다.

## 현재 반영된 코드 흐름
1. 앱 실행 후 예측 테이블(`pred_df`)을 만든다.
2. Google Sheets 저장이 켜져 있거나 `현재 결과 저장/갱신` 버튼을 누르면 서비스 계정으로 Google Sheets에 접속한다.
3. 파일명이 없으면 새로 만들고, 있으면 기존 파일을 연다.
4. `예측기록` 탭에 예측일, 예측종가, 모델 설정, 검증 지표, 피처 목록을 저장한다.
5. 저장 전 기존 행을 확인하여 종목코드와 예측일에 해당하는 실제 종가가 현재 주가 데이터에 있으면 `실제종가`, `오차`, `오차율(%)`을 갱신한다.

## 필요한 배포 설정
Streamlit Cloud Secrets에 아래 중 하나를 설정한다.

```toml
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@...iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."

GOOGLE_SHEETS_AUTO_SAVE = "true"
GOOGLE_SHEET_SHARE_EMAIL = "사용자@gmail.com"
GOOGLE_SHEET_FOLDER_ID = "..."
```

또는 `GOOGLE_SERVICE_ACCOUNT_JSON`에 서비스 계정 JSON 문자열 전체를 넣어도 된다.

## 실제 종가 자동 입력 방식
- 앱이 다시 실행될 때 FinanceDataReader로 최신 종가 데이터를 가져온다.
- 기존 Google Sheets 행 중 `종목코드`가 현재 종목과 같고 `예측일`이 최신 데이터에 포함되면 실제 종가를 입력한다.
- `오차 = 실제종가 - 예측종가`, `오차율(%) = 오차 / 예측종가 * 100`으로 계산한다.
- 주말/휴장일은 예측일을 영업일 기준으로 생성하므로 실제 종가는 다음 데이터 갱신 시점에 채워진다.

## 운영 권장사항
- 서비스 계정이 만든 시트는 기본적으로 서비스 계정 소유이므로 `GOOGLE_SHEET_SHARE_EMAIL`을 반드시 지정한다.
- 자동 저장을 기본으로 켜려면 `GOOGLE_SHEETS_AUTO_SAVE=true`를 설정한다.
- 모델 설정을 바꿔 같은 날짜에 다시 검색하면 같은 `검색일자+종목코드+예측일` 행이 갱신된다.
