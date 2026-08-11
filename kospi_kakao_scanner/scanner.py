import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import FinanceDataReader as fdr
import exchange_calendars as xcals
import pandas as pd
import requests

KST = ZoneInfo("Asia/Seoul")
TOKEN_URL = "https://kauth.kakao.com/oauth/token"
MEMO_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"


def is_krx_session():
    return xcals.get_calendar("XKRX").is_session(pd.Timestamp(datetime.now(KST).date()))


def refresh_access_token():
    data = {
        "grant_type": "refresh_token",
        "client_id": os.environ["KAKAO_REST_API_KEY"],
        "refresh_token": os.environ["KAKAO_REFRESH_TOKEN"],
    }
    secret = os.getenv("KAKAO_CLIENT_SECRET", "").strip()
    if secret:
        data["client_secret"] = secret
    r = requests.post(TOKEN_URL, data=data, timeout=20)
    r.raise_for_status()
    result = r.json()
    if not result.get("access_token"):
        raise RuntimeError(f"Kakao token refresh failed: {result}")
    return result["access_token"]


def find_aligned():
    listing = fdr.StockListing("KOSPI")
    required = {"Code", "Name", "Marcap"}
    if not required.issubset(listing.columns):
        raise RuntimeError(f"Unexpected listing columns: {list(listing.columns)}")
    listing = listing.sort_values("Marcap", ascending=False).head(500)
    end = datetime.now(KST).date()
    start = end - timedelta(days=140)
    found = []
    failures = 0
    for rank, (_, row) in enumerate(listing.iterrows(), 1):
        code = str(row["Code"]).zfill(6)
        try:
            price = fdr.DataReader(code, start.isoformat(), end.isoformat())
            close = pd.to_numeric(price["Close"], errors="coerce").dropna()
            if len(close) < 60:
                continue
            ma = {n: close.rolling(n).mean().iloc[-1] for n in (5, 10, 20, 60)}
            if ma[5] > ma[10] > ma[20] > ma[60]:
                found.append((rank, str(row["Name"]), code, int(close.iloc[-1])))
        except Exception as exc:
            failures += 1
            print(f"WARN {row['Name']}({code}): {exc}", flush=True)
        time.sleep(0.03)
    print(f"aligned={len(found)}, failures={failures}", flush=True)
    return found


def make_messages(stocks):
    header = f"KOSPI 정배열 {datetime.now(KST):%Y-%m-%d} | {len(stocks)}개"
    lines = [f"{rank}. {name}({code}) {price:,}원" for rank, name, code, price in stocks]
    if not lines:
        return [header + "\n조건 충족 종목 없음"]
    chunks, current = [], header
    for line in lines:
        candidate = current + "\n" + line
        if len(candidate) <= 180:
            current = candidate
        else:
            chunks.append(current)
            current = header + "\n" + line
    chunks.append(current)
    total = len(chunks)
    return [(c + f"\n[{i}/{total}]")[:200] for i, c in enumerate(chunks, 1)]


def send(access_token, text):
    link = os.getenv("KAKAO_LINK_URL", "https://finance.naver.com")
    template = {
        "object_type": "text",
        "text": text,
        "link": {"web_url": link, "mobile_web_url": link},
        "button_title": "증시 확인",
    }
    r = requests.post(
        MEMO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        data={"template_object": json.dumps(template, ensure_ascii=False)},
        timeout=20,
    )
    r.raise_for_status()
    result = r.json()
    if result.get("result_code") != 0:
        raise RuntimeError(f"Kakao send failed: {result}")


def main():
    force = os.getenv("FORCE_RUN", "false").lower() in {"1", "true", "yes"}
    if not force and not is_krx_session():
        print("KRX 휴장일: 정상 종료", flush=True)
        return
    token = refresh_access_token()
    stocks = find_aligned()
    messages = make_messages(stocks)
    for i, message in enumerate(messages, 1):
        send(token, message)
        print(f"Kakao message {i}/{len(messages)} sent", flush=True)
        time.sleep(0.5)
    print("모든 메시지 발송 성공", flush=True)


if __name__ == "__main__":
    main()
