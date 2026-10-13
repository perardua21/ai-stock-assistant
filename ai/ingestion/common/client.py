"""
외부 API 공통 호출 유틸 — 인증 토큰 캐싱 + 재시도/레이트리밋 백오프.
각 소스(kis_price, opendart_filing, naver_news)가 공통으로 가져다 씀.
"""
import time

import backoff
import requests


class ApiError(Exception):
    pass


@backoff.on_exception(backoff.expo, (requests.RequestException, ApiError), max_tries=5, jitter=backoff.full_jitter)
def request_json(method: str, url: str, **kwargs) -> dict:
    """requests 호출 + 재시도. 429/5xx는 backoff가 알아서 재시도하도록 ApiError로 변환."""
    resp = requests.request(method, url, timeout=kwargs.pop("timeout", 10), **kwargs)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise ApiError(f"{resp.status_code} from {url}")
    resp.raise_for_status()
    return resp.json()


class TokenCache:
    """OAuth 토큰처럼 만료 시간이 있는 인증 정보를 메모리에 캐싱 (한투 API용)."""

    def __init__(self):
        self._token = None
        self._expires_at = 0

    def get(self, fetch_fn, ttl_margin: int = 60):
        now = time.time()
        if self._token is None or now >= self._expires_at - ttl_margin:
            self._token, ttl = fetch_fn()
            self._expires_at = now + ttl
        return self._token
