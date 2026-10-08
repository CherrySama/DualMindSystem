from __future__ import annotations

import math
import time
from typing import Any, Callable

import httpx


DEFAULT_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504, 529})


class TypeSafeClientError(RuntimeError):
    """The remote request failed, so no robot action may execute."""


class TypeSafeClient:
    def __init__(
        self,
        api_key: str,
        api_url: str = DEFAULT_TYPESAFE_URL,
        timeout_s: float = 25.0,
        retries: int = 2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY is required")
        if not api_url.startswith("https://"):
            raise ValueError("TypeSafe API URL must use HTTPS")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if type(retries) is not int or retries < 0 or retries > 2:
            raise ValueError("retries must be an integer in [0, 2]")

        self._api_key = api_key
        self._api_url = api_url
        self._retries = retries
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout_s,
            follow_redirects=False,
            trust_env=True,
        )
        self.last_call: dict[str, Any] = {}

    def query(self, body: dict[str, Any]) -> object:
        started = time.perf_counter()
        self.last_call = {"attempts": 0, "http_status": None, "error": None}
        try:
            return self._query(body)
        finally:
            self.last_call["latency_ms"] = (time.perf_counter() - started) * 1000.0

    def _query(self, body: dict[str, Any]) -> object:
        response: httpx.Response | None = None
        for attempt in range(self._retries + 1):
            self.last_call["attempts"] += 1
            try:
                response = self._client.post(
                    self._api_url,
                    json=body,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            except httpx.TransportError:
                if attempt == self._retries:
                    self.last_call["error"] = "transport_error"
                    raise TypeSafeClientError(
                        "TypeSafe transport error; no action executed"
                    ) from None
                self._sleep(min(2**attempt, 5))
                continue

            self.last_call["http_status"] = response.status_code
            if (
                response.status_code in RETRYABLE_HTTP_STATUS
                and attempt < self._retries
            ):
                self._sleep(self._retry_delay_s(response, attempt))
                continue
            if response.status_code != 200:
                self.last_call["error"] = f"http_{response.status_code}"
                raise TypeSafeClientError(
                    f"TypeSafe returned HTTP {response.status_code}; no action executed"
                )
            break

        if response is None:
            raise TypeSafeClientError("TypeSafe produced no response; no action executed")
        try:
            decoded = response.json()
        except (ValueError, TypeError):
            self.last_call["error"] = "invalid_json"
            raise TypeSafeClientError(
                "TypeSafe returned invalid JSON; no action executed"
            ) from None
        if not isinstance(decoded, dict):
            self.last_call["error"] = "non_object_json"
            raise TypeSafeClientError(
                "TypeSafe returned non-object JSON; no action executed"
            )
        return decoded

    @staticmethod
    def _retry_delay_s(response: httpx.Response, attempt: int) -> float:
        try:
            delay = float(response.headers.get("retry-after", 2**attempt))
        except ValueError:
            delay = float(2**attempt)
        if not math.isfinite(delay):
            return 1.0
        return min(max(delay, 0.1), 10.0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TypeSafeClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
