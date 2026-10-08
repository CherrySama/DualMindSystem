from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from jev4mujoco.policies.typesafe_client import TypeSafeClient, TypeSafeClientError


class TypeSafeClientTest(unittest.TestCase):
    def test_default_client_uses_system_network_environment(self) -> None:
        with patch("jev4mujoco.policies.typesafe_client.httpx.Client") as client_factory:
            client = TypeSafeClient("test-key")

            client_factory.assert_called_once_with(
                timeout=25.0,
                follow_redirects=False,
                trust_env=True,
            )
            client.close()

    def test_success_uses_bearer_header_and_json_without_redirects(self) -> None:
        observed: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed["url"] = str(request.url)
            observed["authorization"] = request.headers.get("authorization")
            observed["body"] = request.content
            return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {}})

        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(transport=transport, follow_redirects=False)
        client = TypeSafeClient("test-key", client=http_client)

        result = client.query({"model": "jev-1.13.0", "state": {}, "questions": {}})

        self.assertEqual(result["model"], "jev-1.13.0")
        self.assertEqual(observed["url"], "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(observed["authorization"], "Bearer test-key")
        self.assertIn(b'"jev-1.13.0"', observed["body"])
        self.assertEqual(client.last_call["attempts"], 1)
        self.assertEqual(client.last_call["http_status"], 200)
        self.assertNotIn("test-key", repr(client.last_call))
        http_client.close()

    def test_retryable_status_honors_bounded_retry_after(self) -> None:
        attempts = 0
        delays: list[float] = []

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(503, headers={"retry-after": "99"})
            return httpx.Response(200, json={"ok": True})

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = TypeSafeClient(
            "test-key", retries=1, client=http_client, sleep=delays.append
        )

        self.assertEqual(client.query({"request": 1}), {"ok": True})
        self.assertEqual(client.last_call["attempts"], 2)
        self.assertEqual(delays, [10.0])
        http_client.close()

    def test_transport_error_retries_then_fails_without_response_body(self) -> None:
        delays: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("test failure", request=request)

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = TypeSafeClient(
            "test-key", retries=1, client=http_client, sleep=delays.append
        )

        with self.assertRaisesRegex(TypeSafeClientError, "transport error"):
            client.query({"request": 1})

        self.assertEqual(client.last_call["attempts"], 2)
        self.assertEqual(client.last_call["error"], "transport_error")
        self.assertEqual(delays, [1])
        http_client.close()

    def test_non_retryable_status_and_invalid_json_are_rejected(self) -> None:
        unauthorized = httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(401, text="secret response body")
            )
        )
        with self.assertRaisesRegex(TypeSafeClientError, "HTTP 401"):
            TypeSafeClient("test-key", client=unauthorized).query({})
        unauthorized.close()

        invalid_json = httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not-json"))
        )
        client = TypeSafeClient("test-key", client=invalid_json)
        with self.assertRaisesRegex(TypeSafeClientError, "invalid JSON"):
            client.query({})
        self.assertNotIn("not-json", repr(client.last_call))
        invalid_json.close()

    def test_insecure_url_or_invalid_retry_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            TypeSafeClient("test-key", api_url="http://example.test")
        with self.assertRaisesRegex(ValueError, "\[0, 2\]"):
            TypeSafeClient("test-key", retries=3)


if __name__ == "__main__":
    unittest.main()
