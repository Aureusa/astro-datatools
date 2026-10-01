"""Tests for the bulk SAS cutout client (projects/euclid_byol/cutouts.py); no network."""
import pytest
import requests

from projects.euclid_byol.cutouts import (
    EASCutoutClient,
    PermanentCutoutError,
    RateLimiter,
    TransientCutoutError,
)

FITS = b"SIMPLE  =                    T" + b" " * 50


class FakeResponse:
    def __init__(self, status=200, content=FITS, headers=None):
        self.status_code = status
        self.content = content
        self.reason = "reason"
        self.headers = headers or {}


class FakeSession:
    """Returns (or raises) the scripted items in order and records every request."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def post(self, url, data=None, timeout=None):
        self.requests.append((url, data))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(session, retries=3):
    sleeps = []
    client = EASCutoutClient(base_url="https://example.org/", max_retries=retries, backoff_base=1.0,
                             session=session, sleep=sleeps.append)
    return client, sleeps


def fetch(client):
    return client.fetch("/data/tile/VIS/mosaic.fits", "VIS", 102044822, 51.6, -27.5, 0.0035)


def test_success_sends_astroquery_style_request():
    session = FakeSession(FakeResponse())
    client, sleeps = make_client(session)
    assert fetch(client) == FITS
    url, data = session.requests[0]
    assert url == "https://example.org/sas-cutout/cutout"
    assert data["FILEPATH"] == "/data/tile/VIS/mosaic.fits"
    assert data["COLLECTION"] == "VIS" and data["OBSID"] == "102044822"
    assert data["POS"].startswith("CIRCLE,51.6") and data["POS"].endswith(",0.0035000000")
    assert sleeps == []


def test_retries_transient_errors_then_succeeds():
    session = FakeSession(FakeResponse(500), requests.ConnectionError("reset"), FakeResponse(503), FakeResponse())
    client, sleeps = make_client(session)
    assert fetch(client) == FITS
    assert len(session.requests) == 4
    # Exponential backoff with jitter in [0.5, 1.5) x (1, 2, 4).
    for delay, base in zip(sleeps, (1, 2, 4)):
        assert 0.5 * base <= delay < 1.5 * base


def test_permanent_http_error_is_not_retried():
    session = FakeSession(FakeResponse(404))
    client, _ = make_client(session)
    with pytest.raises(PermanentCutoutError, match="404"):
        fetch(client)
    assert len(session.requests) == 1


def test_empty_response_is_retried_and_reported_as_possible_auth_problem():
    session = FakeSession(*[FakeResponse(200, b"")] * 3)
    client, _ = make_client(session, retries=2)
    with pytest.raises(TransientCutoutError, match="login"):
        fetch(client)
    assert len(session.requests) == 3


def test_non_fits_response_is_retried():
    session = FakeSession(FakeResponse(200, b"<html>busy</html>"), FakeResponse())
    client, _ = make_client(session)
    assert fetch(client) == FITS


def test_login_uses_credentials_file(tmp_path):
    creds = tmp_path / "creds"
    creds.write_text("alice\nsecret\n")
    session = FakeSession(FakeResponse(200, b""))
    client, _ = make_client(session)
    client.login(credentials_file=str(creds))
    url, data = session.requests[0]
    assert url == "https://example.org/sas-cutout/login"
    assert data == {"username": "alice", "password": "secret"}
    assert client.logged_in


def test_rate_limiter_spaces_calls():
    now = [100.0]
    slept = []

    def sleep(dt):
        slept.append(dt)
        now[0] += dt

    limiter = RateLimiter(4.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(5):
        limiter.wait()
    assert slept == pytest.approx([0.25] * 4)
    limiter.pause(10.0)
    limiter.wait()
    assert slept[-1] == pytest.approx(10.0)


def test_rate_limiter_disabled():
    slept = []
    limiter = RateLimiter(None, sleep=slept.append)
    for _ in range(3):
        limiter.wait()
    assert slept == []
