"""Bulk client for the Euclid SAS cutout service.

``astro_datatools.surveys.EuclidSurvey.get_cutout`` looks up the tile and writes
a temporary file per call, which is fine interactively but slow for millions
of objects. This client sends the same request as astroquery
(``POST {base_url}/sas-cutout/cutout`` with ``FILEPATH``, ``COLLECTION``,
``OBSID`` and ``POS=CIRCLE,ra,dec,radius``) over a shared ``requests``
session, keeps the FITS bytes in memory, and adds a global rate limit and
retries with exponential backoff. It is safe to call from many threads.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter

#: HTTP statuses worth retrying; any other 4xx is treated as permanent.
TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class CutoutError(RuntimeError):
    """A cutout could not be retrieved."""


class TransientCutoutError(CutoutError):
    """Failure that may succeed on a later attempt (network, 5xx, empty file)."""


class PermanentCutoutError(CutoutError):
    """Failure that will not go away by retrying (e.g. HTTP 400/404)."""


class RateLimiter:
    """Thread-safe limiter spacing calls at least ``1 / rate`` seconds apart.

    :param rate: Maximum calls per second; ``None`` or 0 disables the limit.
    """

    def __init__(self, rate: Optional[float], clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 1.0 / rate if rate else 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        """Block until the next call is allowed."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            self._sleep(slot - now)

    def pause(self, seconds: float) -> None:
        """Hold back every caller for ``seconds`` (e.g. after HTTP 429)."""
        with self._lock:
            self._next = max(self._next, self._clock() + seconds)


class EASCutoutClient:
    """Fetch MER mosaic cutouts from the ESA Euclid archive.

    :param base_url: Archive host, e.g. ``https://eas.esac.esa.int``.
    :param timeout: Per-request timeout in seconds.
    :param max_retries: Retries per cutout after the first attempt.
    :param backoff_base: First backoff in seconds; doubles each retry (with jitter).
    :param backoff_max: Upper limit on a single backoff.
    :param max_requests_per_second: Global rate limit across threads.
    :param pool_size: HTTP connection pool size (set to the number of threads).
    :param session: Pre-built ``requests.Session`` (mainly for tests).
    :param sleep: Sleep function (mainly for tests).
    """

    CUTOUT_PATH = "/sas-cutout/cutout"
    LOGIN_PATH = "/sas-cutout/login"

    def __init__(
        self,
        base_url: str = "https://eas.esac.esa.int",
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        max_requests_per_second: Optional[float] = None,
        pool_size: int = 16,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep = sleep
        self.limiter = RateLimiter(max_requests_per_second, sleep=sleep)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        self.session = session
        self.logged_in = False

    @classmethod
    def from_config(cls, config: dict) -> "EASCutoutClient":
        c = config["cutouts"]
        client = cls(
            base_url=config["archive"]["base_url"],
            timeout=c["timeout_s"],
            max_retries=c["max_retries"],
            backoff_base=c["backoff_base_s"],
            backoff_max=c["backoff_max_s"],
            max_requests_per_second=c["max_requests_per_second"],
            pool_size=c["workers"],
        )
        if config["archive"].get("credentials_file"):
            client.login(credentials_file=config["archive"]["credentials_file"])
        return client

    def login(self, credentials_file: Optional[str] = None, user: Optional[str] = None,
              password: Optional[str] = None) -> None:
        """Log the session in to the cutout service. The password is not stored.

        :param credentials_file: File with the user name and password on two lines.
        :raises CutoutError: If the archive rejects the login.
        """
        if credentials_file is not None:
            with open(credentials_file) as f:
                user, password = f.readline().strip(), f.readline().strip()
        if not user or not password:
            raise ValueError("login() needs a credentials file or both user and password.")
        response = self.session.post(
            self.base_url + self.LOGIN_PATH,
            data={"username": user, "password": password},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise CutoutError(f"Euclid archive login failed: HTTP {response.status_code} {response.reason}")
        self.logged_in = True

    def fetch(self, file_path: str, collection: str, obs_id, ra: float, dec: float, radius_deg: float) -> bytes:
        """Download one cutout and return the FITS file content.

        :param file_path: Full server path of the mosaic (``file_path/file_name``).
        :param collection: Instrument, ``"VIS"`` or ``"NISP"``.
        :param obs_id: Tile index of the mosaic.
        :param ra: Centre RA in degrees.
        :param dec: Centre Dec in degrees.
        :param radius_deg: Cutout radius in degrees (the service returns a square).
        :raises PermanentCutoutError: On a non-retryable HTTP error.
        :raises TransientCutoutError: When all retries are used up.
        """
        params = {
            "TAPCLIENT": "ASTROQUERY",
            "FILEPATH": file_path,
            "COLLECTION": collection,
            "OBSID": str(obs_id),
            "POS": f"CIRCLE,{ra:.10f},{dec:.10f},{radius_deg:.10f}",
        }
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1))
                self._sleep(delay * (0.5 + random.random()))
            self.limiter.wait()
            try:
                response = self.session.post(self.base_url + self.CUTOUT_PATH, data=params, timeout=self.timeout)
            except requests.RequestException as err:
                last_error = f"{type(err).__name__}: {err}"
                continue
            status = response.status_code
            if status == 200:
                content = response.content
                if content.startswith(b"SIMPLE"):
                    return content
                last_error = (
                    "empty response (the archive may require login)" if not content
                    else f"response is not FITS ({content[:60]!r})"
                )
                continue
            last_error = f"HTTP {status} {response.reason}"
            if status not in TRANSIENT_STATUSES:
                raise PermanentCutoutError(last_error)
            retry_after = response.headers.get("Retry-After")
            if status in (429, 503) and retry_after and retry_after.isdigit():
                self.limiter.pause(float(retry_after))
        raise TransientCutoutError(f"{last_error} (after {self.max_retries + 1} attempts)")
