"""Per-tile HDF5 files holding preprocessed cutouts and their download status.

One file per MER tile keeps the number of files small (352 for Q1) and makes a
tile the unit of resumption: rows are created up front for every selected
object, and each row carries a status, so a restarted download only fetches
rows that are still pending or failed.

Layout of ``cutouts/tile_<t>.h5``::

    object_id       int64   (N,)
    ra, dec         float64 (N,)
    images          uint8   (N, S, S)   chunked per image
    status          int8    (N,)        see STATUS_* below
    attempts        int16   (N,)        download runs that tried this row
    noise_sigma, vmax, asinh_a, blank_fraction   float32 (N,)
    error           str     (N,)        last error message
    attrs: tile_index, preprocess_version, preprocess_fingerprint, size, complete
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np

STATUS_PENDING = 0
STATUS_OK = 1
STATUS_FAILED = 2           # failed this run, retried by the next run
STATUS_BLANK = 3            # downloaded but no usable pixels (outside coverage)
STATUS_FAILED_PERMANENT = 4  # permanent HTTP error, or max_attempts reached
STATUS_NAMES = {
    STATUS_PENDING: "pending",
    STATUS_OK: "ok",
    STATUS_FAILED: "failed",
    STATUS_BLANK: "blank",
    STATUS_FAILED_PERMANENT: "failed_permanent",
}
STAT_FIELDS = ("noise_sigma", "vmax", "asinh_a", "blank_fraction")


class FingerprintMismatch(RuntimeError):
    """A file was produced by a different preprocessing function."""


def _compression_kwargs(compression: Optional[str]) -> dict:
    if compression is None:
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    return {"compression": compression}


def check_fingerprint(attrs, version: str, fingerprint: str, what: str) -> None:
    """Raise :class:`FingerprintMismatch` unless ``attrs`` match the given preprocessing."""
    found = (attrs.get("preprocess_version"), attrs.get("preprocess_fingerprint"))
    if found != (version, fingerprint):
        raise FingerprintMismatch(
            f"{what} was made with preprocessing {found[0]} (fingerprint {str(found[1])[:12]}...), "
            f"not {version} ({fingerprint[:12]}...). Never mix preprocessing versions: regenerate "
            "it, or run the old version in a separate output root."
        )


class TileStore:
    """Read/write access to one tile file. Use :meth:`create` or :meth:`open`."""

    def __init__(self, path: os.PathLike, h5: h5py.File):
        self.path = Path(path)
        self.h5 = h5

    @classmethod
    def create(
        cls,
        path: os.PathLike,
        tile_index: int,
        object_ids: Sequence[int],
        ra: Sequence[float],
        dec: Sequence[float],
        size: int,
        preprocess_version: str,
        preprocess_fingerprint: str,
        compression: Optional[str] = None,
    ) -> "TileStore":
        """Create a tile file with one pending row per object (overwrites ``path``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(object_ids)
        h5 = h5py.File(path, "w")
        h5.create_dataset("object_id", data=np.asarray(object_ids, dtype=np.int64))
        h5.create_dataset("ra", data=np.asarray(ra, dtype=np.float64))
        h5.create_dataset("dec", data=np.asarray(dec, dtype=np.float64))
        h5.create_dataset(
            "images", shape=(n, size, size), dtype=np.uint8,
            chunks=(1, size, size) if n else None, **_compression_kwargs(compression),
        )
        h5.create_dataset("status", data=np.full(n, STATUS_PENDING, dtype=np.int8))
        h5.create_dataset("attempts", data=np.zeros(n, dtype=np.int16))
        for name in STAT_FIELDS:
            h5.create_dataset(name, data=np.full(n, np.nan, dtype=np.float32))
        h5.create_dataset("error", shape=(n,), dtype=h5py.string_dtype())
        h5.attrs.update(
            tile_index=int(tile_index),
            preprocess_version=preprocess_version,
            preprocess_fingerprint=preprocess_fingerprint,
            size=int(size),
            complete=False,
        )
        h5.flush()
        return cls(path, h5)

    @classmethod
    def open(cls, path: os.PathLike, mode: str = "r") -> "TileStore":
        return cls(path, h5py.File(path, mode))

    def close(self) -> None:
        if self.h5.id.valid:
            self.h5.close()

    def __enter__(self) -> "TileStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.h5["object_id"].shape[0])

    @property
    def tile_index(self) -> int:
        return int(self.h5.attrs["tile_index"])

    @property
    def object_ids(self) -> np.ndarray:
        return self.h5["object_id"][:]

    @property
    def status(self) -> np.ndarray:
        return self.h5["status"][:]

    @property
    def complete(self) -> bool:
        return bool(self.h5.attrs.get("complete", False))

    def check_fingerprint(self, version: str, fingerprint: str) -> None:
        check_fingerprint(self.h5.attrs, version, fingerprint, f"Tile file {self.path}")

    def rows_to_fetch(self) -> np.ndarray:
        """Rows still pending or failed (retryable)."""
        status = self.status
        return np.flatnonzero((status == STATUS_PENDING) | (status == STATUS_FAILED))

    def write(self, row: int, status: int, image: Optional[np.ndarray] = None,
              stats: Optional[Dict[str, float]] = None, error: str = "",
              count_attempt: bool = True) -> None:
        """Record the outcome for one row."""
        if image is not None:
            self.h5["images"][row] = image
        if stats:
            for name in STAT_FIELDS:
                self.h5[name][row] = stats.get(name, np.nan)
        self.h5["status"][row] = status
        if count_attempt:
            self.h5["attempts"][row] = self.h5["attempts"][row] + 1
        self.h5["error"][row] = error

    def attempts(self, row: int) -> int:
        return int(self.h5["attempts"][row])

    def update_complete(self) -> bool:
        """Set and return the ``complete`` flag (no pending or retryable rows left)."""
        done = len(self.rows_to_fetch()) == 0
        self.h5.attrs["complete"] = done
        return done

    def counts(self) -> Dict[str, int]:
        """Number of rows per status name."""
        status = self.status
        return {name: int((status == code).sum()) for code, name in STATUS_NAMES.items()}

    def flush(self) -> None:
        self.h5.flush()

    def status_table(self):
        """Per-object status, attempts, error and statistics as a DataFrame (no images)."""
        import pandas as pd

        table = pd.DataFrame({
            "object_id": self.object_ids,
            "status": self.status,
            "attempts": self.h5["attempts"][:],
            "error": self.h5["error"].asstr()[:],
        })
        for name in STAT_FIELDS:
            table[name] = self.h5[name][:]
        return table


def tile_counts(paths, tile_index: int) -> Optional[Dict[str, int]]:
    """Status counts of one tile, from its tile file or, once deleted, its status file.

    :return: Counts per status name plus ``rows`` and ``complete``; ``None`` if
        the tile has not been started.
    """
    path = paths.tile_file(tile_index)
    if path.exists():
        with TileStore.open(path) as store:
            counts = store.counts()
            counts.update(rows=len(store), complete=int(store.complete), tile_file_deleted=0)
        return counts
    status_path = paths.tile_status_file(tile_index)
    if status_path.exists():
        import pandas as pd

        status = pd.read_parquet(status_path, columns=["status"])["status"].to_numpy()
        counts = {name: int((status == code).sum()) for code, name in STATUS_NAMES.items()}
        counts.update(rows=len(status), complete=1, tile_file_deleted=1)
        return counts
    return None


class RawSubsetStore:
    """Append-only float32 store for the raw (pre-stretch) cutouts of a subset.

    Layout of ``raw_subset/tile_<t>.h5``: ``object_id`` int64 (N,) and ``raw``
    float32 (N, S, S), both resizable.
    """

    def __init__(self, path: os.PathLike, size: int, compression: Optional[str] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.path, "a")
        if "raw" not in self.h5:
            self.h5.create_dataset("object_id", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
            self.h5.create_dataset(
                "raw", shape=(0, size, size), maxshape=(None, size, size), dtype=np.float32,
                chunks=(1, size, size), **_compression_kwargs(compression),
            )
        self._ids = set(self.h5["object_id"][:].tolist())

    def add(self, object_id: int, raw: np.ndarray) -> None:
        """Append a raw cutout unless this object is already stored."""
        object_id = int(object_id)
        if object_id in self._ids:
            return
        n = self.h5["object_id"].shape[0]
        self.h5["object_id"].resize((n + 1,))
        self.h5["raw"].resize((n + 1,) + self.h5["raw"].shape[1:])
        self.h5["object_id"][n] = object_id
        self.h5["raw"][n] = raw
        self._ids.add(object_id)

    def flush(self) -> None:
        self.h5.flush()

    def close(self) -> None:
        if self.h5.id.valid:
            self.h5.close()


def in_raw_subset(object_id: int, fraction: float) -> bool:
    """Deterministic, uniform pick of a ``fraction`` of objects by id."""
    if fraction >= 1:
        return True
    if fraction <= 0:
        return False
    # splitmix64 finaliser: a stable hash that spreads consecutive ids evenly.
    z = (int(object_id) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return z / 2.0 ** 64 < fraction


def raw_subset_mask(object_ids, fraction: float) -> np.ndarray:
    """Vectorised :func:`in_raw_subset` (same result for every id)."""
    ids = np.asarray(object_ids, dtype=np.int64)
    if fraction >= 1:
        return np.ones(ids.shape, dtype=bool)
    if fraction <= 0:
        return np.zeros(ids.shape, dtype=bool)
    with np.errstate(over="ignore"):
        z = ids.view(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    # z < fraction * 2**64, compared in float64 exactly as in_raw_subset does.
    return z.astype(np.float64) / 2.0 ** 64 < fraction
