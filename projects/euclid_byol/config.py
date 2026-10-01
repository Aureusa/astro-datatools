"""Configuration loading and on-disk layout for the Euclid BYOL pipeline.

A config is a YAML file (see ``configs/pipeline.yaml``). A config may start
with ``extends: other.yaml`` (relative to its own directory); it is then
deep-merged on top of that file, which keeps variants such as
``configs/sample.yaml`` down to the keys they change.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from astro_datatools.logger import setup_logging

from . import preprocess_v1

CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "pipeline.yaml"

#: Preprocessing modules by version name; the config selects one by name.
PREPROCESSORS = {"v1": preprocess_v1}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[os.PathLike] = None, root: Optional[str] = None) -> Dict[str, Any]:
    """Load a pipeline config, resolving ``extends`` and validating it.

    :param path: YAML file (default ``configs/pipeline.yaml``).
    :param root: Override for ``paths.root``.
    :return: Config dict.
    """
    path = Path(path or DEFAULT_CONFIG)
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    parent = config.pop("extends", None)
    if parent is not None:
        config = _deep_merge(load_config(path.parent / parent), config)
    if root is not None:
        config.setdefault("paths", {})["root"] = root
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Check that the config agrees with the frozen preprocessing module."""
    version = config["preprocess"]["version"]
    if version not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocess version '{version}'. Available: {sorted(PREPROCESSORS)}")
    module = get_preprocessor(config)
    size = config["cutouts"]["size_pix"]
    if size != module.OUTPUT_SIZE:
        raise ValueError(
            f"cutouts.size_pix={size} but preprocess {version} produces {module.OUTPUT_SIZE} px images. "
            "The output size is part of the frozen preprocessing; change it only in a new version."
        )


def get_preprocessor(config: Dict[str, Any]):
    """Return the preprocessing module selected by ``preprocess.version``."""
    return PREPROCESSORS[config["preprocess"]["version"]]


class Paths:
    """Directory layout under ``paths.root``.

    ::

        root/
          catalogue/tiles.parquet            one row per MER tile
          catalogue/parts/tile_<t>.parquet   catalogue rows per tile (+ ``selected``)
          catalogue/master_index.parquet     all tiles concatenated
          cutouts/tile_<t>.h5                uint8 images + per-object status
          cutouts/tile_<t>.status.parquet    status only, once tile_<t>.h5 is sharded and deleted
          raw_subset/tile_<t>.h5             float32 raw cutouts of a small subset
          dataset/shard_<n>.h5               final uint8 shards
          dataset/index.parquet              object -> shard/row + catalogue columns
          labels/                            label catalogues and matched subsets
          logs/<stage>.log, stats/<stage>.json
    """

    def __init__(self, root: os.PathLike):
        self.root = Path(root)

    @property
    def catalogue_dir(self) -> Path:
        return self.root / "catalogue"

    @property
    def tiles(self) -> Path:
        return self.catalogue_dir / "tiles.parquet"

    @property
    def catalogue_parts(self) -> Path:
        return self.catalogue_dir / "parts"

    def catalogue_part(self, tile_index: int) -> Path:
        return self.catalogue_parts / f"tile_{int(tile_index)}.parquet"

    @property
    def master_index(self) -> Path:
        return self.catalogue_dir / "master_index.parquet"

    @property
    def cutouts_dir(self) -> Path:
        return self.root / "cutouts"

    def tile_file(self, tile_index: int) -> Path:
        return self.cutouts_dir / f"tile_{int(tile_index)}.h5"

    def tile_status_file(self, tile_index: int) -> Path:
        """Per-object status kept after a sharded tile file is deleted (``delete_tile_files``)."""
        return self.cutouts_dir / f"tile_{int(tile_index)}.status.parquet"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw_subset"

    def raw_file(self, tile_index: int) -> Path:
        return self.raw_dir / f"tile_{int(tile_index)}.h5"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    def shard_file(self, shard_number: int) -> Path:
        return self.dataset_dir / f"shard_{int(shard_number):05d}.h5"

    @property
    def dataset_index(self) -> Path:
        return self.dataset_dir / "index.parquet"

    @property
    def index_parts(self) -> Path:
        return self.dataset_dir / "index_parts"

    @property
    def shard_state(self) -> Path:
        return self.dataset_dir / "_state.json"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def stats_dir(self) -> Path:
        return self.root / "stats"


def get_paths(config: Dict[str, Any]) -> Paths:
    return Paths(config["paths"]["root"])


def stage_logger(config: Dict[str, Any], stage: str):
    """Logger writing to the console and ``logs/<stage>.log``."""
    return setup_logging(
        name=f"euclid_byol.{stage}", log_file=str(get_paths(config).logs_dir / f"{stage}.log")
    )


def write_stats(config: Dict[str, Any], stage: str, stats: Dict[str, Any]) -> Path:
    """Write stage counts to ``stats/<stage>.json`` (with a timestamp and the config path)."""
    path = get_paths(config).stats_dir / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": config.get("_config_path"),
        **stats,
    }
    atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default))
    return path


def atomic_write_text(path: os.PathLike, text: str) -> None:
    """Write a text file via a temporary file and rename, so readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_parquet(df, path: os.PathLike) -> None:
    """Write a DataFrame to parquet via a temporary file and rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)
