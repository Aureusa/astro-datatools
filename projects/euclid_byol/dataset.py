"""Fast readers of the sharded uint8 dataset for training.

Shards are HDF5 files with one gzip-compressed chunk per image. Decompression
inside HDF5 is serialised by h5py's global lock (~5k images/s per process), so
these readers fetch the raw compressed chunks (``read_direct_chunk``) and
decompress them with ``zlib`` in threads, which releases the GIL: ~23k
images/s per process on node861.

Which reader to use (numbers measured on node861):

* :class:`EuclidBlockStream` streams **from NFS** (``/zfsstore``). Random 40 kB
  reads over NFS manage only ~44 images/s, but contiguous blocks run near the
  link speed (~115 MB/s, ~2.9k gzip images/s). It reads blocks of consecutive
  images in a shuffled order and shuffles again in a buffer.
* :class:`EuclidShardDataset` gives random access on **local disk** (e.g. the
  shards copied to the job's ``/scratchdata`` directory, 2.6 GB/s) or page cache.
* :func:`load_into_memory` loads a subset into one uint8 array in RAM, e.g. the
  labelled galaxies for fine-tuning (380k images = 19 GB). The full dataset
  (~20M images, ~1 TB uncompressed) does not fit in memory.

All readers work as PyTorch datasets without importing torch and are safe
with ``DataLoader(num_workers > 0)``: files are opened lazily in each process.

By default (``exclude_flagged=True``) all readers skip objects whose
``det_quality_flag`` has bit 128 or 256 set (:data:`PROBLEM_DET_QUALITY_BITS`).
Validated against the Galaxy Zoo Euclid labels: this removes 60% of the
objects volunteers/Zoobot call "problem" (saturated stars, ghosts, artefacts)
and 1.3% of clean galaxies; 3.3% of the Q1 dataset (142,735 of 4,304,524).
Pass ``exclude_flagged=False`` for every downloaded image.
"""
from __future__ import annotations

import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

INDEX_COLUMNS = ["object_id", "shard", "shard_row"]
#: MER ``det_quality_flag`` bits marking problematic detections (near saturated
#: stars, ghosts, artefacts); see the module docstring for how this was chosen.
PROBLEM_DET_QUALITY_BITS = 128 | 256


def problem_mask(det_quality_flag) -> np.ndarray:
    """True where ``det_quality_flag`` has one of :data:`PROBLEM_DET_QUALITY_BITS` set."""
    flags = pd.Series(det_quality_flag).fillna(0).to_numpy().astype(np.int64)
    return (flags & PROBLEM_DET_QUALITY_BITS) > 0


def read_index(dataset_dir: os.PathLike, columns: Sequence[str] = (), exclude: Sequence[str] = (),
               only: Sequence[str] = (), exclude_flagged: bool = True) -> pd.DataFrame:
    """Load ``index.parquet`` with optional boolean-column filters.

    :param columns: Extra index columns to load.
    :param exclude: Boolean columns (e.g. ``label_morphology``); rows where any is true are dropped.
    :param only: Boolean columns; keep only rows where all are true.
    :param exclude_flagged: Drop objects with a problem ``det_quality_flag`` (see module docstring).
    :raises KeyError: If ``exclude_flagged`` is set but the index has no ``det_quality_flag``.
    """
    path = Path(dataset_dir) / "index.parquet"
    extra = []
    if exclude_flagged:
        import pyarrow.parquet as pq

        if "det_quality_flag" not in pq.read_schema(path).names:
            raise KeyError(f"{path} has no det_quality_flag column; pass exclude_flagged=False.")
        extra = ["det_quality_flag"]
    wanted = list(dict.fromkeys([*INDEX_COLUMNS, *columns, *exclude, *only, *extra]))
    index = pd.read_parquet(path, columns=wanted)
    keep = np.ones(len(index), dtype=bool)
    if exclude_flagged:
        keep &= ~problem_mask(index["det_quality_flag"])
        if "det_quality_flag" not in columns:
            index = index.drop(columns="det_quality_flag")
    for col in exclude:
        keep &= ~index[col].fillna(False).to_numpy(bool)
    for col in only:
        keep &= index[col].fillna(False).to_numpy(bool)
    return index[keep].reset_index(drop=True)


class _ChunkReader:
    """Reads images from an HDF5 ``images`` dataset, decompressing in threads when possible."""

    def __init__(self, dataset: h5py.Dataset):
        self.ds = dataset
        self.shape = dataset.shape[1:]
        # Fast path: one image per chunk, compressed with gzip only (or not at all).
        self.direct = (
            dataset.chunks == (1, *self.shape)
            and dataset.compression in (None, "gzip")
            and not dataset.shuffle
            and not dataset.fletcher32
            and dataset.scaleoffset is None
        )
        self.gzip = dataset.compression == "gzip"

    def read(self, rows: np.ndarray, pool: Optional[ThreadPoolExecutor], out: Optional[np.ndarray] = None) -> np.ndarray:
        """Images at ``rows`` (any order) as a uint8 array."""
        rows = np.asarray(rows, dtype=np.int64)
        if out is None:
            out = np.empty((len(rows), *self.shape), dtype=np.uint8)
        if len(rows) == 0:
            return out
        if not self.direct:
            order = np.argsort(rows)
            lo, hi = int(rows[order[0]]), int(rows[order[-1]]) + 1
            block = self.ds[lo:hi]
            out[order] = block[rows[order] - lo]
            return out
        chunks = [self.ds.id.read_direct_chunk((int(r), 0, 0))[1] for r in rows]

        def decode(i: int) -> None:
            data = zlib.decompress(chunks[i]) if self.gzip else chunks[i]
            out[i] = np.frombuffer(data, dtype=np.uint8).reshape(self.shape)

        if pool is None or len(rows) < 64:
            for i in range(len(rows)):
                decode(i)
        else:
            list(pool.map(decode, range(len(rows)), chunksize=64))
        return out


class _ShardFiles:
    """Per-process cache of open shard files (never shared across fork)."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self._files: Dict[str, Tuple[h5py.File, _ChunkReader]] = {}
        self._pid = None

    def reader(self, name: str) -> _ChunkReader:
        if self._pid != os.getpid():
            self._files, self._pid = {}, os.getpid()
        if name not in self._files:
            h5 = h5py.File(self.dataset_dir / name, "r")
            self._files[name] = (h5, _ChunkReader(h5["images"]))
        return self._files[name][1]

    def close(self) -> None:
        for h5, _ in self._files.values():
            h5.close()
        self._files = {}


class EuclidShardDataset:
    """Map-style random access (``ds[i] -> (uint8 image, object_id)``).

    Use on local disk or page cache; over NFS every item is a random read (~44/s).

    :param dataset_dir: ``<root>/dataset`` (or a local copy of it).
    :param columns: Extra index columns to load (available as ``self.index``).
    :param exclude: Boolean index columns; rows where any is true are dropped,
        e.g. ``["label_morphology"]`` to keep labelled galaxies out of pretraining.
    :param only: Boolean index columns; keep only rows where all are true.
    :param exclude_flagged: Skip objects with a problem ``det_quality_flag`` (default True).
    :param transform: Applied to each uint8 image.
    """

    def __init__(self, dataset_dir: os.PathLike, columns: Sequence[str] = (), exclude: Sequence[str] = (),
                 only: Sequence[str] = (), transform: Optional[Callable] = None, exclude_flagged: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.index = read_index(self.dataset_dir, columns, exclude, only, exclude_flagged=exclude_flagged)
        self._shard = self.index["shard"].to_numpy()
        self._row = self.index["shard_row"].to_numpy()
        self._ids = self.index["object_id"].to_numpy()
        self.transform = transform
        self._files = _ShardFiles(self.dataset_dir)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        image = self._files.reader(self._shard[i]).read(np.array([self._row[i]]), None)[0]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self._ids[i])


class EuclidBlockStream:
    """Iterable dataset streaming shuffled blocks of consecutive images.

    Each epoch, the blocks (``block_size`` consecutive shard rows) are shuffled
    and split across DataLoader workers (and across ``world_size`` ranks for
    distributed training). Each worker reads its blocks sequentially and yields
    images in random order from a buffer of ``shuffle_blocks`` blocks.
    Every selected image is yielded exactly once per epoch. Call
    :meth:`set_epoch` before each epoch to change the order.

    :param dataset_dir: ``<root>/dataset``.
    :param block_size: Images per read (256 x 40 kB = 10 MB, good for NFS).
    :param shuffle_blocks: Blocks mixed in the shuffle buffer.
    :param seed: Base seed; the order depends on ``seed`` and the epoch.
    :param rank: This process's rank for distributed training.
    :param world_size: Number of distributed processes.
    :param decode_threads: zlib threads per DataLoader worker.
    :param exclude: See :class:`EuclidShardDataset`.
    :param only: See :class:`EuclidShardDataset`.
    :param exclude_flagged: See :class:`EuclidShardDataset`.
    :param transform: Applied to each uint8 image.
    """

    def __init__(self, dataset_dir: os.PathLike, block_size: int = 256, shuffle_blocks: int = 16, seed: int = 0,
                 rank: int = 0, world_size: int = 1, decode_threads: int = 4, exclude: Sequence[str] = (),
                 only: Sequence[str] = (), transform: Optional[Callable] = None, exclude_flagged: bool = True):
        self.dataset_dir = Path(dataset_dir)
        index = read_index(self.dataset_dir, exclude=exclude, only=only, exclude_flagged=exclude_flagged)
        self.block_size = int(block_size)
        self.shuffle_blocks = int(shuffle_blocks)
        self.seed = int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
        self.decode_threads = int(decode_threads)
        self.transform = transform
        self.epoch = 0
        self._n = len(index)
        # Blocks of consecutive shard rows; filters leave gaps, which are skipped.
        self.blocks: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for shard, group in index.groupby("shard", sort=True):
            group = group.sort_values("shard_row")
            rows = group["shard_row"].to_numpy(np.int64)
            ids = group["object_id"].to_numpy(np.int64)
            block_of = rows // self.block_size
            for b in np.unique(block_of):
                sel = block_of == b
                self.blocks.append((shard, rows[sel], ids[sel]))
        self._files = _ShardFiles(self.dataset_dir)

    def __len__(self) -> int:
        """Images per epoch over all ranks and workers."""
        return self._n

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _my_blocks(self) -> List[int]:
        order = np.random.default_rng([self.seed, self.epoch]).permutation(len(self.blocks))
        worker_id, num_workers = 0, 1
        try:
            from torch.utils.data import get_worker_info

            info = get_worker_info()
            if info is not None:
                worker_id, num_workers = info.id, info.num_workers
        except ImportError:
            pass
        part = self.rank * num_workers + worker_id
        return list(order[part::self.world_size * num_workers])

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        blocks = self._my_blocks()
        rng = np.random.default_rng([self.seed, self.epoch, self.rank, len(blocks)])
        with ThreadPoolExecutor(self.decode_threads) as pool:
            for start in range(0, len(blocks), self.shuffle_blocks):
                images, ids = [], []
                for b in blocks[start:start + self.shuffle_blocks]:
                    shard, rows, block_ids = self.blocks[b]
                    images.append(self._files.reader(shard).read(rows, pool))
                    ids.append(block_ids)
                images, ids = np.concatenate(images), np.concatenate(ids)
                for i in rng.permutation(len(ids)):
                    image = images[i]
                    if self.transform is not None:
                        image = self.transform(image)
                    yield image, int(ids[i])


def load_into_memory(dataset_dir: os.PathLike, exclude: Sequence[str] = (), only: Sequence[str] = (),
                     max_images: Optional[int] = None, threads: int = 8,
                     memory_fraction: float = 0.8, exclude_flagged: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load (a subset of) the dataset into one uint8 array ``(N, 224, 224)``.

    Reads each shard's rows in order and decompresses in ``threads`` threads.

    :param max_images: Load only the first N selected images (in index order).
    :param memory_fraction: Refuse to allocate more than this fraction of available RAM.
    :param exclude_flagged: Skip objects with a problem ``det_quality_flag`` (default True).
    :return: ``(images, index)``, where row ``i`` of ``index`` describes ``images[i]``.
    :raises MemoryError: If the images would not fit.
    """
    dataset_dir = Path(dataset_dir)
    index = read_index(dataset_dir, exclude=exclude, only=only, exclude_flagged=exclude_flagged)
    if max_images is not None:
        index = index.iloc[:int(max_images)].reset_index(drop=True)
    files = _ShardFiles(dataset_dir)
    try:
        if len(index) == 0:
            return np.zeros((0, 224, 224), dtype=np.uint8), index
        shape = files.reader(index["shard"].iloc[0]).shape
        need = len(index) * int(np.prod(shape))
        available = _available_memory()
        if available is not None and need > memory_fraction * available:
            raise MemoryError(
                f"{len(index)} images need {need / 1e9:.1f} GB but only {available / 1e9:.1f} GB of RAM is "
                "available; select a subset (only/exclude/max_images) or stream with EuclidBlockStream."
            )
        images = np.empty((len(index), *shape), dtype=np.uint8)
        with ThreadPoolExecutor(threads) as pool:
            for shard, group in index.groupby("shard", sort=True):
                positions = group.index.to_numpy()
                rows = group["shard_row"].to_numpy(np.int64)
                order = np.argsort(rows)
                images[positions[order]] = files.reader(shard).read(rows[order], pool)
        return images, index
    finally:
        files.close()


def _available_memory() -> Optional[int]:
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
