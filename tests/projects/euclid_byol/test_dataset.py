"""Tests for the training readers (projects/euclid_byol/dataset.py)."""
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import pytest

from projects.euclid_byol.dataset import EuclidBlockStream, EuclidShardDataset, _ChunkReader, load_into_memory

SIZE = 224


def make_dataset(tmp_path, sizes=(50, 30, 45), compression=("gzip", None, "gzip"), chunk_one=(True, True, False)):
    """Shards with known images (pixel [0, 0..3] encodes the object id); some labelled."""
    rng = np.random.default_rng(0)
    rows, next_id = [], 1000
    for s, (n, comp, one) in enumerate(zip(sizes, compression, chunk_one)):
        ids = np.arange(next_id, next_id + n)
        next_id += n
        images = rng.integers(0, 256, (n, SIZE, SIZE), dtype=np.uint8)
        images[:, 0, :4] = np.frombuffer(ids.astype(">u4").tobytes(), np.uint8).reshape(n, 4)
        name = f"shard_{s:05d}.h5"
        with h5py.File(tmp_path / name, "w") as h5:
            kw = {"compression": comp} if comp else {}
            h5.create_dataset("images", data=images, chunks=(1, SIZE, SIZE) if one else (8, SIZE, SIZE), **kw)
            h5.create_dataset("object_id", data=ids)
        rows.append(pd.DataFrame({"object_id": ids, "shard": name, "shard_row": np.arange(n),
                                  "label_morphology": ids % 7 == 0,
                                  "det_quality_flag": np.zeros(n, dtype=np.int16)}))
    pd.concat(rows).to_parquet(tmp_path / "index.parquet")
    return tmp_path


def decode_id(image):
    return int(np.frombuffer(image[0, :4].tobytes(), ">u4")[0])


def test_chunk_reader_fast_path_matches_h5py(tmp_path):
    make_dataset(tmp_path)
    for shard, direct in (("shard_00000.h5", True), ("shard_00001.h5", True), ("shard_00002.h5", False)):
        with h5py.File(tmp_path / shard, "r") as h5:
            reader = _ChunkReader(h5["images"])
            assert reader.direct == direct
            rows = np.array([7, 3, 20, 0, 3])
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(4) as pool:
                got = reader.read(rows, pool)
            assert np.array_equal(got, h5["images"][:][rows])


def test_map_style_dataset(tmp_path):
    make_dataset(tmp_path)
    ds = EuclidShardDataset(tmp_path)
    assert len(ds) == 125
    for i in (0, 60, 124):
        image, oid = ds[i]
        assert image.shape == (SIZE, SIZE) and decode_id(image) == oid
    n_labelled = int((np.arange(1000, 1125) % 7 == 0).sum())
    assert len(EuclidShardDataset(tmp_path, exclude=["label_morphology"])) == 125 - n_labelled


def test_block_stream_yields_each_image_once_in_shuffled_order(tmp_path):
    make_dataset(tmp_path)
    stream = EuclidBlockStream(tmp_path, block_size=16, shuffle_blocks=3, seed=1)
    epoch0 = [(decode_id(im), oid) for im, oid in stream]
    assert all(a == b for a, b in epoch0)
    ids0 = [oid for _, oid in epoch0]
    assert sorted(ids0) == list(range(1000, 1125)) and len(stream) == 125
    assert ids0 != sorted(ids0)
    assert [oid for _, oid in stream] == ids0  # same epoch: same order
    stream.set_epoch(1)
    ids1 = [oid for _, oid in stream]
    assert sorted(ids1) == sorted(ids0) and ids1 != ids0


def test_block_stream_splits_across_ranks_and_filters(tmp_path):
    make_dataset(tmp_path)
    seen = Counter()
    for rank in range(3):
        stream = EuclidBlockStream(tmp_path, block_size=10, rank=rank, world_size=3, exclude=["label_morphology"])
        seen.update(oid for _, oid in stream)
    expected = [i for i in range(1000, 1125) if i % 7 != 0]
    assert sorted(seen) == expected and set(seen.values()) == {1}


def test_load_into_memory(tmp_path):
    make_dataset(tmp_path)
    images, index = load_into_memory(tmp_path, threads=4)
    assert images.shape == (125, SIZE, SIZE)
    assert [decode_id(im) for im in images] == list(index["object_id"])
    images, index = load_into_memory(tmp_path, only=["label_morphology"])
    assert len(images) == len(index) > 0 and all(index["object_id"] % 7 == 0)
    assert len(load_into_memory(tmp_path, max_images=10)[0]) == 10


def test_load_into_memory_refuses_to_exceed_ram(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    monkeypatch.setattr("projects.euclid_byol.dataset._available_memory", lambda: 1_000_000)
    with pytest.raises(MemoryError, match="EuclidBlockStream"):
        load_into_memory(tmp_path)


def test_flagged_objects_are_excluded_by_default(tmp_path):
    make_dataset(tmp_path)
    index = pd.read_parquet(tmp_path / "index.parquet")
    # 128 and 256 mark problems; 2 (and other bits) must not be treated as problems.
    index.loc[index["object_id"] == 1001, "det_quality_flag"] = 128
    index.loc[index["object_id"] == 1002, "det_quality_flag"] = 256 | 2
    index.loc[index["object_id"] == 1003, "det_quality_flag"] = 2
    index.loc[index["object_id"] == 1004, "det_quality_flag"] = 386  # 256 | 128 | 2, a saturated star
    index.to_parquet(tmp_path / "index.parquet")
    dropped = {1001, 1002, 1004}

    ds = EuclidShardDataset(tmp_path)
    assert len(ds) == 125 - 3 and not dropped & set(ds.index["object_id"])
    assert "det_quality_flag" not in ds.index.columns  # only loaded for filtering
    assert len(EuclidShardDataset(tmp_path, exclude_flagged=False)) == 125
    assert "det_quality_flag" in EuclidShardDataset(tmp_path, columns=["det_quality_flag"]).index.columns

    stream = EuclidBlockStream(tmp_path, block_size=16)
    ids = [oid for _, oid in stream]
    assert len(ids) == len(stream) == 122 and not dropped & set(ids)

    images, idx = load_into_memory(tmp_path)
    assert len(images) == 122 and not dropped & set(idx["object_id"])
    assert len(load_into_memory(tmp_path, exclude_flagged=False)[0]) == 125


def test_exclude_flagged_needs_the_column(tmp_path):
    make_dataset(tmp_path)
    index = pd.read_parquet(tmp_path / "index.parquet").drop(columns="det_quality_flag")
    index.to_parquet(tmp_path / "index.parquet")
    with pytest.raises(KeyError, match="exclude_flagged=False"):
        EuclidShardDataset(tmp_path)
    assert len(EuclidShardDataset(tmp_path, exclude_flagged=False)) == 125
