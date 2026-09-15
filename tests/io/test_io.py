"""Unit tests for astro_datatools.io modules."""
import os
import tempfile
import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits import HDUList, Header, ImageHDU, PrimaryHDU
import h5py

from astro_datatools.io import (
    AstroIO,
    FitsIO,
    FitsReader,
    FitsWriter,
    HDF5IO,
    HDF5Reader,
    HDF5Writer,
    ImageIO,
    ImageReader,
    ImageWriter,
    NpyIO,
    NpyReader,
    NpyWriter,
    get_fits_data,
    get_fits_header,
)
from astro_datatools.io.base import BaseReader, BaseWriter
from astro_datatools.io.registry import IORegistry


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestFitsIO:
    def test_write_and_read_fits_array(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.fits")
        arr = np.random.rand(64, 64).astype(np.float32)

        AstroIO.write(filepath, arr)
        assert os.path.exists(filepath)

        # Direct read via AstroIO returns HDUList
        hdul = AstroIO.read(filepath)
        assert isinstance(hdul, HDUList)
        np.testing.assert_allclose(hdul[0].data, arr)
        hdul.close()

    def test_fits_io_read_write_image(self, temp_dir):
        filepath = os.path.join(temp_dir, "image.fits")
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)

        header = Header()
        header["TELESCOP"] = "VLA"
        header["OBJECT"] = "RadioGalaxy"

        AstroIO.fits.write_image(filepath, arr, header=header)

        # Read image only
        data = AstroIO.fits.read_image(filepath)
        np.testing.assert_allclose(data, arr)

        # Read image + header
        data_read, hdr_read = AstroIO.fits.read_image(filepath, return_header=True)
        np.testing.assert_allclose(data_read, arr)
        assert hdr_read["TELESCOP"] == "VLA"
        assert hdr_read["OBJECT"] == "RadioGalaxy"

    def test_fits_io_read_header(self, temp_dir):
        filepath = os.path.join(temp_dir, "hdr.fits")
        arr = np.ones((5, 5))
        header = Header()
        header["FILTER"] = "LOFAR_HBA"
        AstroIO.fits.write_image(filepath, arr, header=header)

        hdr = AstroIO.fits.read_header(filepath)
        assert hdr["FILTER"] == "LOFAR_HBA"

    def test_fits_multi_extension_fallback(self, temp_dir):
        filepath = os.path.join(temp_dir, "multi.fits")
        primary = PrimaryHDU()  # no data
        img_arr = np.full((8, 8), 42.0, dtype=np.float32)
        ext = ImageHDU(data=img_arr, name="SCI")
        hdul = HDUList([primary, ext])
        hdul.writeto(filepath, overwrite=True)

        # Auto-fallback to extension 1 when primary has no data
        data = AstroIO.fits.read_image(filepath)
        assert data is not None
        np.testing.assert_allclose(data, img_arr)

    def test_fits_context_manager(self, temp_dir):
        filepath = os.path.join(temp_dir, "ctx.fits")
        arr = np.ones((4, 4))
        AstroIO.fits.write_image(filepath, arr)

        with AstroIO.fits.open(filepath) as hdul:
            assert len(hdul) == 1
            np.testing.assert_allclose(hdul[0].data, arr)

    def test_get_fits_header_and_data_helpers(self, temp_dir):
        filepath = os.path.join(temp_dir, "helpers.fits")
        arr = np.eye(4)
        AstroIO.fits.write_image(filepath, arr)

        with fits.open(filepath) as hdul:
            hdr = get_fits_header(0, hdul)
            data = get_fits_data(0, hdul)
            assert isinstance(hdr, Header)
            np.testing.assert_allclose(data, arr)


class TestHDF5IO:
    def test_write_and_read_dataset(self, temp_dir):
        filepath = os.path.join(temp_dir, "test.h5")
        arr = np.random.randn(32, 32).astype(np.float32)
        attrs = {"unit": "mJy/beam", "frequency_mhz": 144.0}

        AstroIO.hdf5.write_dataset(filepath, "observations/radio_map", arr, attrs=attrs)

        read_arr = AstroIO.hdf5.read_dataset(filepath, "observations/radio_map")
        np.testing.assert_allclose(read_arr, arr)

        read_attrs = AstroIO.hdf5.read_attrs(filepath, "observations/radio_map")
        assert read_attrs["unit"] == "mJy/beam"
        assert read_attrs["frequency_mhz"] == 144.0

    def test_write_and_read_dict(self, temp_dir):
        filepath = os.path.join(temp_dir, "nested.hdf5")
        catalog = {
            "sources": {
                "fluxes": np.array([1.2, 3.4, 5.6]),
                "positions": np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
            },
            "survey": "LoTSS",
        }

        AstroIO.hdf5.write_dict(filepath, catalog)

        loaded = AstroIO.hdf5.read_dict(filepath)
        np.testing.assert_allclose(loaded["sources"]["fluxes"], catalog["sources"]["fluxes"])
        np.testing.assert_allclose(loaded["sources"]["positions"], catalog["sources"]["positions"])

    def test_hdf5_list_keys_and_tree(self, temp_dir):
        filepath = os.path.join(temp_dir, "structure.h5")
        AstroIO.hdf5.write_dataset(filepath, "group1/dset1", np.zeros((10, 10)))
        AstroIO.hdf5.write_dataset(filepath, "group1/dset2", np.ones((5,)))
        AstroIO.hdf5.write_dataset(filepath, "group2/dset3", np.full((3, 3), 7))

        keys = AstroIO.hdf5.list_keys(filepath)
        assert set(keys) == {"group1", "group2"}

        tree_str = AstroIO.hdf5.tree(filepath)
        assert "group1" in tree_str
        assert "dset1" in tree_str
        assert "dset2" in tree_str
        assert "group2" in tree_str

    def test_hdf5_context_manager_streaming(self, temp_dir):
        filepath = os.path.join(temp_dir, "stream.h5")
        large_arr = np.arange(1000).reshape(10, 100)
        AstroIO.hdf5.write_dataset(filepath, "cube", large_arr)

        with AstroIO.hdf5.open(filepath) as f:
            slice_read = f["cube"][0:2, 10:20]
            np.testing.assert_allclose(slice_read, large_arr[0:2, 10:20])


class TestNpyIO:
    def test_npy_io(self, temp_dir):
        filepath = os.path.join(temp_dir, "array.npy")
        arr = np.random.rand(10, 20)
        AstroIO.write(filepath, arr)

        loaded = AstroIO.read(filepath)
        np.testing.assert_allclose(loaded, arr)

    def test_npz_io(self, temp_dir):
        filepath = os.path.join(temp_dir, "archive.npz")
        a = np.ones((5, 5))
        b = np.zeros((3, 3))
        AstroIO.numpy.save_npz(filepath, a=a, b=b)

        loaded = AstroIO.numpy.load_npz(filepath)
        assert "a" in loaded and "b" in loaded
        np.testing.assert_allclose(loaded["a"], a)
        np.testing.assert_allclose(loaded["b"], b)


class TestImageIO:
    def test_png_rgb_channels_first(self, temp_dir):
        filepath = os.path.join(temp_dir, "img.png")
        # RGB image (3, 32, 32) uint8
        img_data = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
        AstroIO.write(filepath, img_data)

        read_img = AstroIO.read(filepath)
        assert read_img.shape == (3, 32, 32)
        np.testing.assert_allclose(read_img, img_data)

    def test_jpeg_grayscale(self, temp_dir):
        filepath = os.path.join(temp_dir, "gray.jpg")
        gray_data = np.random.randint(0, 256, (1, 28, 28), dtype=np.uint8)
        AstroIO.image.write(filepath, gray_data)

        read_gray = AstroIO.image.read(filepath, mode="L", channels_first=True)
        assert read_gray.shape == (1, 28, 28)


class TestRegistryAndCustomFormat:
    def test_custom_format_registration(self, temp_dir):
        class DummyReader(BaseReader):
            def read(self, filepath, **kwargs):
                return "custom_read"

        class DummyWriter(BaseWriter):
            def write(self, filepath, data, **kwargs):
                with open(filepath, "w") as f:
                    f.write(str(data))

        custom_registry = IORegistry()
        custom_registry.register_reader(".custom", DummyReader())
        custom_registry.register_writer(".custom", DummyWriter())

        custom_io = AstroIO(registry=custom_registry)
        out_file = os.path.join(temp_dir, "file.custom")
        custom_io.write(out_file, "hello")

        res = custom_io.read(out_file)
        assert res == "custom_read"

    def test_supported_formats(self):
        formats = AstroIO.supported_formats()
        assert ".fits" in formats["read"]
        assert ".h5" in formats["read"]
        assert ".npy" in formats["read"]
        assert ".png" in formats["read"]
