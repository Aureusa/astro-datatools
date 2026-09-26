"""Tests for astro_datatools.surveys."""
import os

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from astro_datatools.surveys import (
    BaseSurvey,
    Cutout,
    EmptyRegions,
    EuclidSurvey,
    SurveyAuthenticationError,
    SurveyError,
    find_empty_regions,
    get_survey,
    list_surveys,
    register_survey,
)
from astro_datatools.surveys.euclid import normalize_band
from astro_datatools.surveys.registry import DEFAULT_SURVEY_REGISTRY

EDFN = SkyCoord(269.73, 66.02, unit="deg")


# --------------------------------------------------------------------------- helpers
def _make_wcs(coord, shape, scale_arcsec=0.1):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return w


def _write_fits(path, data, coord=EDFN, bunit="MJy/sr"):
    header = _make_wcs(coord, data.shape).to_header()
    header["BUNIT"] = bunit
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(path, overwrite=True)


def _gauss(img, y, x, amp, sigma):
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    img += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))


# --------------------------------------------------------------------------- base / registry
class TestBaseAndRegistry:
    def test_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseSurvey()

    def test_dummy_survey_registers_and_is_retrievable(self):
        @register_survey("dummy_test_survey")
        class DummySurvey(BaseSurvey):
            name = "dummy_test_survey"

            def __init__(self, depth=1, **kwargs):
                super().__init__(**kwargs)
                self.depth = depth

            def query_images(self, coordinate, radius, **filters):
                return Table({"name": ["img"]})

            def get_cutout(self, coordinate, size, band=None, output_path=None, **kwargs):
                data = np.zeros((10, 10))
                return Cutout(data, fits.Header(), _make_wcs(EDFN, data.shape), 1 * u.arcsec, self.name)

        try:
            assert "dummy_test_survey" in list_surveys()
            s = get_survey("Dummy_Test_Survey", depth=3)
            assert isinstance(s, DummySurvey) and s.depth == 3
            assert len(s.query_images(EDFN, 1 * u.arcmin)) == 1
            with pytest.raises(NotImplementedError):
                s.query_catalog(EDFN, 1 * u.arcmin)
            # default login/logout are no-ops that track state
            s.login()
            assert s.logged_in
            with s:
                pass
            assert not s.logged_in
        finally:
            DEFAULT_SURVEY_REGISTRY._surveys.pop("dummy_test_survey", None)

    def test_register_rejects_non_survey(self):
        with pytest.raises(TypeError):
            register_survey("bad")(object)

    def test_unknown_survey(self):
        with pytest.raises(ValueError, match="Available surveys"):
            get_survey("does-not-exist")

    def test_euclid_registered(self):
        assert "euclid" in list_surveys()
        assert isinstance(get_survey("euclid", client=object()), EuclidSurvey)

    def test_cutout_from_fits(self, tmp_path):
        path = str(tmp_path / "c.fits")
        _write_fits(path, np.ones((50, 60)))
        c = Cutout.from_fits(path, survey="x", band="VIS")
        assert c.shape == (50, 60)
        assert c.unit == "MJy/sr"
        assert c.pixel_scale.to_value(u.arcsec) == pytest.approx(0.1)
        assert c.center.separation(EDFN).arcsec < 0.2


# --------------------------------------------------------------------------- empty regions
class TestFindEmptyRegions:
    @pytest.fixture
    def image_with_sources(self):
        rng = np.random.default_rng(0)
        img = rng.normal(10.0, 1.0, size=(200, 200))
        sources = [(30, 30), (30, 150), (100, 100), (160, 40), (170, 170), (60, 90)]
        for y, x in sources:
            _gauss(img, y, x, amp=50.0, sigma=3.0)
        return img, sources

    def test_windows_avoid_sources_and_do_not_overlap(self, image_with_sources):
        img, sources = image_with_sources
        size = 32
        res = find_empty_regions(img, size=size, n=4, nsigma=5.0, stride=8)
        assert isinstance(res, EmptyRegions)
        assert res.patches.shape == (4, size, size)
        assert res.patches.dtype == np.float32
        assert np.all(res.source_fractions == 0)
        # sources (with ~3 sigma halo) do not fall inside any window
        for top, left in res.corners:
            for y, x in sources:
                assert not (top - 9 <= y < top + size + 9 and left - 9 <= x < left + size + 9)
        # non-overlapping
        for i in range(4):
            for j in range(i + 1, 4):
                (t1, l1), (t2, l2) = res.corners[i], res.corners[j]
                assert abs(t1 - t2) >= size or abs(l1 - l2) >= size
        # median subtracted, noise sigma ~1
        assert abs(np.median(res.patches)) < 0.2
        assert res.sky_sigma == pytest.approx(1.0, rel=0.1)
        assert res.source_mask[100, 100]

    def test_patches_match_image(self, image_with_sources):
        img, _ = image_with_sources
        res = find_empty_regions(img, size=20, n=2)
        t, l = res.corners[0]
        np.testing.assert_allclose(res.patches[0], img[t:t + 20, l:l + 20] - res.sky_median, rtol=1e-5)

    def test_skips_nan_and_zero_regions(self):
        rng = np.random.default_rng(1)
        img = rng.normal(5.0, 1.0, size=(128, 128))
        img[:, :64] = 0.0          # no coverage
        img[:32, 64:] = np.nan     # masked
        res = find_empty_regions(img, size=16, n=6, stride=4)
        for top, left in res.corners:
            window = img[top:top + 16, left:left + 16]
            assert np.all(np.isfinite(window)) and np.all(window != 0)
            assert left >= 64 and top >= 32

    def test_too_few_windows_raises(self):
        img = np.random.default_rng(2).normal(size=(64, 64))
        with pytest.raises(ValueError, match="only 4"):
            find_empty_regions(img, size=32, n=5, stride=8)

    def test_max_source_fraction(self, image_with_sources):
        img, _ = image_with_sources
        with pytest.raises(ValueError):
            find_empty_regions(img, size=100, n=4, max_source_fraction=0.0)

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            find_empty_regions(np.zeros((3, 10, 10)), size=4, n=1)
        with pytest.raises(ValueError):
            find_empty_regions(np.ones((10, 10)), size=20, n=1)

    def test_centers_world(self, image_with_sources):
        img, _ = image_with_sources
        res = find_empty_regions(img, size=20, n=2)
        res.wcs = _make_wcs(EDFN, img.shape)
        coords = res.centers_world()
        assert len(coords) == 2


# --------------------------------------------------------------------------- Euclid (mocked)
class FakeJob:
    def __init__(self, table):
        self.table = table

    def get_results(self):
        return self.table


class FakeEuclid:
    """Minimal stand-in for astroquery's EuclidClass."""

    def __init__(self, write_data=True):
        self.write_data = write_data
        self.queries = []
        self.cutout_calls = []
        self.login_calls = []
        self.logged_out = False

    def launch_job_async(self, query, verbose=False):
        self.queries.append(query)
        return FakeJob(
            Table(
                {
                    "file_name": ["far.fits", "EUC_MER_BGSUB-MOSAIC-VIS_TILE102159191_00.00.fits"],
                    "file_path": ["/repo/far/VIS", "/euclid/repository_idr/iqr1/Q1_R1/MER/102159191/VIS"],
                    "tile_index": [1, 102159191],
                    "instrument_name": ["VIS", "VIS"],
                    "filter_name": ["VIS", "VIS"],
                    "release_name": ["Q1_R1", "Q1_R1"],
                    "ra": [270.5, 269.7],
                    "dec": [66.3, 66.0],
                }
            )
        )

    def get_cutout(self, *, file_path, instrument, id, coordinate, radius, output_file, verbose=False):
        self.cutout_calls.append(dict(file_path=file_path, instrument=instrument, id=id,
                                      coordinate=coordinate, radius=radius, output_file=output_file))
        if self.write_data:
            rng = np.random.default_rng(3)
            _write_fits(output_file, rng.normal(0.0, 1.0, (300, 300)), coord=coordinate)
        else:
            open(output_file, "wb").close()  # anonymous access: 0-byte file
        return [output_file]

    def login(self, user=None, password=None, credentials_file=None, verbose=False):
        self.login_calls.append((user, password, credentials_file))

    def logout(self, verbose=False):
        self.logged_out = True


class TestEuclidSurvey:
    def test_query_contains_coordinate_and_band(self):
        fake = FakeEuclid()
        survey = EuclidSurvey(client=fake)
        table = survey.query_images(EDFN, 30 * u.arcsec, band="vis", instrument="VIS", release="Q1_R1")
        assert len(table) == 2
        q = fake.queries[0]
        assert "sedm.mosaic_product" in q
        assert "269.73000000" in q and "66.02000000" in q
        assert f"{(30 * u.arcsec).to_value(u.deg):.8f}" in q
        assert "filter_name='VIS'" in q
        assert "instrument_name='VIS'" in q
        assert "release_name='Q1_R1'" in q
        assert "INTERSECTS(CIRCLE('ICRS'" in q

    def test_query_accepts_tuple_and_float_radius(self):
        fake = FakeEuclid()
        EuclidSurvey(client=fake).query_images((10.0, -5.0), 60)
        assert "10.00000000, -5.00000000, 0.01666667" in fake.queries[0]
        assert "filter_name=" not in fake.queries[0]

    def test_query_rejects_injection(self):
        with pytest.raises(ValueError):
            EuclidSurvey(client=FakeEuclid()).query_images(EDFN, 1, band="VIS' OR 1=1 --")

    def test_normalize_band(self):
        assert normalize_band("vis") == "VIS"
        assert normalize_band("H") == "NIR_H"
        assert normalize_band("nir-j") == "NIR_J"
        assert normalize_band("HSC_g") == "HSC_g"

    def test_get_cutout_picks_closest_tile(self, tmp_path):
        fake = FakeEuclid()
        survey = EuclidSurvey(client=fake)
        out = str(tmp_path / "cut.fits")
        cut = survey.get_cutout(EDFN, 20 * u.arcsec, output_path=out)
        call = fake.cutout_calls[0]
        assert call["file_path"].endswith("102159191/VIS/EUC_MER_BGSUB-MOSAIC-VIS_TILE102159191_00.00.fits")
        assert call["instrument"] == "VIS" and call["id"] == "102159191"
        assert call["radius"].to_value(u.arcsec) == pytest.approx(10.0)
        assert cut.shape == (300, 300)
        assert cut.survey == "euclid" and cut.band == "VIS" and cut.path == out
        assert cut.pixel_scale.to_value(u.arcsec) == pytest.approx(0.1)
        assert cut.meta["tile_index"] == "102159191"

    def test_get_cutout_uses_cache(self, tmp_path):
        fake = FakeEuclid()
        survey = EuclidSurvey(client=fake)
        out = str(tmp_path / "cut.fits")
        survey.get_cutout(EDFN, 20 * u.arcsec, output_path=out)
        survey.get_cutout(EDFN, 20 * u.arcsec, output_path=out)
        assert len(fake.cutout_calls) == 1

    def test_empty_file_raises_and_is_removed(self, tmp_path):
        fake = FakeEuclid(write_data=False)
        survey = EuclidSurvey(client=fake)
        out = str(tmp_path / "cut.fits")
        with pytest.raises(SurveyAuthenticationError, match="login"):
            survey.get_cutout(EDFN, 20 * u.arcsec, output_path=out)
        assert not os.path.exists(out)

    def test_failed_download_raises(self, tmp_path):
        fake = FakeEuclid()
        fake.get_cutout = lambda **kw: None
        with pytest.raises(SurveyError):
            EuclidSurvey(client=fake).get_cutout(EDFN, 20, output_path=str(tmp_path / "x.fits"))

    def test_no_tile_raises(self):
        fake = FakeEuclid()
        fake.launch_job_async = lambda q, verbose=False: FakeJob(Table({"file_name": []}))
        with pytest.raises(SurveyError, match="No Euclid"):
            EuclidSurvey(client=fake).get_cutout(EDFN, 20)

    def test_login_logout(self):
        fake = FakeEuclid()
        survey = EuclidSurvey(client=fake)
        survey.login(user="someone")
        assert fake.login_calls == [("someone", None, None)]
        assert survey.logged_in
        survey.logout()
        assert fake.logged_out and not survey.logged_in

    def test_find_empty_sky_with_cache(self, tmp_path):
        fake = FakeEuclid()
        survey = EuclidSurvey(client=fake)
        res = survey.find_empty_sky(n=3, size_pix=64, cutout_size=30 * u.arcsec, cache_dir=str(tmp_path))
        assert res.patches.shape == (3, 64, 64)
        assert res.meta["bunit"] == "MJy/sr"
        assert res.meta["survey"] == "euclid"
        assert res.meta["coordinate"].separation(EDFN).arcsec < 1e-6
        assert os.path.dirname(res.meta["cutout_path"]) == str(tmp_path)
        # second call is served from the cache
        survey.find_empty_sky(n=3, size_pix=64, cutout_size=30 * u.arcsec, cache_dir=str(tmp_path))
        assert len(fake.cutout_calls) == 1


# --------------------------------------------------------------------------- live (opt-in)
@pytest.mark.network
def test_live_euclid_metadata_query():
    pytest.importorskip("astroquery")
    tiles = EuclidSurvey().query_images(EDFN, 30 * u.arcsec, band="VIS")
    assert len(tiles) >= 1
    assert all("VIS" in str(name) for name in tiles["file_name"])


def test_query_retries_then_raises():
    fake = FakeEuclid()
    calls = []

    def failing(q, verbose=False):
        calls.append(q)
        return None

    fake.launch_job_async = failing
    survey = EuclidSurvey(client=fake)
    survey.retry_wait = 0.0
    with pytest.raises(SurveyError, match="query failed"):
        survey.query_images(EDFN, 10)
    assert len(calls) == 1 + survey.query_retries


# --------------------------------------------------------------------------- Euclid gain estimate (mocked)
class FakeCatalogueEuclid:
    """Returns MER-catalogue-like aperture fluxes whose errors follow err^2 = a + k * F."""

    def __init__(self, gain, zeropoint=24.6, sky_term=2e-3, n=5000, seed=0):
        rng = np.random.default_rng(seed)
        self.k = 3631e6 * 10 ** (-0.4 * zeropoint) / gain
        f = 10 ** rng.uniform(-2, 1.5, n)
        err2 = (sky_term + self.k * f) * rng.lognormal(0, 0.05, n)
        self.table = Table({"f": f, "e": np.sqrt(err2)})
        self.queries = []

    def launch_job_async(self, query, verbose=False):
        self.queries.append(query)
        return FakeJob(self.table)


def test_estimate_gain_recovers_poisson_slope():
    fake = FakeCatalogueEuclid(gain=2800.0)
    result = EuclidSurvey(client=fake).estimate_gain(EDFN, 0.1 * u.deg)
    np.testing.assert_allclose(result["gain"], 2800.0, rtol=0.05)
    query = fake.queries[0]
    assert "flux_vis_2fwhm_aper" in query and "fluxerr_vis_2fwhm_aper" in query
    assert "spurious_flag=0" in query and "CIRCLE('ICRS', 269.730000, 66.020000, 0.100000)" in query


def test_estimate_gain_rejects_bad_tokens():
    with pytest.raises(ValueError):
        EuclidSurvey(client=FakeCatalogueEuclid(gain=1.0)).estimate_gain(EDFN, aperture="2fwhm; DROP")


def test_empty_regions_noise_correlation_kernel():
    from astro_datatools.augment.noise import _convolve_spatial, kernel_autocorrelation

    kernel_true = np.array([[0.03, 0.12, 0.03], [0.12, 1.0, 0.12], [0.03, 0.12, 0.03]])
    rng = np.random.default_rng(3)
    sky = _convolve_spatial(rng.normal(0, 1, (800, 800)), kernel_true / np.sqrt((kernel_true**2).sum()))
    _gauss(sky, 400, 400, 500, 3)  # a bright source (added in place) that the masks must exclude
    regions = find_empty_regions(sky, size=200, n=4)
    masks = regions.patch_masks()
    assert masks.shape == (4, 200, 200) and masks.dtype == bool
    kernel, acf = regions.noise_correlation_kernel()
    np.testing.assert_allclose(acf, kernel_autocorrelation(kernel_true, 1), atol=0.02)
