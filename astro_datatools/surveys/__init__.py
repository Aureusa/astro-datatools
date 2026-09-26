"""Uniform interface for querying astronomical survey archives.

Usage::

    from astro_datatools.surveys import get_survey
    euclid = get_survey("euclid")

New archives are added by subclassing :class:`BaseSurvey` and decorating the
class with :func:`register_survey`; see :mod:`astro_datatools.surveys.registry`.
Optional client libraries (e.g. astroquery) are imported lazily.
"""
from .base import (
    BaseSurvey,
    Cutout,
    SurveyAuthenticationError,
    SurveyError,
    to_angle,
    to_skycoord,
)
from .empty_regions import EmptyRegions, find_empty_regions
from .registry import (
    DEFAULT_SURVEY_REGISTRY,
    SurveyRegistry,
    get_survey,
    list_surveys,
    register_survey,
)
from .euclid import EuclidSurvey  # noqa: E402  (registers "euclid")

__all__ = [
    "BaseSurvey",
    "Cutout",
    "SurveyError",
    "SurveyAuthenticationError",
    "EmptyRegions",
    "find_empty_regions",
    "SurveyRegistry",
    "DEFAULT_SURVEY_REGISTRY",
    "register_survey",
    "get_survey",
    "list_surveys",
    "EuclidSurvey",
    "to_skycoord",
    "to_angle",
]
