"""Registry of survey archive clients.

Adding a new survey (e.g. LOFAR)
--------------------------------
1. Create ``astro_datatools/surveys/lofar.py`` with a class inheriting from
   :class:`~astro_datatools.surveys.base.BaseSurvey`.
2. Set ``name = "lofar"`` and implement ``query_images`` (return an astropy
   ``Table``) and ``get_cutout`` (return a
   :class:`~astro_datatools.surveys.base.Cutout`). Override ``login``/``logout``
   only if the archive needs authentication, and ``query_catalog`` if useful.
   Import heavy/optional client libraries lazily inside methods.
3. Decorate the class with ``@register_survey("lofar")`` and import the module
   in ``astro_datatools/surveys/__init__.py`` so registration runs on import.

It is then available as ``get_survey("lofar")``.
"""
from typing import Dict, List, Type

from .base import BaseSurvey


class SurveyRegistry:
    """Registry mapping survey names to :class:`BaseSurvey` subclasses."""

    def __init__(self):
        self._surveys: Dict[str, Type[BaseSurvey]] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower().strip()

    def register(self, name: str, survey_cls: Type[BaseSurvey]) -> None:
        """Register a survey class under ``name`` (case-insensitive).

        :param name: Survey name.
        :param survey_cls: Subclass of :class:`BaseSurvey`.
        """
        if not (isinstance(survey_cls, type) and issubclass(survey_cls, BaseSurvey)):
            raise TypeError(f"{survey_cls!r} is not a subclass of BaseSurvey.")
        self._surveys[self._normalize(name)] = survey_cls

    def get_class(self, name: str) -> Type[BaseSurvey]:
        """Return the survey class registered under ``name``.

        :raises ValueError: If no survey is registered under that name.
        """
        key = self._normalize(name)
        if key not in self._surveys:
            raise ValueError(
                f"Unknown survey '{name}'. Available surveys: {self.available}"
            )
        return self._surveys[key]

    def create(self, name: str, **kwargs) -> BaseSurvey:
        """Instantiate the survey registered under ``name``.

        :param name: Survey name.
        :param kwargs: Passed to the survey constructor.
        :return: Survey client instance.
        """
        return self.get_class(name)(**kwargs)

    @property
    def available(self) -> List[str]:
        """Sorted list of registered survey names."""
        return sorted(self._surveys.keys())


DEFAULT_SURVEY_REGISTRY = SurveyRegistry()


def register_survey(name: str):
    """Class decorator registering a :class:`BaseSurvey` subclass in the default registry.

    :param name: Name used with :func:`get_survey`.
    """

    def decorator(cls):
        DEFAULT_SURVEY_REGISTRY.register(name, cls)
        return cls

    return decorator


def get_survey(name: str, **kwargs) -> BaseSurvey:
    """Create a survey client by name, e.g. ``get_survey("euclid")``.

    :param name: Registered survey name (case-insensitive).
    :param kwargs: Passed to the survey constructor.
    :return: Survey client instance.
    """
    return DEFAULT_SURVEY_REGISTRY.create(name, **kwargs)


def list_surveys() -> List[str]:
    """Return the names of all registered surveys."""
    return DEFAULT_SURVEY_REGISTRY.available
