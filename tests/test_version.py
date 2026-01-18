"""Unit tests for version module."""

from chimera import __version__
from chimera.version import (
    PROJECT_DESCRIPTION,
    PROJECT_FULL_NAME,
    PROJECT_NAME,
    __author__,
    __license__,
    __version_info__,
)


class TestVersion:
    """Tests for version information."""

    def test_version_is_string(self) -> None:
        """Version should be a string."""
        assert isinstance(__version__, str)

    def test_version_format(self) -> None:
        """Version should follow semantic versioning format."""
        parts = __version__.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_version_info_is_tuple(self) -> None:
        """Version info should be a tuple of integers."""
        assert isinstance(__version_info__, tuple)
        assert len(__version_info__) == 3
        for part in __version_info__:
            assert isinstance(part, int)

    def test_version_info_matches_version(self) -> None:
        """Version info should match version string."""
        expected = tuple(int(x) for x in __version__.split("."))
        assert __version_info__ == expected


class TestProjectMetadata:
    """Tests for project metadata."""

    def test_project_name(self) -> None:
        """Project name should be CHIMERA."""
        assert PROJECT_NAME == "CHIMERA"

    def test_project_full_name(self) -> None:
        """Project full name should expand the acronym."""
        assert "Calibrated" in PROJECT_FULL_NAME
        assert "Meta-cognitive" in PROJECT_FULL_NAME

    def test_project_description_not_empty(self) -> None:
        """Project description should not be empty."""
        assert len(PROJECT_DESCRIPTION) > 0

    def test_author_defined(self) -> None:
        """Author should be defined."""
        assert __author__ is not None
        assert len(__author__) > 0

    def test_license_is_apache(self) -> None:
        """License should be Apache 2.0."""
        assert "Apache" in __license__
