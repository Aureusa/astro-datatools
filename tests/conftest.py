"""Shared pytest configuration."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.network (they access remote archives).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="network test; use --run-network to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
