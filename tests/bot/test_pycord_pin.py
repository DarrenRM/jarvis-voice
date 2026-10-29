"""pyproject's py-cord pin and the DAVE patch's version gate must agree.

apply() is a no-op on any other py-cord, so a looser pin would install a release where
voice receive silently runs unpatched.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

from src.bot.pycord_dave_patch import PATCHED_VERSIONS

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def pycord_requirement() -> Requirement:
    deps = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["dependencies"]
    return next(Requirement(d) for d in deps if Requirement(d).name == "py-cord")


def test_pin_allows_exactly_the_patched_releases():
    spec = pycord_requirement().specifier
    for version in PATCHED_VERSIONS:
        assert spec.contains(version), f"{version} is patched but not allowed by {spec}"
    for version in ("2.6.0", "2.7.1", "2.8.2", "2.9.0"):
        assert not spec.contains(version), f"{spec} allows unpatched py-cord {version}"
