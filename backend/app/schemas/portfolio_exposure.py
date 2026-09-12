"""FEAT-003 — the portfolio build response with its industry-group
exposure block.

Kept out of ``schemas/portfolio.py`` (and out of the package's frozen
``__all__``) so the base ``ModelPortfolio`` contract is untouched: the
route returns this subclass, which adds one additive field. Anything
that already reads a ``ModelPortfolio`` keeps working; the block is an
extra key.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from .portfolio import ModelPortfolio


class ModelPortfolioWithExposure(ModelPortfolio):
    """``industry_exposure`` — weight by GICS industry group computed from
    the stored classifications, the unmapped remainder (named, weighted),
    and the relevant rows of the latest cross-industry snapshot. States:
    ``ok`` | ``taxonomy_not_imported`` | ``unavailable``; the block never
    fails the build."""
    industry_exposure: dict[str, Any] = Field(default_factory=dict)
