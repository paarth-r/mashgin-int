from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from src.models import CatalogItem


class Recognizer(ABC):
    """Identify a crop against the product catalog."""

    @abstractmethod
    def identify(self, crop: np.ndarray) -> tuple[CatalogItem | None, float]:
        """
        Args:
            crop: HxWx3 BGR numpy array of an item region.
        Returns:
            (CatalogItem or None if below threshold, similarity score 0-1)
        """
        ...

    @abstractmethod
    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Return the normalised embedding vector for a crop (for matcher use)."""
        ...
