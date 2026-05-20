from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from src.models import BBox


class Detector(ABC):
    """Detect items in a single image, returning a list of bounding boxes."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[BBox]:
        """
        Args:
            image: HxWx3 BGR numpy array.
        Returns:
            List of BBox in pixel coordinates.
        """
        ...
