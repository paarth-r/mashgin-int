from __future__ import annotations
from abc import ABC, abstractmethod
from src.models import Detection, MatchedProduct


class Matcher(ABC):
    """Cluster a flat list of detections (across cameras) into matched products."""

    @abstractmethod
    def match(self, detections: list[Detection]) -> list[MatchedProduct]:
        ...
