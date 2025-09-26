from .geomux import geomux
from .utils import assignment_statistics

__all__ = ["geomux", "assignment_statistics"]

import logging

logging.basicConfig(
    format="%(asctime)s :: %(levelname)s :: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
