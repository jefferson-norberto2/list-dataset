"""Synthetic image distortion and degradation generator package."""

from __future__ import annotations

from dataset.distortion.algorithms.base import BaseDistortion
from dataset.distortion.algorithms.blur import BlurDistortion
from dataset.distortion.algorithms.contrast import ContrastDistortion
from dataset.distortion.algorithms.fnoise import FnoiseDistortion
from dataset.distortion.algorithms.jpeg import JpegDistortion
from dataset.distortion.algorithms.jpeg2000 import Jpeg2000Distortion
from dataset.distortion.algorithms.poisson import PoissonDistortion
from dataset.distortion.generator import DistortionGenerator

__all__ = [
    "BaseDistortion",
    "BlurDistortion",
    "ContrastDistortion",
    "DistortionGenerator",
    "FnoiseDistortion",
    "Jpeg2000Distortion",
    "JpegDistortion",
    "PoissonDistortion",
]
