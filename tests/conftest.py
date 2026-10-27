"""Shared pytest fixtures for dataset curator tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def clean_rgb_image() -> np.ndarray:
    """Creates a natural-like high-contrast RGB test pattern."""
    h, w = 600, 800
    y, x = np.mgrid[0:h, 0:w]
    # Smooth gradient with superimposed high-frequency checkerboard
    base = ((x / w) * 180 + 35).astype(np.uint8)
    checker = (((x // 16) + (y // 16)) % 2) * 50
    pattern = np.clip(base + checker, 20, 230).astype(np.uint8)

    rgb = np.stack(
        [pattern, np.roll(pattern, 10, axis=0), np.roll(pattern, 20, axis=1)], axis=-1
    )
    return rgb


@pytest.fixture
def flat_image() -> np.ndarray:
    """Creates a completely flat gray image (zero contrast, zero entropy, zero sharpness)."""
    return np.full((512, 512, 3), 128, dtype=np.uint8)


@pytest.fixture
def clipped_image() -> np.ndarray:
    """Creates an image with high clipping (under- and over-exposure)."""
    img = np.full((100, 100), 128, dtype=np.uint8)
    # 5% black (0), 5% white (255) -> 10% clipped
    img[:5, :] = 0
    img[5:10, :] = 255
    return img
