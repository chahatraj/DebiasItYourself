"""Shared plotting style for DIY paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


PAPER_FONT_FAMILY = "Cantarell"

PAPER_FONT_FILES = [
    Path("/usr/share/fonts/abattis-cantarell/Cantarell-Regular.otf"),
    Path("/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf"),
    Path("/usr/share/fonts/abattis-cantarell/Cantarell-Oblique.otf"),
    Path("/usr/share/fonts/abattis-cantarell/Cantarell-BoldOblique.otf"),
]


def register_paper_font() -> None:
    """Register the paper figure font files so Matplotlib does not fall back."""
    for font_path in PAPER_FONT_FILES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def use_nimbus_sans(extra: dict[str, object] | None = None) -> None:
    """Use the current paper figure font, with optional rc overrides."""
    register_paper_font()
    rc: dict[str, object] = {
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT_FAMILY, "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if extra:
        rc.update(extra)
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = [PAPER_FONT_FAMILY, "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams.update(rc)
