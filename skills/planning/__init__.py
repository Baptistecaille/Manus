"""
Planning skill package - Templates and utilities for filesystem-based planning.
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

__all__ = ["TEMPLATES_DIR"]
