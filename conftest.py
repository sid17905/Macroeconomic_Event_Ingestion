"""Ensure the project root is importable as the `src` package under pytest."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
