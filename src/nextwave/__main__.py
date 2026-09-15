"""Minimal package entry point used by the repository smoke check."""

from __future__ import annotations

import argparse

from nextwave import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="nextwave")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
