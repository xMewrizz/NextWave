from __future__ import annotations

import argparse

from . import __version__


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
