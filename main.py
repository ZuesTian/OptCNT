"""Compatibility entrypoint that delegates to src.main."""

from multiprocessing import freeze_support

from src.main import main


if __name__ == "__main__":
    freeze_support()
    main()
