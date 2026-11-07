"""Entry point for the interactive viewer: ``py -m arena_ai.viewer.game``."""

from arena_ai.config import load_config
from arena_ai.viewer.app import App


def main() -> None:
    """Open the viewer on the start menu."""
    App(load_config()).run()


if __name__ == "__main__":
    main()
