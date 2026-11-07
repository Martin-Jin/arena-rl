"""The interface every viewer scene implements."""

from typing import Protocol

import pygame


class Scene(Protocol):
    """A full-screen view that the app forwards events, time and drawing to."""

    def handle_event(self, event: pygame.event.Event) -> None: ...

    def update(self, dt: float) -> None:
        """Advance animations by ``dt`` real seconds."""

    def draw(self, screen: pygame.Surface) -> None: ...
