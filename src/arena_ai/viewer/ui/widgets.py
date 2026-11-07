"""Menu widgets with mouse and keyboard support: buttons, sliders, toggles and dropdowns."""

from collections.abc import Callable, Sequence

import pygame

from arena_ai.viewer.ui.theme import Color, Theme

ENTER_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
ROW_RADIUS = 10


class Widget:
    """Base class: a rectangle that can be focused, clicked and drawn."""

    def __init__(self, enabled_fn: Callable[[], bool] | None = None) -> None:
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._enabled_fn = enabled_fn

    @property
    def enabled(self) -> bool:
        """Disabled widgets are drawn muted and skipped by keyboard focus."""
        return self._enabled_fn() if self._enabled_fn else True

    def on_key(self, key: int) -> None:
        """Handle a key press while this widget has focus."""

    def on_mouse_down(self, pos: tuple[int, int]) -> None:
        """Handle a left click inside the widget."""

    def on_mouse_up(self, pos: tuple[int, int]) -> None:
        """Handle the left button being released anywhere."""

    def on_mouse_motion(self, pos: tuple[int, int]) -> None:
        """Handle the mouse moving anywhere (used for dragging)."""

    def draw(self, surface: pygame.Surface, theme: Theme, focused: bool) -> None:
        raise NotImplementedError

    # Widgets that pop something over the rest of the menu (a dropdown list)
    # take over input while it is open and draw it after everything else.
    @property
    def captures_input(self) -> bool:
        return False

    def handle_captured_event(self, event: pygame.event.Event) -> None:
        """Handle any event while ``captures_input`` is true."""

    def draw_overlay(self, surface: pygame.Surface, theme: Theme) -> None:
        """Draw the popped-over part, on top of every other widget."""


class Button(Widget):
    """A centered text button. The primary style uses the accent color."""

    def __init__(
        self,
        label: str,
        on_click: Callable[[], None],
        primary: bool = False,
        enabled_fn: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(enabled_fn)
        self.label = label
        self.on_click = on_click
        self.primary = primary
        self._pressed = False

    def on_key(self, key: int) -> None:
        if key in ENTER_KEYS:
            self.on_click()

    def on_mouse_down(self, pos: tuple[int, int]) -> None:
        self._pressed = True

    def on_mouse_up(self, pos: tuple[int, int]) -> None:
        pressed, self._pressed = self._pressed, False
        if pressed and self.enabled and self.rect.collidepoint(pos):
            self.on_click()

    def draw(self, surface: pygame.Surface, theme: Theme, focused: bool) -> None:
        active = focused and self.enabled
        if self.primary:
            fill = theme.color("accent_hover" if active else "accent")
            border = fill
            text_color = theme.color("on_accent")
        else:
            fill = theme.color("button_hover" if active else "button")
            border = theme.color("accent" if active else "button_border")
            text_color = theme.color("text")
        if not self.enabled:
            text_color = theme.color("text_muted")
        pygame.draw.rect(surface, fill, self.rect, border_radius=ROW_RADIUS)
        pygame.draw.rect(
            surface, border, self.rect, width=2 if active else 1, border_radius=ROW_RADIUS
        )
        theme.text(surface, self.label, "body", text_color, self.rect.center, "center")


class _Row(Widget):
    """A settings row: a label on the left and a control on the right."""

    def __init__(
        self, label: str, enabled_fn: Callable[[], bool] | None = None
    ) -> None:
        super().__init__(enabled_fn)
        self.label = label

    def draw_row(self, surface: pygame.Surface, theme: Theme, focused: bool) -> None:
        active = focused and self.enabled
        pygame.draw.rect(
            surface,
            theme.color("button_hover" if active else "button"),
            self.rect,
            border_radius=ROW_RADIUS,
        )
        pygame.draw.rect(
            surface,
            theme.color("accent" if active else "button_border"),
            self.rect,
            width=2 if active else 1,
            border_radius=ROW_RADIUS,
        )
        theme.text(
            surface,
            self.label,
            "body",
            theme.color("text" if self.enabled else "text_muted"),
            (self.rect.x + 18, self.rect.centery),
            "midleft",
        )


class Slider(_Row):
    """A 0-1 slider that can be dragged or nudged with the left/right keys."""

    def __init__(
        self,
        label: str,
        get: Callable[[], float],
        set_: Callable[[float], None],
        on_release: Callable[[], None] | None = None,
        step: float = 0.05,
        enabled_fn: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(label, enabled_fn)
        self._get = get
        self._set = set_
        self._on_release = on_release
        self._step = step
        self._dragging = False

    @property
    def track(self) -> pygame.Rect:
        return pygame.Rect(self.rect.right - 300, self.rect.centery - 3, 220, 6)

    def _set_from_x(self, x: float) -> None:
        track = self.track
        self._set(round(min(1.0, max(0.0, (x - track.x) / track.width)), 2))

    def on_key(self, key: int) -> None:
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            direction = 1 if key == pygame.K_RIGHT else -1
            self._set(round(min(1.0, max(0.0, self._get() + direction * self._step)), 2))
            if self._on_release:
                self._on_release()

    def on_mouse_down(self, pos: tuple[int, int]) -> None:
        if self.track.inflate(16, 28).collidepoint(pos):
            self._dragging = True
            self._set_from_x(pos[0])

    def on_mouse_motion(self, pos: tuple[int, int]) -> None:
        if self._dragging:
            self._set_from_x(pos[0])

    def on_mouse_up(self, pos: tuple[int, int]) -> None:
        if self._dragging:
            self._dragging = False
            if self._on_release:
                self._on_release()

    def draw(self, surface: pygame.Surface, theme: Theme, focused: bool) -> None:
        self.draw_row(surface, theme, focused)
        track = self.track
        value = self._get()
        pygame.draw.rect(surface, theme.color("track"), track, border_radius=3)
        fill = pygame.Rect(track.x, track.y, round(track.width * value), track.height)
        pygame.draw.rect(surface, theme.color("accent"), fill, border_radius=3)
        knob = (track.x + round(track.width * value), track.centery)
        pygame.draw.circle(surface, theme.color("accent"), knob, 10)
        pygame.draw.circle(surface, theme.color("on_accent"), knob, 6)
        theme.text(
            surface,
            f"{round(value * 100)}%",
            "small",
            theme.color("text_muted"),
            (self.rect.right - 16, self.rect.centery),
            "midright",
        )


class Toggle(_Row):
    """An on/off switch."""

    def __init__(
        self,
        label: str,
        get: Callable[[], bool],
        set_: Callable[[bool], None],
        enabled_fn: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(label, enabled_fn)
        self._get = get
        self._set = set_

    def on_key(self, key: int) -> None:
        if key in ENTER_KEYS + (pygame.K_LEFT, pygame.K_RIGHT):
            self._set(not self._get())

    def on_mouse_down(self, pos: tuple[int, int]) -> None:
        self._set(not self._get())

    def draw(self, surface: pygame.Surface, theme: Theme, focused: bool) -> None:
        self.draw_row(surface, theme, focused)
        on = self._get()
        pill = pygame.Rect(0, 0, 52, 28)
        pill.midright = (self.rect.right - 16, self.rect.centery)
        pygame.draw.rect(
            surface, theme.color("accent" if on else "track"), pill, border_radius=14
        )
        knob_x = pill.right - 14 if on else pill.x + 14
        pygame.draw.circle(surface, theme.color("on_accent"), (knob_x, pill.centery), 11)


def _draw_chevron(
    surface: pygame.Surface, color: Color, center: tuple[int, int], direction: int
) -> None:
    """Draw a small left (-1) or right (1) arrow head."""
    x, y = center
    points = [(x - direction * 3, y - 6), (x + direction * 3, y), (x - direction * 3, y + 6)]
    pygame.draw.lines(surface, color, False, points, 2)


class Selector(_Row):
    """Cycle through a short list of options with arrows (best for 2-3 options)."""

    def __init__(
        self,
        label: str,
        options: Sequence[tuple[str, str]],
        get: Callable[[], str],
        set_: Callable[[str], None],
        option_enabled: Callable[[str], bool] | None = None,
        enabled_fn: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(label, enabled_fn)
        self.options = list(options)
        self._get = get
        self._set = set_
        self._option_enabled = option_enabled or (lambda value: True)

    @property
    def control(self) -> pygame.Rect:
        return pygame.Rect(self.rect.right - 320, self.rect.y, 304, self.rect.height)

    def _cycle(self, step: int) -> None:
        values = [value for value, _ in self.options]
        index = values.index(self._get()) if self._get() in values else 0
        for _ in range(len(values)):
            index = (index + step) % len(values)
            if self._option_enabled(values[index]):
                self._set(values[index])
                return

    def on_key(self, key: int) -> None:
        if key == pygame.K_LEFT:
            self._cycle(-1)
        elif key in ENTER_KEYS + (pygame.K_RIGHT,):
            self._cycle(1)

    def on_mouse_down(self, pos: tuple[int, int]) -> None:
        if self.control.collidepoint(pos):
            self._cycle(-1 if pos[0] < self.control.centerx else 1)

    def draw(self, surface: pygame.Surface, theme: Theme, focused: bool) -> None:
        self.draw_row(surface, theme, focused)
        control = self.control
        current = dict(self.options).get(self._get(), "")
        muted = theme.color("text_muted")
        _draw_chevron(surface, muted, (control.x + 12, control.centery), -1)
        _draw_chevron(surface, muted, (control.right - 12, control.centery), 1)
        theme.text(
            surface,
            current,
            "small",
            theme.color("text" if self.enabled else "text_muted"),
            control.center,
            "center",
        )


class Dropdown(_Row):
    """Pick one option from a list that opens under the control.

    Use this for longer lists; ``Selector`` suits two or three options.
    """

    ITEM_HEIGHT = 38

    def __init__(
        self,
        label: str,
        options: Sequence[tuple[str, str]],
        get: Callable[[], str],
        set_: Callable[[str], None],
        option_enabled: Callable[[str], bool] | None = None,
        enabled_fn: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(label, enabled_fn)
        self.options = list(options)
        self._get = get
        self._set = set_
        self._option_enabled = option_enabled or (lambda value: True)
        self._open = False
        self._highlight = 0

    @property
    def control(self) -> pygame.Rect:
        return pygame.Rect(self.rect.right - 320, self.rect.y, 304, self.rect.height)

    @property
    def captures_input(self) -> bool:
        return self._open

    def _list_rect(self, surface_height: int = 720) -> pygame.Rect:
        control = self.control
        height = self.ITEM_HEIGHT * len(self.options) + 12
        top = control.bottom + 4
        if top + height > surface_height - 8:  # Not enough room below: open upward.
            top = control.y - 4 - height
        return pygame.Rect(control.x, top, control.width, height)

    def _item_rect(self, index: int) -> pygame.Rect:
        box = self._list_rect()
        return pygame.Rect(
            box.x + 6,
            box.y + 6 + index * self.ITEM_HEIGHT,
            box.width - 12,
            self.ITEM_HEIGHT,
        )

    def _open_list(self) -> None:
        values = [value for value, _ in self.options]
        self._highlight = values.index(self._get()) if self._get() in values else 0
        self._open = True

    def _choose(self, index: int) -> None:
        value = self.options[index][0]
        if self._option_enabled(value):
            self._set(value)
            self._open = False

    def _move_highlight(self, step: int) -> None:
        for _ in range(len(self.options)):
            self._highlight = (self._highlight + step) % len(self.options)
            if self._option_enabled(self.options[self._highlight][0]):
                return

    def on_key(self, key: int) -> None:
        if key in ENTER_KEYS + (pygame.K_RIGHT, pygame.K_LEFT):
            self._open_list()

    def on_mouse_down(self, pos: tuple[int, int]) -> None:
        if self.control.collidepoint(pos):
            self._open_list()

    def handle_captured_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._open = False
            elif event.key == pygame.K_UP:
                self._move_highlight(-1)
            elif event.key == pygame.K_DOWN:
                self._move_highlight(1)
            elif event.key in ENTER_KEYS:
                self._choose(self._highlight)
        elif event.type == pygame.MOUSEMOTION:
            for index in range(len(self.options)):
                if self._item_rect(index).collidepoint(event.pos):
                    self._highlight = index
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index in range(len(self.options)):
                if self._item_rect(index).collidepoint(event.pos):
                    self._choose(index)
                    return
            self._open = False  # A click anywhere else closes the list.

    def draw(self, surface: pygame.Surface, theme: Theme, focused: bool) -> None:
        self.draw_row(surface, theme, focused)
        control = self.control
        current = dict(self.options).get(self._get(), "")
        theme.text(
            surface,
            current,
            "small",
            theme.color("text" if self.enabled else "text_muted"),
            (control.x + 14, control.centery),
            "midleft",
        )
        x, y = control.right - 16, control.centery
        # A chevron pointing down, or up while the list is open.
        tilt = -3 if self._open else 3
        pygame.draw.lines(
            surface,
            theme.color("text_muted"),
            False,
            [(x - 6, y - tilt), (x, y + tilt), (x + 6, y - tilt)],
            2,
        )

    def draw_overlay(self, surface: pygame.Surface, theme: Theme) -> None:
        if not self._open:
            return
        theme.draw_panel(surface, self._list_rect(surface.get_height()), radius=ROW_RADIUS)
        for index, (value, label) in enumerate(self.options):
            item = self._item_rect(index)
            enabled = self._option_enabled(value)
            if index == self._highlight and enabled:
                pygame.draw.rect(surface, theme.color("button_hover"), item, border_radius=8)
            selected = value == self._get()
            color = theme.color("accent" if selected else "text" if enabled else "text_muted")
            theme.text(surface, label, "small", color, (item.x + 10, item.centery), "midleft")


class Menu:
    """A group of widgets sharing keyboard focus and mouse handling."""

    def __init__(self, widgets: Sequence[Widget]) -> None:
        self.widgets = list(widgets)
        self.focus = next((i for i, w in enumerate(self.widgets) if w.enabled), 0)

    def _move_focus(self, step: int) -> None:
        for _ in range(len(self.widgets)):
            self.focus = (self.focus + step) % len(self.widgets)
            if self.widgets[self.focus].enabled:
                return

    @property
    def captures_input(self) -> bool:
        """True while a widget (an open dropdown) needs every event itself."""
        return any(widget.captures_input for widget in self.widgets)

    def handle_event(self, event: pygame.event.Event) -> None:
        for widget in self.widgets:
            if widget.captures_input:
                widget.handle_captured_event(event)
                return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self._move_focus(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_TAB):
                self._move_focus(1)
            elif self.widgets and self.widgets[self.focus].enabled:
                self.widgets[self.focus].on_key(event.key)
        elif event.type == pygame.MOUSEMOTION:
            for index, widget in enumerate(self.widgets):
                if widget.enabled and widget.rect.collidepoint(event.pos):
                    self.focus = index
                widget.on_mouse_motion(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, widget in enumerate(self.widgets):
                if widget.enabled and widget.rect.collidepoint(event.pos):
                    self.focus = index
                    widget.on_mouse_down(event.pos)
                    break
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for widget in self.widgets:
                widget.on_mouse_up(event.pos)

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        for index, widget in enumerate(self.widgets):
            widget.draw(surface, theme, index == self.focus)
        for widget in self.widgets:  # Popped-over parts go on top of everything.
            widget.draw_overlay(surface, theme)


def stack(
    widgets: Sequence[Widget],
    center_x: int,
    top: int,
    width: int,
    height: int,
    gap: int,
) -> int:
    """Lay widgets out in a centered column and return the bottom edge."""
    y = top
    for widget in widgets:
        widget.rect = pygame.Rect(0, 0, width, height)
        widget.rect.midtop = (center_x, y)
        y += height + gap
    return y - gap
