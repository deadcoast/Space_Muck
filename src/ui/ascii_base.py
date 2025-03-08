"""
Space Muck ASCII UI Framework
----------------------------
An exponential ASCII UI system for Space Muck, a galaxy space mining game
with cellular automaton and symbiotic system inspirations.
"""

import contextlib
import time
import random
import curses
import math
import pygame
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any


class UIStyle(Enum):
    """Different visual styles for UI components"""

    SYMBIOTIC = auto()  # Organic, evolving patterns
    ASTEROID = auto()  # Rough, mineral-like patterns
    MECHANICAL = auto()  # Ship/tech inspired rigid patterns
    QUANTUM = auto()  # Probability-wave inspired patterns
    FLEET = auto()  # Organized, military-like patterns

    @classmethod
    def get_border_chars(cls, style) -> Dict[str, str]:
        """Get border characters based on the selected style"""
        borders = {
            cls.SYMBIOTIC: {
                "tl": "╭",
                "tr": "╮",
                "bl": "╰",
                "br": "╯",
                "h": "─",
                "v": "│",
                "fill": " ",
            },
            cls.ASTEROID: {
                "tl": "┌",
                "tr": "┐",
                "bl": "└",
                "br": "┘",
                "h": "═",
                "v": "║",
                "fill": "·",
            },
            cls.MECHANICAL: {
                "tl": "╔",
                "tr": "╗",
                "bl": "╚",
                "br": "╝",
                "h": "═",
                "v": "║",
                "fill": " ",
            },
            cls.QUANTUM: {
                "tl": "╒",
                "tr": "╕",
                "bl": "╘",
                "br": "╛",
                "h": "╌",
                "v": "╎",
                "fill": "·",
            },
            cls.FLEET: {
                "tl": "┏",
                "tr": "┓",
                "bl": "┗",
                "br": "┛",
                "h": "━",
                "v": "┃",
                "fill": " ",
            },
        }
        return borders.get(style, borders[cls.MECHANICAL])

    @classmethod
    def get_style_for_converter(cls, converter_type: str) -> "UIStyle":
        """Get appropriate style based on converter type"""
        style_map = {
            "BASIC": cls.MECHANICAL,
            "ADVANCED": cls.QUANTUM,
            "ORGANIC": cls.SYMBIOTIC,
            "FLEET": cls.FLEET,
            "MINING": cls.ASTEROID,
        }
        return style_map.get(converter_type.upper(), cls.MECHANICAL)


class AnimationStyle(Enum):
    """Animation styles for transitions"""

    CELLULAR = auto()  # Cell-by-cell reveal mimicking Game of Life
    FRACTAL = auto()  # Recursive splitting pattern
    WARP = auto()  # Space-warp style transitions
    QUANTUM_FLUX = auto()  # Probability wave collapse
    MINERAL_GROWTH = auto()  # Crystal-like growth patterns

    @classmethod
    def get_animation_for_style(cls, ui_style: UIStyle) -> "AnimationStyle":
        """Get appropriate animation style based on UI style"""
        animation_map = {
            UIStyle.SYMBIOTIC: cls.CELLULAR,
            UIStyle.ASTEROID: cls.MINERAL_GROWTH,
            UIStyle.MECHANICAL: cls.FRACTAL,
            UIStyle.QUANTUM: cls.QUANTUM_FLUX,
            UIStyle.FLEET: cls.WARP,
        }
        return animation_map.get(ui_style, cls.CELLULAR)


class UIElement:
    """Base class for all UI elements with cellular automaton-inspired behaviors"""

    def __init__(self, x: int, y: int, width: int, height: int, style: UIStyle):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.style = style
        self.visible = True
        self.border_chars = UIStyle.get_border_chars(style)
        self.animation_history = []  # Tracks cellular automaton states
        self.animation_style = AnimationStyle.get_animation_for_style(style)
        # Initialize style-specific animation state
        self.animation_state = {
            "active": False,
            "progress": 0.0,
            "duration": 0.5,  # seconds
            "start_time": 0,
            "cells": [[False for _ in range(width)] for _ in range(height)],
            "phase": 0,
            "style_data": self._init_style_data(),
        }

    def start_animation(self, duration: float = 0.5) -> None:
        """Start an animation sequence.

        Args:
            duration: Animation duration in seconds
        """
        self.animation_state["active"] = True
        self.animation_state["progress"] = 0.0
        self.animation_state["duration"] = duration
        self.animation_state["start_time"] = time.time()
        self.animation_state["phase"] = 0

        if self.animation_style == AnimationStyle.CELLULAR:
            # Initialize with random cells
            self.animation_state["cells"] = [
                [random.random() < 0.3 for _ in range(self.width)]
                for _ in range(self.height)
            ]
        elif self.animation_style == AnimationStyle.QUANTUM_FLUX:
            # Initialize with probability wave pattern
            self.animation_state["cells"] = [
                [math.sin(x / 2 + y / 2) > 0 for x in range(self.width)]
                for y in range(self.height)
            ]

    def update_animation(self) -> None:
        """Update animation state based on elapsed time"""
        if not self.animation_state["active"]:
            return

        current_time = time.time()
        elapsed = current_time - self.animation_state["start_time"]
        self.animation_state["progress"] = min(
            1.0, elapsed / self.animation_state["duration"]
        )

        if self.animation_state["progress"] >= 1.0:
            self.animation_state["active"] = False
            return

        if self.animation_style == AnimationStyle.CELLULAR:
            if elapsed - self.animation_state["phase"] > 0.1:  # Update every 0.1s
                self._evolve_cellular_grid()
                self.animation_state["phase"] = elapsed
        elif self.animation_style == AnimationStyle.QUANTUM_FLUX:
            if elapsed - self.animation_state["phase"] > 0.05:  # Update every 0.05s
                self._update_quantum_pattern(elapsed)
                self.animation_state["phase"] = elapsed

    def _update_quantum_pattern(self, elapsed: float) -> None:
        """Update quantum flux animation pattern"""
        freq = 4.0  # Base frequency
        for y in range(self.height):
            for x in range(self.width):
                phase = elapsed * freq
                value = math.sin(x / 2 + y / 2 + phase)
                self.animation_state["cells"][y][x] = value > 0

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> pygame.Rect:
        """Draw the UI element with animation effects.

        Args:
            surface: Pygame surface to draw on
            font: Font to use for text

        Returns:
            pygame.Rect: The area that was drawn
        """
        if not self.visible:
            return pygame.Rect(self.x, self.y, self.width, self.height)

        # Create element surface
        char_width, char_height = font.size("X")
        element_width = self.width * char_width
        element_height = self.height * char_height
        element_surf = pygame.Surface((element_width, element_height), pygame.SRCALPHA)

        # Draw animation effects if active
        if self.animation_state["active"]:
            self.update_animation()
            self._draw_animation_effects(element_surf, font)

        # Draw border with style-specific characters
        for i in range(self.width):
            if i == 0:
                self._draw_char(element_surf, font, 0, 0, self.border_chars["tl"])
                self._draw_char(
                    element_surf, font, 0, self.height - 1, self.border_chars["bl"]
                )
            elif i == self.width - 1:
                self._draw_char(element_surf, font, i, 0, self.border_chars["tr"])
                self._draw_char(
                    element_surf, font, i, self.height - 1, self.border_chars["br"]
                )
            else:
                self._draw_char(element_surf, font, i, 0, self.border_chars["h"])
                self._draw_char(
                    element_surf, font, i, self.height - 1, self.border_chars["h"]
                )

        for i in range(1, self.height - 1):
            self._draw_char(element_surf, font, 0, i, self.border_chars["v"])
            self._draw_char(
                element_surf, font, self.width - 1, i, self.border_chars["v"]
            )

        return surface.blit(element_surf, (self.x, self.y))

    def _draw_char(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        char: str,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Draw a single character at the specified position."""
        char_surf = font.render(char, True, color)
        char_width = font.size("X")[0]
        surface.blit(char_surf, (x * char_width, y * font.get_height()))

    def _init_style_data(self) -> Dict[str, Any]:
        """Initialize style-specific animation data."""
        if self.style == UIStyle.QUANTUM:
            return {
                "particles": [
                    (
                        random.randint(0, self.width - 1),
                        random.randint(0, self.height - 1),
                        random.random() * 2 * math.pi,
                    )
                    for _ in range(5)
                ],
                "wave_phase": 0.0,
            }
        elif self.style == UIStyle.SYMBIOTIC:
            return {
                "growth_points": [
                    (
                        random.randint(0, self.width - 1),
                        random.randint(0, self.height - 1),
                    )
                    for _ in range(3)
                ],
                "tendrils": [],
            }
        elif self.style == UIStyle.MECHANICAL:
            return {
                "gears": [
                    (
                        random.randint(0, self.width - 1),
                        random.randint(0, self.height - 1),
                        random.choice([4, 6, 8]),
                    )
                    for _ in range(3)
                ],
                "rotation": 0.0,
            }
        elif self.style == UIStyle.ASTEROID:
            return {
                "asteroids": [
                    (
                        random.randint(0, self.width - 1),
                        random.randint(0, self.height - 1),
                        random.random() * 2 * math.pi,
                        random.random() * 0.5 + 0.5,
                    )
                    for _ in range(5)
                ]
            }
        elif self.style == UIStyle.FLEET:
            return {
                "ships": [
                    (
                        random.randint(0, self.width - 1),
                        random.randint(0, self.height - 1),
                        random.choice(["<", ">", "^", "v"]),
                    )
                    for _ in range(4)
                ],
                "formation_phase": 0.0,
            }
        return {}

    def _draw_animation_effects(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw current animation frame effects."""
        if not self.animation_state["active"]:
            return

        if self.style == UIStyle.QUANTUM:
            self._draw_quantum_animation(surface, font)
        elif self.style == UIStyle.SYMBIOTIC:
            self._draw_symbiotic_animation(surface, font)
        elif self.style == UIStyle.MECHANICAL:
            self._draw_mechanical_animation(surface, font)
        elif self.style == UIStyle.ASTEROID:
            self._draw_asteroid_animation(surface, font)
        elif self.style == UIStyle.FLEET:
            self._draw_fleet_animation(surface, font)
        else:
            self._draw_cellular_animation(surface, font)

    def _draw_asteroid_animation(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw asteroid field animation frame."""
        data = self.animation_state["style_data"]
        progress = self.animation_state["progress"]

        # Update and draw asteroids
        for i, (x, y, angle, size) in enumerate(data["asteroids"]):
            # Move asteroids
            new_x = x + math.cos(angle) * 0.2
            new_y = y + math.sin(angle) * 0.2

            # Wrap around edges
            new_x = (new_x + self.width) % self.width
            new_y = (new_y + self.height) % self.height

            data["asteroids"][i] = (new_x, new_y, angle, size)

            # Draw asteroid with size-based character
            if size > 0.8:
                char = "*"
            elif size > 0.6:
                char = "+"
            else:
                char = "."
            color = (200, 180, 160, int(255 * progress))
            self._draw_char(surface, font, int(new_x), int(new_y), char, color)

    def _draw_quantum_animation(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw quantum flux animation frame."""
        data = self.animation_state["style_data"]
        progress = self.animation_state["progress"]

        # Update particle positions
        for i, (x, y, angle) in enumerate(data["particles"]):
            # Move particles in wave pattern
            new_x = x + math.cos(angle + data["wave_phase"]) * 0.2
            new_y = y + math.sin(angle + data["wave_phase"]) * 0.2

            # Wrap around edges
            new_x = (new_x + self.width) % self.width
            new_y = (new_y + self.height) % self.height

            data["particles"][i] = (new_x, new_y, angle)

            # Draw quantum particle
            chars = [".", "*", "o", "O", "0", "@", "#", "%", "&"]
            idx = int(progress * (len(chars) - 1))
            char = chars[idx]

            # Calculate color with quantum interference pattern
            wave = 0.7 + 0.3 * math.sin(data["wave_phase"] * 2)
            color = (int(180 * wave), int(180 * wave), 255, int(255 * progress))

            self._draw_char(surface, font, int(new_x), int(new_y), char, color)

        # Update wave phase
        data["wave_phase"] += 0.1

    def animate(self, stdscr, style: AnimationStyle = AnimationStyle.CELLULAR):
        """Animate the appearance of the element"""
        if style == AnimationStyle.CELLULAR:
            self._cellular_animation(stdscr)
        elif style == AnimationStyle.FRACTAL:
            self._fractal_animation(stdscr)
        elif style == AnimationStyle.WARP:
            self._warp_animation(stdscr)
        elif style == AnimationStyle.QUANTUM_FLUX:
            self._quantum_animation(stdscr)
        elif style == AnimationStyle.MINERAL_GROWTH:
            self._mineral_growth_animation(stdscr)

    def _draw_fleet_animation(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw fleet movement animation frame."""
        data = self.animation_state["style_data"]
        progress = self.animation_state["progress"]

        # Update formation phase
        data["formation_phase"] += 0.1

        # Update and draw ships
        for i, (x, y, ship_type) in enumerate(data["ships"]):
            # Move ships in formation pattern
            offset = math.sin(data["formation_phase"] + i * 0.5) * 0.3
            new_x = x + offset
            new_y = y + math.cos(data["formation_phase"] + i * 0.5) * 0.2

            # Keep ships in bounds
            new_x = max(0, min(self.width - 1, new_x))
            new_y = max(0, min(self.height - 1, new_y))

            data["ships"][i] = (new_x, new_y, ship_type)

            # Draw ship
            color = (180, 220, 255, int(255 * progress))
            self._draw_char(surface, font, int(new_x), int(new_y), ship_type, color)

    def _draw_symbiotic_animation(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw symbiotic growth animation frame."""
        data = self.animation_state["style_data"]
        progress = self.animation_state["progress"]

        # Grow tendrils from growth points
        if random.random() < 0.3:
            for x, y in data["growth_points"]:
                if len(data["tendrils"]) < 20:  # Limit total tendrils
                    angle = random.random() * 2 * math.pi
                    data["tendrils"].append((x, y, angle, 1))

        # Update and draw tendrils
        new_tendrils = []
        for x, y, angle, length in data["tendrils"]:
            if length < 5:  # Limit tendril length
                new_x = x + math.cos(angle) * length * 0.5
                new_y = y + math.sin(angle) * length * 0.5

                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    char = random.choice(["/", "\\", "|", "-"])
                    color = (100, 255, 100, int(255 * progress))
                    self._draw_char(surface, font, int(x), int(y), char, color)
                    new_tendrils.append((new_x, new_y, angle, length + 0.5))

        data["tendrils"] = new_tendrils

    def _draw_mechanical_animation(
        self, surface: pygame.Surface, font: pygame.font.Font
    ) -> None:
        """Draw mechanical gear animation frame."""
        data = self.animation_state["style_data"]
        progress = self.animation_state["progress"]

        # Update gear rotation
        data["rotation"] += 0.1

        # Draw gears
        for x, y, size in data["gears"]:
            # Draw gear teeth
            for i in range(size):
                angle = data["rotation"] + (i * 2 * math.pi / size)
                tooth_x = x + math.cos(angle) * 1.5
                tooth_y = y + math.sin(angle) * 1.5

                if 0 <= tooth_x < self.width and 0 <= tooth_y < self.height:
                    char = "+"  # Gear tooth
                    color = (200, 200, 200, int(255 * progress))
                    self._draw_char(
                        surface, font, int(tooth_x), int(tooth_y), char, color
                    )

            # Draw gear center
            if 0 <= x < self.width and 0 <= y < self.height:
                char = "@"  # Gear center
                color = (180, 180, 180, int(255 * progress))
                self._draw_char(surface, font, int(x), int(y), char, color)

    def _fractal_animation(self, stdscr):
        """Recursive revealing animation"""

        def draw_segment(start_x, start_y, segment_width, segment_height):
            if segment_width <= 1 or segment_height <= 1:
                return

            # Draw the outline of this segment
            for i in range(segment_width):
                with contextlib.suppress(curses.error):
                    stdscr.addstr(start_y, start_x + i, self.border_chars["h"])
                    stdscr.addstr(
                        start_y + segment_height - 1,
                        start_x + i,
                        self.border_chars["h"],
                    )
            for i in range(segment_height):
                with contextlib.suppress(curses.error):
                    stdscr.addstr(start_y + i, start_x, self.border_chars["v"])
                    stdscr.addstr(
                        start_y + i, start_x + segment_width - 1, self.border_chars["v"]
                    )
            stdscr.refresh()
            time.sleep(0.05)

            # Split into quadrants and recurse
            mid_x = segment_width // 2
            mid_y = segment_height // 2

            draw_segment(start_x, start_y, mid_x, mid_y)
            draw_segment(start_x + mid_x, start_y, segment_width - mid_x, mid_y)
            draw_segment(start_x, start_y + mid_y, mid_x, segment_height - mid_y)
            draw_segment(
                start_x + mid_x,
                start_y + mid_y,
                segment_width - mid_x,
                segment_height - mid_y,
            )

        draw_segment(self.x, self.y, self.width, self.height)
        # Final reveal
        self.draw(stdscr)

    def _warp_animation(self, stdscr):
        """Space-warp inspired expanding animation"""
        # Start from center and expand outward
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2

        max_radius = max(self.width, self.height)

        for radius in range(1, max_radius, 2):
            # Clear previous
            for y in range(
                max(0, self.y), min(self.y + self.height, stdscr.getmaxyx()[0])
            ):
                for x in range(
                    max(0, self.x), min(self.x + self.width, stdscr.getmaxyx()[1])
                ):
                    with contextlib.suppress(curses.error):
                        stdscr.addstr(y, x, " ")
            # Draw expanding circles
            for angle in range(0, 360, 10):
                angle_rad = angle * 3.14159 / 180
                x = int(center_x + radius * 0.5 * math.cos(angle_rad))
                y = int(center_y + radius * 0.25 * math.sin(angle_rad))

                if 0 <= x < stdscr.getmaxyx()[1] and 0 <= y < stdscr.getmaxyx()[0]:
                    with contextlib.suppress(curses.error):
                        stdscr.addstr(y, x, "·")
            stdscr.refresh()
            time.sleep(0.05)

        # Final reveal
        self.draw(stdscr)

    def _quantum_animation(self, stdscr):
        """Quantum-inspired probability wave collapse animation"""
        # Create a grid of probabilities
        probs = [
            [random.random() for _ in range(self.width)] for _ in range(self.height)
        ]

        # Animation phases
        for _ in range(5):
            # Update probabilities based on neighbors (quantum-like interference)
            new_probs = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
            for y in range(self.height):
                for x in range(self.width):
                    # Average with neighbors
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                neighbors.append(probs[ny][nx])

                    # Calculate new probability with quantum-like randomness
                    avg = sum(neighbors) / len(neighbors) if neighbors else 0.5
                    new_probs[y][x] = (avg + probs[y][x]) / 2 + (
                        random.random() - 0.5
                    ) * 0.1
                    new_probs[y][x] = max(0, min(1, new_probs[y][x]))  # Clamp to [0,1]

            # Update grid
            probs = new_probs

            # Visualize the current state - higher prob = more visible
            for y in range(self.height):
                for x in range(self.width):
                    if random.random() < probs[y][x]:  # Probabilistic rendering
                        with contextlib.suppress(curses.error):
                            char = " ·:+*#@"[min(6, int(probs[y][x] * 7))]
                            stdscr.addstr(self.y + y, self.x + x, char)
            stdscr.refresh()
            time.sleep(0.1)

        # Final "collapse" to actual UI
        self.draw(stdscr)

    def _mineral_growth_animation(self, stdscr):
        """Crystalline growth pattern animation"""
        # Start with seed points
        active_points = []
        crystal = [[False for _ in range(self.width)] for _ in range(self.height)]

        # Seed the border
        for i in range(self.width):
            active_points.extend(((i, 0), (i, self.height - 1)))
            crystal[0][i] = True
            crystal[self.height - 1][i] = True

        for i in range(1, self.height - 1):
            active_points.extend(((0, i), (self.width - 1, i)))
            crystal[i][0] = True
            crystal[i][self.width - 1] = True

        # Crystal growth patterns
        growth_chars = ["·", ":", "▪", "▫", "▣", "▨", "◆", "◇", "◈"]

        # Animation steps
        for _ in range(min(self.width, self.height) // 2):
            if not active_points:
                break

            # Select random active points to grow from
            growth_points = random.sample(active_points, min(len(active_points), 5))
            new_active_points = []

            for x, y in growth_points:
                # Try to grow in random directions
                directions = [
                    (0, 1),
                    (1, 0),
                    (0, -1),
                    (-1, 0),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ]
                random.shuffle(directions)

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and not crystal[ny][nx]
                    ):
                        # Grow in this direction
                        crystal[ny][nx] = True
                        new_active_points.append((nx, ny))

                        # Draw the crystal growth
                        char = random.choice(growth_chars)
                        with contextlib.suppress(curses.error):
                            stdscr.addstr(self.y + ny, self.x + nx, char)
            # Update active points
            active_points = list(set(active_points + new_active_points))
            # Remove points that can't grow anymore
            active_points = [
                (x, y)
                for x, y in active_points
                if any(
                    0 <= x + dx < self.width
                    and 0 <= y + dy < self.height
                    and not crystal[y + dy][x + dx]
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                )
            ]

            stdscr.refresh()
            time.sleep(0.05)

        # Final draw
        self.draw(stdscr)


class Menu(UIElement):
    """Interactive menu with dynamically evolving options"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
        options: List[str],
        style: UIStyle = UIStyle.SYMBIOTIC,
    ):
        super().__init__(x, y, width, height, style)
        self.title = title
        self.options = options
        self.selected_index = 0
        self.active = False
        # Evolutionary metrics for each option (how often used)
        self.option_metrics = [0] * len(options)
        # Mutation rate for menu evolution
        self.mutation_rate = 0.05

    def draw(self, stdscr):
        """Draw menu with options"""
        super().draw(stdscr)

        # Draw title
        title_x = self.x + (self.width - len(self.title)) // 2
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 1, title_x, self.title, curses.A_BOLD)
        # Draw options with evolutionary prominence
        for i, option in enumerate(self.options):
            # Option appearance changes based on usage patterns
            prefix = "  "
            if i == self.selected_index and self.active:
                prefix = "▶ "  # Selection indicator
                with contextlib.suppress(curses.error):
                    # Highlighted option
                    stdscr.addstr(
                        self.y + 3 + i, self.x + 2, prefix + option, curses.A_REVERSE
                    )
            else:
                # Normal option with prominence-based rendering
                attr = curses.A_NORMAL
                # Calculate prominence factor based on usage
                prominence = min(1.0, 0.5 + self.option_metrics[i] * 0.1)

                if prominence > 0.8:
                    attr = curses.A_BOLD

                # Apply adaptive decoration based on usage
                evolving_chars = ["·", "∘", "○", "◌", "◍", "◎", "●"]
                decoration = evolving_chars[
                    min(len(evolving_chars) - 1, int(prominence * len(evolving_chars)))
                ]

                prefix = f"{decoration} "

                with contextlib.suppress(curses.error):
                    stdscr.addstr(self.y + 3 + i, self.x + 2, prefix + option, attr)

    def handle_input(self, key: int) -> Optional[str]:
        """Handle navigation and selection input"""
        if not self.active:
            return None

        if key == curses.KEY_UP:
            self.selected_index = (self.selected_index - 1) % len(self.options)
        elif key == curses.KEY_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.options)
        elif key in [curses.KEY_ENTER, ord("\n"), ord(" ")]:
            # Record the selection in metrics
            self.option_metrics[self.selected_index] += 1
            self._evolve_menu()
            return self.options[self.selected_index]

        return None

    def _evolve_menu(self):
        """Simulate evolution of the menu based on usage patterns"""
        # Reorder options based on usage, but with randomness
        if random.random() < self.mutation_rate:
            # Sometimes shuffle the least used options
            least_used = sorted(
                range(len(self.option_metrics)), key=lambda i: self.option_metrics[i]
            )[:2]
            if len(least_used) >= 2:
                i, j = least_used[0], least_used[1]
                self.options[i], self.options[j] = self.options[j], self.options[i]
                self.option_metrics[i], self.option_metrics[j] = (
                    self.option_metrics[j],
                    self.option_metrics[i],
                )


class FleetDisplay(UIElement):
    """Displays fleet information with cellular automaton-based animations"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        fleet_data: Dict[str, Any],
        style: UIStyle = UIStyle.FLEET,
    ):
        super().__init__(x, y, width, height, style)
        self.fleet_data = fleet_data
        self.animation_frame = 0
        self.ship_patterns = self._generate_ship_patterns()

    def _generate_ship_patterns(self) -> Dict[str, List[str]]:
        """Generate ASCII art patterns for different ship types"""
        return {
            "miner": [
                "⤧≡≡≡◊",
                "⤧≡≡◊≡",
                "⤧≡◊≡≡",
                "⤧◊≡≡≡",
            ],
            "fighter": [
                "/=|=\\",
                "/≡|≡\\",
                "/≡∫≡\\",
                "/≡|≡\\",
            ],
            "capital": [
                "<[≡≡≡≡≡]>",
                "<[≡≡≡≡≡]>",
                "<[≡≡≡≡≡]>",
                "<[≡≡≡≡≡]>",
            ],
        }

    def draw(self, stdscr):
        """Draw the fleet display with animated ships"""
        super().draw(stdscr)

        # Draw title
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 1, self.x + 2, "FLEET STATUS", curses.A_BOLD)
        # Draw ships with animation
        y_offset = 3
        for ship_type, count in self.fleet_data.items():
            if ship_type in self.ship_patterns:
                pattern = self.ship_patterns[ship_type][
                    self.animation_frame % len(self.ship_patterns[ship_type])
                ]

                with contextlib.suppress(curses.error):
                    # Ship type and count
                    stdscr.addstr(
                        self.y + y_offset,
                        self.x + 2,
                        f"{ship_type.capitalize()}: {count}",
                    )

                    # Animated ships (show up to 5)
                    for i in range(min(count, 5)):
                        ship_x = self.x + 18 + i * (len(pattern) + 1)
                        if ship_x + len(pattern) < self.x + self.width - 1:
                            stdscr.addstr(self.y + y_offset, ship_x, pattern)
                y_offset += 2

        # Update animation frame
        self.animation_frame += 1

    def update_fleet(self, new_fleet_data: Dict[str, Any]):
        """Update the fleet information"""
        self.fleet_data = new_fleet_data


class AsteroidFieldVisualizer(UIElement):
    """Cellular automaton-based asteroid field visualizer"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        density: float = 0.3,
        style: UIStyle = UIStyle.ASTEROID,
    ):
        super().__init__(x, y, width, height, style)
        self.density = density
        self.automaton_grid = self._initialize_grid()
        self.generation = 0
        self.asteroid_chars = ["·", "∙", "•", "◦", "○", "◌", "◍", "◎", "●", "@"]

    def _initialize_grid(self) -> List[List[float]]:
        """Initialize the asteroid field grid with random values"""
        grid = []
        for _ in range(self.height - 2):
            row = []
            for _ in range(self.width - 2):
                # Random value representing asteroid density/size
                value = random.random() if random.random() < self.density else 0
                row.append(value)
            grid.append(row)
        return grid

    def _evolve_grid(self):    # sourcery skip: low-code-quality
        """Evolve the asteroid field using cellular automaton rules"""
        new_grid = [[0 for _ in range(self.width - 2)] for _ in range(self.height - 2)]

        for y in range(self.height - 2):
            for x in range(self.width - 2):
                # Calculate the average of neighboring cells
                total = 0
                count = 0

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width - 2 and 0 <= ny < self.height - 2:
                            total += self.automaton_grid[ny][nx]
                            count += 1

                avg = total / count if count > 0 else 0

                # Evolution rules for asteroids (forming clusters)
                current = self.automaton_grid[y][x]
                if current > 0:
                    # Existing asteroids evolve based on neighbors
                    if avg < 0.2:  # Isolated asteroids shrink
                        new_grid[y][x] = max(0, current - 0.1)
                    elif avg > 0.6:  # Dense clusters grow
                        new_grid[y][x] = min(1.0, current + 0.05)
                    else:  # Stable clusters stay similar
                        new_grid[y][x] = current + (random.random() - 0.5) * 0.1
                        new_grid[y][x] = max(0, min(1.0, new_grid[y][x]))
                elif avg > 0.5 and random.random() < 0.1:
                    new_grid[y][x] = random.random() * 0.3
                else:
                    new_grid[y][x] = 0

        self.automaton_grid = new_grid
        self.generation += 1

    def draw(self, stdscr):
        """Draw the asteroid field visualization"""
        super().draw(stdscr)

        # Draw title
        with contextlib.suppress(curses.error):
            stdscr.addstr(
                self.y + 1,
                self.x + 2,
                f"ASTEROID FIELD - GEN {self.generation}",
                curses.A_BOLD,
            )
        # Draw the asteroid grid
        for y in range(self.height - 2):
            for x in range(self.width - 2):
                value = self.automaton_grid[y][x]
                if value > 0:
                    # Convert value to asteroid character
                    char_index = min(
                        len(self.asteroid_chars) - 1,
                        int(value * len(self.asteroid_chars)),
                    )
                    char = self.asteroid_chars[char_index]

                    with contextlib.suppress(curses.error):
                        stdscr.addstr(self.y + y + 2, self.x + x + 1, char)
        # Occasionally evolve the grid
        if random.random() < 0.1:
            self._evolve_grid()


class SymbioteEvolutionMonitor(UIElement):
    """Monitor for the symbiote evolution process"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        style: UIStyle = UIStyle.SYMBIOTIC,
    ):
        super().__init__(x, y, width, height, style)
        self.evolution_stage = 0
        self.fitness_values = [0.0] * 20  # Track last 20 fitness values
        self.mutation_rate = 0.2
        self.generation = 0
        self.symbiote_patterns = self._generate_patterns()

    def _generate_patterns(self) -> List[List[str]]:
        """Generate evolving patterns for symbiotes at different stages"""
        return [
            # Stage 0: Simple forms
            ["∙", "·", "∙", "·"],
            # Stage 1: Growing
            ["∘", "○", "∘", "○"],
            # Stage 2: Basic structures
            ["◌", "◍", "◎", "●"],
            # Stage 3: Complex structures
            ["◐", "◑", "◒", "◓"],
            # Stage 4: Advanced forms
            ["◧", "◨", "◩", "◪"],
            # Stage 5: Final forms
            ["⬡", "⬢", "⬣", "⏣"],
        ]

    def draw(self, stdscr):
        """Draw the symbiote evolution monitor"""
        super().draw(stdscr)

        # Draw title
        with contextlib.suppress(curses.error):
            stdscr.addstr(
                self.y + 1,
                self.x + 2,
                f"SYMBIOTE EVOLUTION - GEN {self.generation}",
                curses.A_BOLD,
            )
        # Draw evolution stage and pattern
        stage_text = f"Stage: {self.evolution_stage}/5"
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 3, self.x + 2, stage_text)

            pattern = self.symbiote_patterns[min(5, self.evolution_stage)]
            pattern_index = self.generation % len(pattern)

            # Draw current pattern (larger)
            current_pattern = pattern[pattern_index] * 3
            stdscr.addstr(self.y + 3, self.x + 2 + len(stage_text) + 2, current_pattern)
        # Draw fitness trend
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 5, self.x + 2, "Fitness trend:")

            # Calculate trend line using simple bars
            for i, val in enumerate(self.fitness_values):
                bar_height = int(val * 5)  # Scale to 0-5
                for j in range(5):
                    char = "▓" if j < bar_height else "░"
                    if self.x + 2 + i < self.x + self.width - 1:
                        stdscr.addstr(self.y + 10 - j, self.x + 2 + i, char)
        # Draw mutation rate
        with contextlib.suppress(curses.error):
            stdscr.addstr(
                self.y + 12, self.x + 2, f"Mutation rate: {self.mutation_rate:.2f}"
            )
        # Draw generation count
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 14, self.x + 2, f"Generation: {self.generation}")

    def update_evolution(self, fitness: float):
        """Update evolution state with new fitness value"""
        # Update fitness history
        self.fitness_values.pop(0)
        self.fitness_values.append(fitness)

        # Update generation
        self.generation += 1

        # Potentially evolve to next stage
        if (
            self.generation > 10
            and sum(self.fitness_values[-5:]) > 0.8 * 5
            and self.evolution_stage < 5
        ):
            self.evolution_stage += 1

        # Adapt mutation rate based on fitness trend
        if len(self.fitness_values) >= 5:
            recent_avg = sum(self.fitness_values[-5:]) / 5
            older_avg = sum(self.fitness_values[-10:-5]) / 5

            if recent_avg > older_avg:
                # Reducing mutation as we're improving
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            else:
                # Increasing mutation as we're plateauing
                self.mutation_rate = min(0.5, self.mutation_rate * 1.1)


class MiningStatus(UIElement):
    """Display for resource mining operations and statistics"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        resources: Dict[str, float],
        style: UIStyle = UIStyle.MECHANICAL,
    ):
        super().__init__(x, y, width, height, style)
        self.resources = resources
        self.extraction_rate = {k: 0.0 for k in resources}
        self.efficiency = 0.85
        self.symbols = {
            "Iron": "Fe",
            "Copper": "Cu",
            "Gold": "Au",
            "Titanium": "Ti",
            "Quantum": "Qm",
        }

    def draw(self, stdscr):
        """Draw the mining status display"""
        super().draw(stdscr)

        # Draw title
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 1, self.x + 2, "MINING OPERATIONS", curses.A_BOLD)
        # Draw efficiency meter
        with contextlib.suppress(curses.error):
            stdscr.addstr(self.y + 3, self.x + 2, f"Efficiency: {self.efficiency:.0%}")
            meter_width = self.width - 6
            filled_width = int(meter_width * self.efficiency)

            meter = "▓" * filled_width + "░" * (meter_width - filled_width)
            stdscr.addstr(self.y + 4, self.x + 2, meter)
        # Draw resources and extraction rates
        y_offset = 6
        for resource, amount in self.resources.items():
            symbol = self.symbols.get(resource, resource[:2])

            with contextlib.suppress(curses.error):
                # Resource name and amount
                stdscr.addstr(self.y + y_offset, self.x + 2, f"{resource} ({symbol}):")
                stdscr.addstr(self.y + y_offset, self.x + 16, f"{amount:.1f}")

                # Extraction rate with arrow indicator
                rate = self.extraction_rate[resource]
                rate_str = f"{rate:+.2f}/s"
                direction = "↑" if rate > 0 else "↓" if rate < 0 else "→"

                if rate > 0:
                    stdscr.addstr(
                        self.y + y_offset,
                        self.x + 24,
                        f"{direction} {rate_str}",
                        curses.A_BOLD,
                    )
                else:
                    stdscr.addstr(
                        self.y + y_offset, self.x + 24, f"{direction} {rate_str}"
                    )
            y_offset += 2

    def update_resources(
        self,
        resources: Dict[str, float],
        extraction_rate: Dict[str, float],
        efficiency: float,
    ):
        """Update resource information"""
        self.resources = resources
        self.extraction_rate = extraction_rate
        self.efficiency = efficiency


class SpaceMuckMainUI:
    """Main UI manager for Space Muck game"""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        self.ui_elements = {}
        self.active_element = None
        self.running = True

        # Initialize colors if terminal supports it
        self._init_colors()

        # Create UI elements
        self._init_ui_elements()

    def _init_colors(self):
        """Initialize color pairs for the UI"""
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()

            # Define color pairs for different UI elements
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Symbiotic elements
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Asteroid elements
            curses.init_pair(3, curses.COLOR_CYAN, -1)  # Mechanical elements
            curses.init_pair(4, curses.COLOR_MAGENTA, -1)  # Quantum elements
            curses.init_pair(5, curses.COLOR_BLUE, -1)  # Fleet elements

    def _init_ui_elements(self):
        """Create and initialize all UI elements"""
        # Create main menu
        main_menu = Menu(
            2,
            2,
            30,
            20,
            "SPACE MUCK",
            [
                "Start Expedition",
                "Fleet Management",
                "Research",
                "Trading",
                "Settings",
                "Exit",
            ],
            UIStyle.MECHANICAL,
        )
        self.ui_elements["main_menu"] = main_menu
        main_menu.active = True
        self.active_element = "main_menu"

        # Fleet display
        fleet_data = {"miner": 3, "fighter": 2, "capital": 1}
        fleet_display = FleetDisplay(34, 2, 44, 12, fleet_data, UIStyle.FLEET)
        self.ui_elements["fleet_display"] = fleet_display

        # Asteroid field visualizer
        asteroid_field = AsteroidFieldVisualizer(34, 15, 44, 20, 0.3, UIStyle.ASTEROID)
        self.ui_elements["asteroid_field"] = asteroid_field

        # Symbiote evolution monitor
        symbiote_monitor = SymbioteEvolutionMonitor(2, 23, 30, 25, UIStyle.SYMBIOTIC)
        self.ui_elements["symbiote_monitor"] = symbiote_monitor

        # Mining status
        resources = {
            "Iron": 256.0,
            "Copper": 128.5,
            "Gold": 32.1,
            "Titanium": 64.7,
            "Quantum": 15.3,
        }
        mining_status = MiningStatus(80, 2, 38, 20, resources, UIStyle.MECHANICAL)
        self.ui_elements["mining_status"] = mining_status

        # Set some initial extraction rates for demo
        mining_status.extraction_rate = {
            "Iron": 1.2,
            "Copper": 0.8,
            "Gold": 0.15,
            "Titanium": -0.3,
            "Quantum": 0.05,
        }

    def draw(self):
        """Draw all UI elements"""
        self.stdscr.clear()

        # Draw fancy header
        self._draw_header()

        # Draw all UI elements
        for name, element in self.ui_elements.items():
            element.draw(self.stdscr)

        self.stdscr.refresh()

    def _draw_header(self):
        """Draw the game header with ASCII art"""
        header_text = [
            
            "╭─────────────────────────────────────────────────────╮",
            "│                                                     │",
            "│ ░░      ░░░       ░░░░      ░░░░      ░░░        ░░ │",
            "│ ▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒ │",
            "│ ▓▓      ▓▓▓       ▓▓▓  ▓▓▓▓  ▓▓  ▓▓▓▓▓▓▓▓       ▓▓▓`│",
            "│ ███████  ██  ████████        ██  ████  ██  ████████ │",
            "│ ██      ███  ████████  ████  ███      ███        ██ │",
            "│ ███████████████████████████████████████████████████ |",
            "|                                                     |",
            "|     ░░░  ░░░░  ░░  ░░░░  ░░░      ░░░  ░░░░  ░      |",
            "|     ▒▒▒   ▒▒   ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒  ▒▒      |",
            "|     ▓▓▓        ▓▓  ▓▓▓▓  ▓▓  ▓▓▓▓▓▓▓▓     ▓▓▓▓      |",
            "|     ███  █  █  ██  ████  ██  ████  ██  ███  ██      |",
            "|     ███  ████  ███      ████      ███  ████  █      |",
            "|     ██████████████████████████████████████████      |",
            "╰─────────────────────────────────────────────────────╯",
        ]

        for i, line in enumerate(header_text):
            with contextlib.suppress(curses.error):
                self.stdscr.addstr(i, 0, line)

    def handle_input(self):
        """Handle user input"""
        key = self.stdscr.getch()

        # ESC key to exit
        if key == 27:
            self.running = False
            return

        # Pass input to active element
        if self.active_element == "main_menu":
            result = self.ui_elements["main_menu"].handle_input(key)
            if result == "Exit":
                self.running = False
            elif result == "Fleet Management":
                # Demo: Update fleet when selecting this option
                new_fleet = {"miner": 4, "fighter": 3, "capital": 1}
                self.ui_elements["fleet_display"].update_fleet(new_fleet)
            elif result == "Research":
                # Demo: Update symbiote evolution when selecting this option
                self.ui_elements["symbiote_monitor"].update_evolution(random.random())
            elif result == "Trading":
                # Demo: Update mining status when selecting this option
                current_resources = self.ui_elements["mining_status"].resources
                new_resources = {
                    k: v + random.uniform(-10, 20) for k, v in current_resources.items()
                }
                new_rates = {
                    k: random.uniform(-0.5, 1.5) for k in current_resources.keys()
                }
                self.ui_elements["mining_status"].update_resources(
                    new_resources, new_rates, random.uniform(0.5, 0.95)
                )

    def main_loop(self):
        """Main game loop"""
        # Perform initial animations
        for name, element in self.ui_elements.items():
            # Use different animation styles for different elements
            if name == "main_menu":
                element.animate(self.stdscr, AnimationStyle.WARP)
            elif name == "fleet_display":
                element.animate(self.stdscr, AnimationStyle.FRACTAL)
            elif name == "asteroid_field":
                element.animate(self.stdscr, AnimationStyle.CELLULAR)
            elif name == "symbiote_monitor":
                element.animate(self.stdscr, AnimationStyle.QUANTUM_FLUX)
            elif name == "mining_status":
                element.animate(self.stdscr, AnimationStyle.MINERAL_GROWTH)

        # Main game loop
        while self.running:
            self.draw()
            self.handle_input()

            # Update asteroid field occasionally
            if random.random() < 0.05:
                self.ui_elements["asteroid_field"]._evolve_grid()

            # Animate fleet display on each frame
            self.ui_elements["fleet_display"].animation_frame += 1

            # Sleep briefly to control frame rate
            time.sleep(0.05)


def run_space_muck_ui():
    """Initialize and run the Space Muck UI"""

    def main(stdscr):
        # Set up terminal
        curses.curs_set(0)  # Hide cursor
        stdscr.timeout(100)  # Non-blocking input with 100ms timeout

        # Create and run UI
        ui = SpaceMuckMainUI(stdscr)
        ui.main_loop()

    curses.wrapper(main)


if __name__ == "__main__":
    run_space_muck_ui()
