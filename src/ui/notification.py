"""
NotificationManager class: Handles on-screen notifications and control tooltips.

This module manages game notifications, alerts, and control help displayed to the player.
It provides a scrollable notification panel and customizable tooltip system.
"""

import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import pygame

from config import WINDOW_HEIGHT, COLOR_TEXT
from .draw_utils import draw_text, draw_panel


class NotificationManager:
    """
    Displays on-screen notifications in a scrollable side panel.
    Each notification is a list: [text, duration, color, timestamp, importance]
    """

    def __init__(self) -> None:
        """Initialize the notification manager with default settings."""
        # Notification storage
        self.notifications: List[Dict[str, Any]] = []
        self.max_stored_notifications = 100  # Maximum number of notifications to store
        self.max_visible_notifications = (
            10  # Maximum number of notifications shown at once
        )

        # Panel settings
        self.scroll_offset = 0
        self.panel_width = 300
        self.panel_height = WINDOW_HEIGHT - 40
        self.panel_x = 10
        self.panel_y = 20
        self.panel_alpha = 200  # Transparency level (0-255)

        # Display settings
        self.fade_timer = 0
        self.show_full_panel = False  # Toggle between compact and expanded view
        self.compact_notification_count = (
            3  # Number of notifications to show when compact
        )
        self.notification_height = 25  # Height of each notification in pixels
        self.fade_speed = 0.5  # Speed at which notifications fade out
        self.last_notification_time = 0  # Track when the last notification was added

        # Filter system
        self.category_filters = {
            "all": True,
            "system": True,
            "mining": True,
            "race": True,
            "event": True,
            "upgrade": True,
        }
        self.active_filter = "all"

        # Control tooltips
        self.tooltips = [
            {"key": "S", "description": "Open Shop"},
            {"key": "R", "description": "Seed Asteroids"},
            {"key": "SPACE", "description": "Mine"},
            {"key": "↑↓←→", "description": "Move Ship"},
            {"key": "TAB", "description": "Switch Category (in Shop)"},
            {"key": "ESC", "description": "Close Shop"},
            {"key": "N", "description": "Toggle Notification Panel"},
            {"key": "+/-", "description": "Zoom In/Out"},
            {"key": "A", "description": "Toggle Auto-Mine"},
            {"key": "M", "description": "Toggle Minimap"},
            {"key": "F", "description": "Feed Symbiotes"},
        ]

        # Recent notification highlight
        self.highlight_time = 2.0  # How long to highlight new notifications (seconds)

    def add(
        self,
        text: str,
        duration: int = 300,
        color: Tuple[int, int, int] = COLOR_TEXT,
        category: str = "system",
        importance: int = 1,
    ) -> None:
        """
        Add a new notification to the list.

        Args:
            text: Notification message text
            duration: How long to display (in frames)
            color: RGB color tuple
            category: Category for filtering ("system", "mining", "race", "event", "upgrade")
            importance: Priority level (1-3, higher is more important)
        """
        # Create new notification entry
        notification = {
            "text": text,
            "duration": duration,
            "color": color,
            "category": category,
            "importance": importance,
            "timestamp": time.time(),
            "read": False,
            "id": len(self.notifications),
        }

        # Add to the list
        self.notifications.insert(0, notification)

        # Update notification time
        self.last_notification_time = time.time()

        # Prune old notifications if we exceed the limit
        if len(self.notifications) > self.max_stored_notifications:
            self.notifications = self.notifications[: self.max_stored_notifications]

        # Reset scroll if panel is showing
        if self.show_full_panel:
            self.scroll_offset = 0

        # Log to console as well
        logging.info(f"[{category.upper()}] {text}")

        # Show the panel briefly
        self.fade_timer = 180

    def update(self) -> None:
        """Update notification states and handle fading."""
        # Update fade timer
        if self.fade_timer > 0:
            self.fade_timer -= 1

        # Mark notifications as read after they've been visible for a while
        current_time = time.time()
        for notification in self.notifications:
            if (
                not notification["read"]
                and current_time - notification["timestamp"] > self.highlight_time
            ):
                notification["read"] = True

        # Remove expired notifications
        self.notifications = [n for n in self.notifications if n["duration"] > 0]

        # Decrease duration of visible notifications
        for notification in self.notifications:
            notification["duration"] -= 1

    def get_filtered_notifications(self) -> List[Dict[str, Any]]:
        """Get notifications filtered by the current category filter."""
        if self.active_filter == "all":
            return self.notifications
        return [n for n in self.notifications if n["category"] == self.active_filter]

    def toggle_filter(self, category: str) -> None:
        """Toggle a category filter on/off."""
        if category in self.category_filters:
            self.category_filters[category] = not self.category_filters[category]

        # Update active filter
        if category == "all" or self.category_filters[category]:
            self.active_filter = category
        else:
            # If we disabled the current filter, switch to "all"
            self.active_filter = "all"

        # Reset scroll position
        self.scroll_offset = 0

    def draw_tooltips(self, surface: pygame.Surface, x: int, y: int) -> None:
        """
        Draw control tooltips at the specified position.

        Args:
            surface: Pygame surface to draw on
            x: X position
            y: Y position
        """
        # Draw background
        tooltip_height = len(self.tooltips) * 20 + 10
        tooltip_width = 250

        # Create tooltip background
        pygame.draw.rect(
            surface,
            (20, 20, 30, 180),  # Semi-transparent background
            (x, y, tooltip_width, tooltip_height),
            border_radius=5,
        )

        pygame.draw.rect(
            surface,
            (100, 100, 140, 200),  # Border color
            (x, y, tooltip_width, tooltip_height),
            2,
            border_radius=5,
        )

        # Draw header
        draw_text(surface, "Controls:", x + 10, y + 5, 18, COLOR_TEXT)

        # Draw key in a box
        key_width = 30
        # Draw key bindings
        for i, tooltip in enumerate(self.tooltips):
            key_x = x + 10
            key_y = y + 30 + i * 20

            pygame.draw.rect(
                surface, (50, 50, 70), (key_x, key_y, key_width, 18), border_radius=3
            )

            # Draw key text (centered in box)
            draw_text(
                surface,
                tooltip["key"],
                key_x + key_width // 2,
                key_y + 2,
                14,
                COLOR_TEXT,
                align="center",
            )

            # Draw description
            draw_text(
                surface,
                tooltip["description"],
                key_x + key_width + 10,
                key_y + 2,
                14,
                COLOR_TEXT,
            )

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the notification panel and current notifications.

        Args:
            surface: Pygame surface to draw on
        """
        # Always show at least the most recent notifications (compact mode)
        if not self.show_full_panel and self.fade_timer <= 0:
            self._draw_compact_notifications(surface)
            return

        # Draw full notification panel
        self._draw_full_panel(surface)

    def _draw_compact_notifications(self, surface: pygame.Surface) -> None:
        """Draw compact notification panel showing only recent notifications."""
        filtered_notifications = self.get_filtered_notifications()
        visible_count = min(
            self.compact_notification_count, len(filtered_notifications)
        )

        if visible_count == 0:
            return

        # Draw a small transparent panel for recent notifications
        compact_height = visible_count * self.notification_height + 10

        # Calculate alpha based on time since last notification
        time_since_last = time.time() - self.last_notification_time
        alpha = max(100, min(200, 200 - int(time_since_last * 30)))

        # Draw small panel
        pygame.draw.rect(
            surface,
            (30, 30, 40, alpha),
            (self.panel_x, self.panel_y, self.panel_width, compact_height),
            border_radius=5,
        )

        # Draw each notification
        for i in range(visible_count):
            self._draw_compact_notification(
                surface, filtered_notifications[i], i, alpha
            )

        # Draw "more" indicator if there are additional notifications
        if len(filtered_notifications) > visible_count:
            self._draw_more_indicator(
                surface, filtered_notifications, visible_count, compact_height, alpha
            )

    def _draw_compact_notification(
        self,
        surface: pygame.Surface,
        notification: Dict[str, Any],
        index: int,
        alpha: int,
    ) -> None:
        """Draw a single notification in compact mode."""
        text = notification["text"]
        color = notification["color"]

        # Highlight new notifications
        notification_age = time.time() - notification["timestamp"]
        if notification_age < self.highlight_time:
            # Draw highlight background
            pygame.draw.rect(
                surface,
                (60, 60, 80, alpha),
                (
                    self.panel_x + 5,
                    self.panel_y + 5 + index * self.notification_height,
                    self.panel_width - 10,
                    self.notification_height - 2,
                ),
                border_radius=3,
            )

        # Draw notification text
        draw_text(
            surface,
            text,
            self.panel_x + 10,
            self.panel_y + 5 + index * self.notification_height + 4,
            14,
            color,
            alpha=alpha,
        )

    def _draw_more_indicator(
        self,
        surface: pygame.Surface,
        notifications: List[Dict[str, Any]],
        visible_count: int,
        compact_height: int,
        alpha: int,
    ) -> None:
        """Draw indicator showing there are more notifications available."""
        draw_text(
            surface,
            f"+ {len(notifications) - visible_count} more (Press N)",
            self.panel_x + self.panel_width - 10,
            self.panel_y + compact_height - 15,
            12,
            COLOR_TEXT,
            align="right",
            alpha=alpha,
        )

    def _draw_full_panel(self, surface: pygame.Surface) -> None:
        """Draw the full notification panel with all notifications."""
        # Calculate panel alpha
        alpha = self.panel_alpha
        if not self.show_full_panel:
            # Fade in/out effect
            alpha = int((self.fade_timer / 180) * self.panel_alpha)

        # Draw panel background with header
        self._draw_panel_background(surface, alpha)

        # Draw category filters
        self._draw_category_filters(surface, alpha)

        # Draw notifications
        filtered_notifications = self.get_filtered_notifications()
        visible_count = min(self.max_visible_notifications, len(filtered_notifications))

        if visible_count > 0:
            self._draw_notification_content(filtered_notifications, surface, alpha)

    def _draw_panel_background(self, surface: pygame.Surface, alpha: int) -> None:
        """Draw the main notification panel background and header."""
        draw_panel(
            surface,
            pygame.Rect(
                self.panel_x, self.panel_y, self.panel_width, self.panel_height
            ),
            color=(30, 30, 40, alpha),
            border_color=(100, 100, 140, alpha),
            border_width=2,
            header="Notifications",
            header_height=30,
            header_color=(40, 40, 60, alpha),
        )

    def _draw_category_filters(self, surface: pygame.Surface, alpha: int) -> None:
        """Draw the category filter buttons at the top of the panel."""
        filters = ["all", "system", "mining", "race", "event", "upgrade"]
        filter_width = self.panel_width / len(filters)

        for i, filter_name in enumerate(filters):
            filter_x = self.panel_x + i * filter_width
            filter_rect = pygame.Rect(filter_x, self.panel_y + 30, filter_width, 20)

            # Selected filter has different background
            if self.active_filter == filter_name:
                pygame.draw.rect(surface, (60, 60, 80, alpha), filter_rect)

            # Draw filter name
            draw_text(
                surface,
                filter_name.capitalize(),
                filter_x + filter_width // 2,
                self.panel_y + 32,
                12,
                (COLOR_TEXT if self.category_filters[filter_name] else (100, 100, 100)),
                align="center",
                alpha=alpha,
            )

            # Draw separator
            if i < len(filters) - 1:
                pygame.draw.line(
                    surface,
                    (60, 60, 80, alpha),
                    (filter_x + filter_width, self.panel_y + 30),
                    (filter_x + filter_width, self.panel_y + 50),
                    1,
                )

    def _draw_notification_content(self, filtered_notifications, surface, alpha):
        content_y = self.panel_y + 55  # Below filters

        # Draw scrollbar if needed
        if len(filtered_notifications) > self.max_visible_notifications:
            self._draw_scrollbar(filtered_notifications, content_y, surface, alpha)
        # Adjust visible notifications based on scroll offset
        start_idx = self.scroll_offset
        end_idx = min(
            start_idx + self.max_visible_notifications,
            len(filtered_notifications),
        )
        visible_notifications = filtered_notifications[start_idx:end_idx]

        # Draw each visible notification
        for i, notification in enumerate(visible_notifications):
            text = notification["text"]
            color = notification["color"]
            category = notification["category"]
            importance = notification["importance"]
            read = notification["read"]

            # Background color based on importance and read status
            bg_color = (40, 40, 50, alpha)  # Default
            if not read:  # New notification
                bg_color = (60, 60, 80, alpha)
            elif importance >= 3:  # High importance
                bg_color = (80, 50, 50, alpha)
            elif importance == 2:  # Medium importance
                bg_color = (70, 70, 50, alpha)

            # Draw notification background
            notification_y = content_y + i * self.notification_height
            pygame.draw.rect(
                surface,
                bg_color,
                (
                    self.panel_x + 5,
                    notification_y,
                    self.panel_width - 20,
                    self.notification_height - 2,
                ),
                border_radius=3,
            )

            # Draw category indicator
            category_color = {
                "system": (100, 100, 255),
                "mining": (255, 215, 0),
                "race": (255, 100, 100),
                "event": (0, 255, 200),
                "upgrade": (200, 100, 255),
            }.get(category, (150, 150, 150))

            pygame.draw.rect(
                surface,
                category_color,
                (
                    self.panel_x + 5,
                    notification_y,
                    3,
                    self.notification_height - 2,
                ),
                border_radius=3,
            )

            # Draw notification text
            max_text_width = self.panel_width - 30  # Allow space for scrollbar
            draw_text(
                surface,
                text,
                self.panel_x + 12,  # Adjust for category indicator
                notification_y + 4,
                14,
                color,
                alpha=alpha,
                max_width=max_text_width,
            )

        # Draw empty state if no notifications
        if not filtered_notifications:
            draw_text(
                surface,
                "No notifications",
                self.panel_x + self.panel_width // 2,
                self.panel_y + self.panel_height // 2,
                16,
                (150, 150, 150, alpha),
                align="center",
            )

    def _draw_scrollbar(self, filtered_notifications, content_y, surface, alpha):
        scrollbar_height = self.panel_height - 65  # Account for header and filters
        visible_ratio = self.max_visible_notifications / len(filtered_notifications)
        thumb_height = max(30, scrollbar_height * visible_ratio)

        # Calculate thumb position
        max_scroll = len(filtered_notifications) - self.max_visible_notifications
        scroll_ratio = self.scroll_offset / max_scroll if max_scroll > 0 else 0
        thumb_y = content_y + scroll_ratio * (scrollbar_height - thumb_height)

        # Draw track
        scrollbar_x = self.panel_x + self.panel_width - 15
        pygame.draw.rect(
            surface,
            (40, 40, 50, alpha),
            (scrollbar_x, content_y, 10, scrollbar_height),
            border_radius=5,
        )

        # Draw thumb
        pygame.draw.rect(
            surface,
            (80, 80, 100, alpha),
            (scrollbar_x, thumb_y, 10, thumb_height),
            border_radius=5,
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle Pygame events related to the notification panel.

        Args:
            event: Pygame event object

        Returns:
            bool: True if event was handled, False otherwise
        """
        if event.type == pygame.KEYDOWN:
            return self._handle_key_event(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_mouse_event(event)
        return False

    def _handle_key_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard events for the notification panel."""
        # Toggle notification panel
        if event.key == pygame.K_n:
            self.show_full_panel = not self.show_full_panel
            self.fade_timer = 0 if self.show_full_panel else 180
            return True

        # Only process other keys if panel is visible
        if not self.show_full_panel:
            return False

        # Handle scrolling keys
        if self._handle_scroll_key(event):
            return True

        # Handle filter keys
        return bool(self._handle_filter_key(event))

    def _handle_scroll_key(self, event: pygame.event.Event) -> bool:
        """Handle keyboard scrolling for notifications."""
        filtered_notifications = self.get_filtered_notifications()
        max_scroll = max(
            0, len(filtered_notifications) - self.max_visible_notifications
        )

        # Simple scroll up/down
        if event.key == pygame.K_UP:
            self.scroll_offset = max(0, self.scroll_offset - 1)
            return True
        elif event.key == pygame.K_DOWN:
            self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
            return True

        # Page up/down
        elif event.key == pygame.K_PAGEUP:
            self.scroll_offset = max(
                0, self.scroll_offset - self.max_visible_notifications
            )
            return True
        elif event.key == pygame.K_PAGEDOWN:
            self.scroll_offset = min(
                max_scroll, self.scroll_offset + self.max_visible_notifications
            )
            return True

        # Home/end
        elif event.key == pygame.K_HOME:
            self.scroll_offset = 0
            return True
        elif event.key == pygame.K_END:
            self.scroll_offset = max_scroll
            return True

        return False

    def _handle_filter_key(self, event: pygame.event.Event) -> bool:
        """Handle number keys for toggling notification filters."""
        if event.key in [
            pygame.K_1,
            pygame.K_2,
            pygame.K_3,
            pygame.K_4,
            pygame.K_5,
            pygame.K_6,
        ]:
            filter_idx = event.key - pygame.K_1
            filters = ["all", "system", "mining", "race", "event", "upgrade"]
            if filter_idx < len(filters):
                self.toggle_filter(filters[filter_idx])
                return True
        return False

    def _handle_mouse_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events for the notification panel."""
        # Only process if panel is visible and click is within panel
        if not (
            self.show_full_panel
            and self.panel_x <= event.pos[0] <= self.panel_x + self.panel_width
        ):
            return False

        # Handle filter clicks
        if self.panel_y + 30 <= event.pos[1] <= self.panel_y + 50:
            return self._handle_filter_click(event)

        # Handle scrolling with mouse wheel
        if self.panel_y <= event.pos[1] <= self.panel_y + self.panel_height:
            return self._handle_mouse_scroll(event)

        return False

    def _handle_filter_click(self, event: pygame.event.Event) -> bool:
        """Handle clicks on filter buttons."""
        filters = [
            "all",
            "system",
            "mining",
            "race",
            "event",
            "upgrade",
        ]
        filter_width = self.panel_width / len(filters)
        filter_idx = int((event.pos[0] - self.panel_x) / filter_width)

        if 0 <= filter_idx < len(filters):
            self.toggle_filter(filters[filter_idx])
            return True
        return False

    def _handle_mouse_scroll(self, event: pygame.event.Event) -> bool:
        """Handle mouse wheel scrolling."""
        filtered_notifications = self.get_filtered_notifications()
        max_scroll = max(
            0,
            len(filtered_notifications) - self.max_visible_notifications,
        )

        if event.button == 4:  # Scroll up
            self.scroll_offset = max(0, self.scroll_offset - 1)
            return True
        elif event.button == 5:  # Scroll down
            self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
            return True

        return False

    def clear(self, category: Optional[str] = None) -> None:
        """
        Clear all notifications or only those in a specific category.

        Args:
            category: Optional category to clear, or None for all
        """
        if category:
            self.notifications = [
                n for n in self.notifications if n["category"] != category
            ]
        else:
            self.notifications = []

    def notify_event(self, type_str: str, message: str, importance: int = 1) -> None:
        """
        Create a notification for a game event with appropriate formatting.

        Args:
            type_str: Type of event ("discovery", "warning", "achievement", etc)
            message: Event message
            importance: Priority level (1-3)
        """
        if type_str == "discovery":
            self.add(
                f"DISCOVERY: {message}",
                color=(0, 255, 200),
                category="event",
                importance=2,
            )
        elif type_str == "warning":
            self.add(
                f"WARNING: {message}",
                color=(255, 100, 100),
                category="event",
                importance=3,
            )
        elif type_str == "achievement":
            self.add(
                f"ACHIEVEMENT: {message}",
                color=(255, 215, 0),
                category="event",
                importance=2,
            )
        elif type_str == "race":
            self.add(
                f"RACE EVENT: {message}",
                color=(200, 100, 255),
                category="race",
                importance=2,
            )
        elif type_str == "mining":
            self.add(message, color=(200, 200, 200), category="mining", importance=1)
        elif type_str == "upgrade":
            self.add(
                f"UPGRADE: {message}",
                color=(100, 255, 100),
                category="upgrade",
                importance=2,
            )
        else:
            self.add(message, category="system")

    def get_notification_count(self, category: Optional[str] = None) -> int:
        """
        Get the count of notifications, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            int: Number of notifications
        """
        if category:
            return len([n for n in self.notifications if n["category"] == category])
        return len(self.notifications)

    def get_unread_count(self) -> int:
        """
        Get the count of unread notifications.

        Returns:
            int: Number of unread notifications
        """
        return len([n for n in self.notifications if not n["read"]])
