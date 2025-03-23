# Standard library imports
import logging
import math
import random
from typing import Tuple, Optional, List, Dict

import pygame

# Import all standard UI colors for consistency across components
from config import COLOR_TEXT

# Local application imports
from ui.ui_base.ui_style import UIStyle
from ui.ui_helpers.animation_helper import AnimationStyle

# Third-party library imports


class RenderHelper:
    """Helper class for standardized rendering across UI components.

    This class provides utility methods for rendering UI elements with consistent
    styling, animation effects, and error handling. It centralizes common rendering
    code to reduce duplication and ensure a unified visual approach.
    """

    @staticmethod
    def draw_char(
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        char: str,
        color: Tuple[int, int, int] = COLOR_TEXT,
        char_coords: bool = True,
    ) -> None:
        """Draw a single character at the specified position.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X position (in character coordinates if char_coords is True, else pixels)
            y: Y position (in character coordinates if char_coords is True, else pixels)
            char: Character to draw
            color: RGB color tuple for the character
            char_coords: If True, x and y are in character coordinates, else pixels
        """
        try:
            char_surf = font.render(char, True, color)
            if char_coords:
                char_width = font.size("X")[0]
                char_height = font.get_height()
                pixel_x = x * char_width
                pixel_y = y * char_height
            else:
                pixel_x = x
                pixel_y = y

            surface.blit(char_surf, (pixel_x, pixel_y))
        except Exception as e:
            logging.error(f"Error drawing character: {e}")

    @staticmethod
    def draw_text(
        surface: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int] = COLOR_TEXT,
        char_coords: bool = True,
    ) -> None:
        """Draw a text string at the specified position.

        Args:
            surface: Surface to draw on
            font: Font to use for rendering
            x: X position (in character coordinates if char_coords is True, else pixels)
            y: Y position (in character coordinates if char_coords is True, else pixels)
            text: Text to draw
            color: RGB color tuple for the text
            char_coords: If True, x and y are in character coordinates, else pixels
        """
        try:
            text_surf = font.render(text, True, color)
            if char_coords:
                char_width = font.size("X")[0]
                char_height = font.get_height()
                pixel_x = x * char_width
                pixel_y = y * char_height
            else:
                pixel_x = x
                pixel_y = y

            surface.blit(text_surf, (pixel_x, pixel_y))
        except Exception as e:
            logging.error(f"Error drawing text: {e}")

    @staticmethod
    def get_style_color(
        style: UIStyle, base_color: Tuple[int, int, int] = COLOR_TEXT
    ) -> Tuple[int, int, int]:
        """Get the appropriate color for a given UI style.

        Args:
            style: The UI style to get the color for
            base_color: The base color to adjust based on style

        Returns:
            RGB color tuple adjusted for the specified style
        """
        try:
            r, g, b = base_color

            if style == UIStyle.QUANTUM:
                # Quantum style: blue-shifted
                return (int(r * 0.8), int(g * 0.9), min(255, int(b * 1.2)))
            elif style == UIStyle.SYMBIOTIC:
                # Symbiotic style: green-shifted
                return (int(r * 0.8), min(255, int(g * 1.2)), int(b * 0.8))
            elif style == UIStyle.MECHANICAL:
                # Mechanical style: slight blue tint
                return (r, g, min(255, int(b * 1.1)))
            elif style == UIStyle.ASTEROID:
                # Asteroid style: orange/red tint
                return (min(255, int(r * 1.2)), int(g * 0.9), int(b * 0.8))
            elif style == UIStyle.FLEET:
                # Fleet style: cyan tint
                return (int(r * 0.8), min(255, int(g * 1.1)), min(255, int(b * 1.1)))
            else:
                return base_color
        except Exception as e:
            logging.error(f"Error getting style color: {e}")
            return base_color

    @staticmethod
    def apply_animation_effect(
        color: Tuple[int, int, int],
        style: UIStyle,
        progress: float,
        phase: float = 0.0,
        animation_type: Optional[AnimationStyle] = None,
    ) -> Tuple[int, int, int]:
        """Apply style-specific animation effects to a color.

        Args:
            color: Base RGB color tuple
            style: UI style to apply effects for
            progress: Animation progress from 0.0 to 1.0
            phase: Animation phase for cyclic effects
            animation_type: Optional specific animation type to override style-based animation

        Returns:
            Modified RGB color tuple with animation effects applied
        """
        try:
            r, g, b = color
            
            # Handle specific animation type if provided
            if animation_type is not None:
                return RenderHelper._apply_specific_animation(
                    color, animation_type, progress, phase
                )

            # Apply style-specific color effects
            if style == UIStyle.QUANTUM:
                # Quantum style: color shifts with subtle pulsing
                r_adj = int(min(255, r * (0.8 + 0.4 * math.sin(progress * math.pi))))
                b_adj = int(min(255, b * (0.8 + 0.4 * math.cos(progress * math.pi))))
                return (r_adj, g, b_adj)
            elif style == UIStyle.SYMBIOTIC:
                # Symbiotic style: gradual green enhancement
                g_boost = int(min(255, g * (1.0 + 0.3 * progress)))
                return (r, g_boost, b)
            elif style == UIStyle.ASTEROID:
                # Asteroid style: warming effect (red channel boost)
                r_boost = int(min(255, r * (1.0 + 0.3 * progress)))
                return (r_boost, g, b)
            elif style == UIStyle.FLEET:
                # Fleet style: blue/cyan pulsing
                b_boost = int(min(255, b * (1.0 + 0.2 * math.sin(phase))))
                g_boost = int(min(255, g * (1.0 + 0.1 * math.sin(phase))))
                return (r, g_boost, b_boost)
            elif style == UIStyle.MECHANICAL or style is None:
                # Mechanical style or default: subtle brightening of all channels
                factor = 1.0 + 0.2 * progress
                return (
                    int(min(255, r * factor)),
                    int(min(255, g * factor)),
                    int(min(255, b * factor)),
                )
            return color
        except Exception as e:
            logging.error(f"Error applying animation effect: {e}")
            return color
            
    @staticmethod
    def _apply_specific_animation(
        color: Tuple[int, int, int],
        animation_type: AnimationStyle,
        progress: float,
        phase: float,
    ) -> Tuple[int, int, int]:
        """Apply specific animation effect based on animation type
        
        Args:
            color: Base RGB color tuple
            animation_type: Animation type to apply
            progress: Animation progress (0.0-1.0)
            phase: Animation phase for cyclic effects
            
        Returns:
            Modified RGB color tuple with animation applied
        """
        r, g, b = color
        
        # Apply different effects based on animation type
        if animation_type == AnimationStyle.PULSE:
            # Pulsing intensity effect
            pulse = 0.7 + 0.5 * math.sin(phase * 6.0)
            return (
                int(min(255, r * pulse)), 
                int(min(255, g * pulse)), 
                int(min(255, b * pulse))
            )
            
        elif animation_type == AnimationStyle.GLITCH:
            # Random color channel shifts for glitch effect
            if random.random() < 0.2:  # Only apply to some frames
                glitch_value = random.choice([0.2, 0.5, 0.8, 1.2, 1.5])
                channel = random.randint(0, 2)
                if channel == 0:
                    r = int(min(255, r * glitch_value))
                elif channel == 1:
                    g = int(min(255, g * glitch_value))
                else:
                    b = int(min(255, b * glitch_value))
            return (r, g, b)
            
        elif animation_type == AnimationStyle.DATA_STREAM:
            # Subtle color variation based on phase
            blue_boost = int(min(255, b * (1.1 + 0.2 * math.sin(phase * 4.0))))
            return (r, g, blue_boost)
            
        elif animation_type == AnimationStyle.PARTICLE:
            # Particle-like fading based on progress
            fade = 1.0 - progress  # Fade out as progress increases
            return (
                int(r * fade),
                int(g * fade),
                int(b * fade)
            )
            
        elif animation_type == AnimationStyle.SPARKLE:
            # Sparkling effect with random brightness
            if random.random() < 0.15:  # Only sparkle sometimes
                sparkle = 1.0 + random.uniform(0.2, 0.8)
                return (
                    int(min(255, r * sparkle)),
                    int(min(255, g * sparkle)),
                    int(min(255, b * sparkle))
                )
            return color
            
        # Return original color for unhandled types
        return color
        
    @staticmethod
    def get_character_set(
        density_level: int = 1, 
        style: UIStyle = UIStyle.MECHANICAL
    ) -> List[str]:
        """Get a set of ASCII characters for a given density level and style
        
        Args:
            density_level: Density level (1-5, 1=sparse, 5=dense)
            style: UI style to determine character set
            
        Returns:
            List of characters for the specified density and style
        """
        # Base character sets by density level
        density_sets = {
            1: ['.', '·', ' '],                             # Very sparse
            2: ['.', '·', ':', '·', ' '],                   # Sparse
            3: ['.', ':', ';', '•', '·', '*', ' '],         # Medium
            4: ['.', ':', ';', '*', '+', '=', '•', '%'],    # Dense
            5: ['.', ':', ';', '*', '+', '=', '#', '%', '@'] # Very dense
        }
        
        # Get base set and modify by style
        level = max(1, min(5, density_level))
        base_set = density_sets.get(level, density_sets[3])
        
        # Stylized character sets
        if style == UIStyle.SYMBIOTIC:
            if level >= 3:
                return ['.', '~', '•', '*', '∞', '○', '●', '◌', '◍']
            return ['.', '~', '•', '○', '◌']
        elif style == UIStyle.MECHANICAL:
            if level >= 3:
                return ['.', ':', '=', '+', '#', '/', '\\', '{', '}']
            return ['.', ':', '+', '/']
        elif style == UIStyle.ASTEROID:
            if level >= 3:  
                return ['.', '°', '·', '*', '#', '○', '@', '&', '%']
            return ['.', '°', '*', '○']
        elif style == UIStyle.QUANTUM:
            if level >= 3:
                return ['.', '·', ':', '•', 'ᚉ', '⧿', '⊛', '⊗', 'Φ']
            return ['.', '·', '•', '⊛']
        elif style == UIStyle.FLEET:
            if level >= 3:
                return ['.', '·', '>', '<', '^', 'v', '|', '-', '+']
            return ['.', '>', '<', '^']
        
        # Default
        return base_set
        
    @staticmethod
    def get_direction_characters(style: UIStyle = UIStyle.MECHANICAL) -> Dict[str, str]:
        """Get directional characters for a given UI style
        
        Args:
            style: UI style to determine character set
            
        Returns:
            Dictionary mapping directions to characters
        """
        # Define character sets by style
        if style == UIStyle.SYMBIOTIC:
            return {
                'up': '⋀',
                'down': '⋁',
                'left': '⊲',
                'right': '⊳',
                'upleft': '⌜',
                'upright': '⌝',
                'downleft': '⌞',
                'downright': '⌟'
            }
        elif style == UIStyle.MECHANICAL:
            return {
                'up': '^',
                'down': 'v',
                'left': '<',
                'right': '>',
                'upleft': '/',
                'upright': '\\',
                'downleft': '\\',
                'downright': '/'
            }
        elif style == UIStyle.ASTEROID:
            return {
                'up': '↑',
                'down': '↓',
                'left': '←',
                'right': '→',
                'upleft': '↖',
                'upright': '↗',
                'downleft': '↙',
                'downright': '↘'
            }
        elif style == UIStyle.QUANTUM:
            return {
                'up': '△',
                'down': '▽',
                'left': '◁',
                'right': '▷',
                'upleft': '⋰',
                'upright': '⋱',
                'downleft': '⋯',
                'downright': '⋮'
            }
        elif style == UIStyle.FLEET:
            return {
                'up': '^',
                'down': 'v',
                'left': '<',
                'right': '>',
                'upleft': '{',
                'upright': '}',
                'downleft': '(',
                'downright': ')'
            }
        
        # Default (MECHANICAL)
        return {
            'up': '^',
            'down': 'v',
            'left': '<',
            'right': '>',
            'upleft': '/',
            'upright': '\\',
            'downleft': '\\',
            'downright': '/'
        }
