"""
ui_component_enhancer.py

This module provides enhancements to UI components with animation and particle effects.
"""

# Standard library imports
import math
import random
from typing import Dict, Optional, Any, TypeVar

# Third-party library imports
import pygame

# Local application imports
# from ui.ui_base.ui_style import UIStyle  # Uncomment if needed in the future
from ui.ui_helpers.animation_helper import AnimationStyle
from ui.ui_helpers.animation_integrator import AnimationIntegrator
from ui.ui_base.ascii_ui import ASCIIBox, ASCIIButton, ASCIIPanel, ASCIIProgressBar
from ui.ui_element.ui_element import UIElement


T = TypeVar('T', bound=UIElement)


class UIComponentEnhancer:
    """Enhances UI components with animation and particle effects."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'UIComponentEnhancer':
        """Get or create the singleton instance.
        
        Returns:
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the UI component enhancer."""
        self.animation_integrator = AnimationIntegrator.get_instance()
        self.enhanced_elements: Dict[int, Dict[str, Any]] = {}
    
    def update(self, dt: float) -> None:
        """Update all enhanced UI components.
        
        Args:
            dt: Time delta in seconds
        """
        # Update animation integrator
        self.animation_integrator.update(dt)
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render all enhanced UI component effects.
        
        Args:
            surface: Surface to render on
            font: Font to use for rendering
        """
        # Render animation effects
        self.animation_integrator.render(surface, font)
    
    def enhance_element(self, 
                       element: UIElement, 
                       animation_type: Optional[AnimationStyle] = None,
                       duration: float = -1.0,  # -1 means infinite
                       intensity: float = 0.5,
                       **kwargs: Any) -> str:
        """Enhance a UI element with animation effects.
        
        Args:
            element: UI element to enhance
            animation_type: Type of animation to apply
            duration: Duration of animation in seconds (-1 for infinite)
            intensity: Animation intensity (0.0 to 1.0)
            **kwargs: Additional animation parameters
            
        Returns:
            ID of the enhancement
        """
        # Generate enhancement ID
        element_id = id(element)
        enhancement_id = f"enh_{element_id}"
        
        # Store enhancement data
        self.enhanced_elements[element_id] = {
            'element': element,
            'animation_type': animation_type,
            'duration': duration,
            'intensity': intensity,
            'params': kwargs
        }
        
        # If animation type is specified, add the animation
        if animation_type:
            rect = pygame.Rect(element.x, element.y, element.width, element.height)
            
            # Add the animation via the integrator
            anim_id = self.animation_integrator.add_animation(
                animation_type,
                duration if duration > 0 else 3600.0,  # Use long duration for "infinite"
                rect,
                intensity=intensity,
                **kwargs
            )
            
            # Store animation ID
            self.enhanced_elements[element_id]['anim_id'] = anim_id
            
            # Override draw method
            self._enhance_element_drawing(element, animation_type, intensity)
        
        return enhancement_id
    
    def _enhance_element_drawing(self, 
                               element: UIElement, 
                               animation_type: AnimationStyle,
                               intensity: float) -> None:
        """Enhance element drawing with animation effects.
        
        Args:
            element: UI element to enhance
            animation_type: Type of animation
            intensity: Animation intensity
        """
        # Store original draw method
        original_draw = element.draw
        
        # Create enhanced draw method
        def enhanced_draw(surface, font):
            # Call original draw first
            original_draw(surface, font)
            
            # Apply post-drawing animation effects if needed
            if animation_type in [AnimationStyle.SPARKLE, AnimationStyle.PARTICLE]:
                # These are handled by the particle system, no need for additional drawing
                pass
            
            elif animation_type == AnimationStyle.PULSE:
                # Pulsing border highlight effect
                self._draw_pulsing_border(element, surface, font, intensity)
            
            elif animation_type == AnimationStyle.GLITCH:
                # Occasional glitch effect on borders
                if random.random() < 0.05 * intensity:
                    self._draw_glitch_effect(element, surface, font, intensity)
            
            elif animation_type == AnimationStyle.DATA_STREAM:
                # Data stream effect on borders
                self._draw_data_effect(element, surface, font, intensity)
        
        # Replace the draw method
        element.draw = enhanced_draw
    
    @staticmethod
    def _draw_pulsing_border(element: UIElement, 
                            surface: pygame.Surface, 
                            font: pygame.font.Font,
                            intensity: float) -> None:
        """Draw a pulsing border effect.
        
        Args:
            element: UI element
            surface: Surface to draw on
            font: Font to use for rendering
            intensity: Effect intensity
        """
        # Calculate pulse value
        t = pygame.time.get_ticks() / 1000.0
        pulse_value = 0.5 + 0.5 * math.sin(t * 4.0) * intensity
        
        # Get element color
        color = (255, 255, 255)  # Default white
        if hasattr(element, 'border_color'):
            color = element.border_color
        
        # Adjust color by pulse value
        pulsed_color = (
            int(min(255, color[0] * (1.0 + pulse_value * 0.5))),
            int(min(255, color[1] * (1.0 + pulse_value * 0.5))),
            int(min(255, color[2] * (1.0 + pulse_value * 0.5)))
        )
        
        # Draw a border rectangle with the pulsed color
        for x in range(element.x, element.x + element.width):
            for y in [element.y, element.y + element.height - 1]:
                if 0 <= x < surface.get_width() // font.size(' ')[0] and 0 <= y < surface.get_height() // font.size(' ')[1]:
                    pos = (x * font.size(' ')[0], y * font.size(' ')[1])
                    surface.blit(font.render('█', True, pulsed_color), pos)
                    
        for y in range(element.y, element.y + element.height):
            for x in [element.x, element.x + element.width - 1]:
                if 0 <= x < surface.get_width() // font.size(' ')[0] and 0 <= y < surface.get_height() // font.size(' ')[1]:
                    pos = (x * font.size(' ')[0], y * font.size(' ')[1])
                    surface.blit(font.render('█', True, pulsed_color), pos)
    
    @staticmethod
    def _draw_glitch_effect(element: UIElement, 
                          surface: pygame.Surface, 
                          font: pygame.font.Font,
                          intensity: float) -> None:
        """Draw a glitch effect.
        
        Args:
            element: UI element
            surface: Surface to draw on
            font: Font to use for rendering
            intensity: Effect intensity
        """
        # Glitch characters
        glitch_chars = ['#', '@', '!', '&', '%', '$', '?', '=', '*', '+', '¦', 'x']
        
        # Number of glitch artifacts based on intensity
        num_glitches = int(3 + 7 * intensity)
        
        # Create random glitch artifacts
        for _ in range(num_glitches):
            # Random position within the element
            x = random.randint(element.x, element.x + element.width - 1)
            y = random.randint(element.y, element.y + element.height - 1)
            
            # Random glitch character
            char = random.choice(glitch_chars)
            
            # Random color
            color = (
                random.randint(150, 255),
                random.randint(50, 200),
                random.randint(150, 255)
            )
            
            # Draw glitch character
            if 0 <= x < surface.get_width() // font.size(' ')[0] and 0 <= y < surface.get_height() // font.size(' ')[1]:
                pos = (x * font.size(' ')[0], y * font.size(' ')[1])
                surface.blit(font.render(char, True, color), pos)
    
    @staticmethod
    def _draw_data_effect(element: UIElement, 
                        surface: pygame.Surface, 
                        font: pygame.font.Font,
                        intensity: float) -> None:
        """Draw a data stream effect.
        
        Args:
            element: UI element
            surface: Surface to draw on
            font: Font to use for rendering
            intensity: Effect intensity
        """
        # Data characters
        data_chars = ['0', '1', '7', '9', '$', '%', '#', '&', '@', '!', ';', ':', '.']
        
        # Calculate the number of data points based on element size and intensity
        area = element.width * element.height
        num_data_points = int(area * 0.05 * intensity)
        
        # Limit to a reasonable number
        num_data_points = min(num_data_points, 20)
        
        # Create data point artifacts
        for _ in range(num_data_points):
            # Random position within the element
            x = random.randint(element.x, element.x + element.width - 1)
            y = random.randint(element.y, element.y + element.height - 1)
            
            # Alternate between edge and interior positioning
            if random.random() < 0.7:  # 70% chance to be on edge
                if random.random() < 0.5:  # 50% horizontal edge
                    y = random.choice([element.y, element.y + element.height - 1])
                else:  # 50% vertical edge
                    x = random.choice([element.x, element.x + element.width - 1])
            
            # Random data character
            char = random.choice(data_chars)
            
            # Matrix green color with variation
            green_value = random.randint(150, 255)
            color = (0, green_value, 0)
            
            # Draw data character
            if 0 <= x < surface.get_width() // font.size(' ')[0] and 0 <= y < surface.get_height() // font.size(' ')[1]:
                pos = (x * font.size(' ')[0], y * font.size(' ')[1])
                surface.blit(font.render(char, True, color), pos)
    
    def remove_enhancement(self, element: UIElement) -> bool:
        """Remove enhancements from a UI element.
        
        Args:
            element: UI element to remove enhancements from
            
        Returns:
            True if enhancements were removed, False otherwise
        """
        element_id = id(element)
        if element_id in self.enhanced_elements:
            enhancement = self.enhanced_elements[element_id]
            
            # Remove animation if applicable
            if 'anim_id' in enhancement:
                self.animation_integrator.remove_animation(enhancement['anim_id'])
            
            # Restore original draw method if possible
            if hasattr(element, '_original_draw'):
                element.draw = getattr(element, '_original_draw')
            
            # Remove enhancement data
            del self.enhanced_elements[element_id]
            return True
        return False
    
    def apply_animation_to_box(self, 
                              box: ASCIIBox, 
                              animation_type: AnimationStyle,
                              duration: float = -1.0,
                              intensity: float = 0.5,
                              **kwargs: Any) -> str:
        """Apply animation to an ASCIIBox.
        
        Args:
            box: ASCIIBox to animate
            animation_type: Type of animation
            duration: Duration in seconds (-1 for infinite)
            intensity: Animation intensity
            **kwargs: Additional parameters
            
        Returns:
            Enhancement ID
        """
        return self.enhance_element(box, animation_type, duration, intensity, **kwargs)
    
    def apply_animation_to_button(self, 
                                button: ASCIIButton, 
                                animation_type: AnimationStyle,
                                duration: float = -1.0,
                                intensity: float = 0.5,
                                **kwargs: Any) -> str:
        """Apply animation to an ASCIIButton.
        
        Args:
            button: ASCIIButton to animate
            animation_type: Type of animation
            duration: Duration in seconds (-1 for infinite)
            intensity: Animation intensity
            **kwargs: Additional parameters
            
        Returns:
            Enhancement ID
        """
        return self.enhance_element(button, animation_type, duration, intensity, **kwargs)
    
    def apply_animation_to_panel(self, 
                               panel: ASCIIPanel, 
                               animation_type: AnimationStyle,
                               duration: float = -1.0,
                               intensity: float = 0.5,
                               **kwargs: Any) -> str:
        """Apply animation to an ASCIIPanel.
        
        Args:
            panel: ASCIIPanel to animate
            animation_type: Type of animation
            duration: Duration in seconds (-1 for infinite)
            intensity: Animation intensity
            **kwargs: Additional parameters
            
        Returns:
            Enhancement ID
        """
        return self.enhance_element(panel, animation_type, duration, intensity, **kwargs)
    
    def apply_animation_to_progress_bar(self, 
                                      progress_bar: ASCIIProgressBar, 
                                      animation_type: AnimationStyle,
                                      duration: float = -1.0,
                                      intensity: float = 0.5,
                                      **kwargs: Any) -> str:
        """Apply animation to an ASCIIProgressBar.
        
        Args:
            progress_bar: ASCIIProgressBar to animate
            animation_type: Type of animation
            duration: Duration in seconds (-1 for infinite)
            intensity: Animation intensity
            **kwargs: Additional parameters
            
        Returns:
            Enhancement ID
        """
        return self.enhance_element(progress_bar, animation_type, duration, intensity, **kwargs)
