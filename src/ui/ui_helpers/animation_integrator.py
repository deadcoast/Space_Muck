"""
animation_integrator.py

This module provides integration of animation effects with UI components.
"""

# Standard library imports
import math
import random
from typing import Dict, Optional, Tuple, Any, Callable

# Third-party library imports
import pygame

# Local application imports
# from config import COLOR_TEXT  # Uncomment if needed in the future
from ui.ui_base.ui_style import UIStyle
from ui.ui_helpers.animation_helper import AnimationStyle
from ui.ui_helpers.particle_effect_manager import ParticleEffectManager
from ui.ui_helpers.render_helper import RenderHelper


class AnimationIntegrator:
    """Integrates animation effects with UI components."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'AnimationIntegrator':
        """Get or create the singleton instance.
        
        Returns:
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the animation integrator."""
        self.particle_manager = ParticleEffectManager()
        self.active_animations: Dict[str, Dict[str, Any]] = {}
        self.animation_id_counter = 0
    
    def update(self, dt: float) -> None:
        """Update all animations.
        
        Args:
            dt: Time delta in seconds
        """
        # Update particle systems
        self.particle_manager.update(dt)
        
        # Update other animations
        completed = []
        for anim_id, anim_data in self.active_animations.items():
            anim_data['elapsed'] += dt
            progress = min(1.0, anim_data['elapsed'] / anim_data['duration'])
            
            # Update progress
            anim_data['progress'] = progress
            
            # Call update callback if provided
            if anim_data['update_callback']:
                anim_data['update_callback'](progress)
            
            # Check if animation has completed
            if progress >= 1.0:
                # Call completion callback if provided
                if anim_data['complete_callback']:
                    anim_data['complete_callback']()
                completed.append(anim_id)
        
        # Remove completed animations
        for anim_id in completed:
            del self.active_animations[anim_id]
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render all animations.
        
        Args:
            surface: Surface to render on
            font: Font to use for rendering
        """
        # Render particle effects
        self.particle_manager.render(surface, font)
    
    def add_animation(self, 
                     animation_type: AnimationStyle, 
                     duration: float, 
                     element_rect: pygame.Rect,
                     update_callback: Optional[Callable[[float], None]] = None,
                     complete_callback: Optional[Callable[[], None]] = None,
                     **kwargs: Any) -> str:
        """Add a new animation to an element.
        
        Args:
            animation_type: Type of animation to add
            duration: Duration of the animation in seconds
            element_rect: Rectangle defining the element's position and size
            update_callback: Optional callback for updates (passed progress value)
            complete_callback: Optional callback for animation completion
            **kwargs: Additional animation-specific parameters
            
        Returns:
            ID of the created animation
        """
        # Generate unique animation ID
        anim_id = f"anim_{self.animation_id_counter}"
        self.animation_id_counter += 1
        
        # Set up base animation data
        self.active_animations[anim_id] = {
            'type': animation_type,
            'duration': duration,
            'elapsed': 0.0,
            'progress': 0.0,
            'rect': element_rect,
            'update_callback': update_callback,
            'complete_callback': complete_callback,
            'params': kwargs
        }
        
        # Create particle effects for appropriate animation types
        if animation_type in [AnimationStyle.PARTICLE, AnimationStyle.SPARKLE, 
                             AnimationStyle.GLITCH, AnimationStyle.DATA_STREAM]:
            self._setup_particle_animation(anim_id, animation_type, element_rect, duration, **kwargs)
        
        return anim_id
    
    def _setup_particle_animation(self, 
                                 anim_id: str, 
                                 animation_type: AnimationStyle, 
                                 element_rect: pygame.Rect,
                                 duration: float,
                                 **kwargs: Any) -> None:
        """Set up a particle-based animation.
        
        Args:
            anim_id: Animation ID
            animation_type: Type of animation
            element_rect: Rectangle of the element
            duration: Duration in seconds
            **kwargs: Additional parameters
        """
        self.particle_manager.create_effect(
            anim_id,  # Use animation ID as system ID
            animation_type,
            element_rect.x,
            element_rect.y,
            element_rect.width,
            element_rect.height,
            duration,
            **kwargs
        )
    
    def remove_animation(self, anim_id: str) -> bool:
        """Remove an animation.
        
        Args:
            anim_id: ID of the animation to remove
            
        Returns:
            True if the animation was removed, False otherwise
        """
        if anim_id in self.active_animations:
            del self.active_animations[anim_id]
            self.particle_manager.remove_particle_system(anim_id)
            return True
        return False
    
    def get_animated_character(self, 
                              char: str, 
                              anim_type: AnimationStyle, 
                              progress: float, 
                              **kwargs: Any) -> str:
        """Get an animated version of a character.
        
        Args:
            char: Original character
            anim_type: Animation type
            progress: Animation progress (0.0 to 1.0)
            **kwargs: Additional animation parameters
            
        Returns:
            Animated character
        """
        intensity = kwargs.get('intensity', 0.5)
        
        if anim_type == AnimationStyle.GLITCH:
            # Random character replacement with technical symbols
            if random.random() < 0.1 * intensity:
                glitch_chars = ['#', '@', '!', '&', '%', '$', '?', '=', '*', '+']
                return random.choice(glitch_chars)
            return char
            
        elif anim_type == AnimationStyle.PULSE:
            # No character change for pulse (color change handled elsewhere)
            return char
            
        elif anim_type == AnimationStyle.DATA_STREAM:
            # Randomly replace with data characters
            if random.random() < 0.05 * intensity:
                data_chars = ['0', '1', '7', '9', '$', '%', '#', '&', '@']
                return random.choice(data_chars)
            return char
            
        elif anim_type == AnimationStyle.SPARKLE:
            # Occasionally add sparkle characters
            if random.random() < 0.03 * intensity:
                sparkle_chars = ['*', '+', '.', '·', '✧', '✦']
                return random.choice(sparkle_chars)
            return char
            
        # Default: return the original character
        return char
    
    def get_animated_color(self, 
                          base_color: Tuple[int, int, int], 
                          anim_type: AnimationStyle, 
                          progress: float, 
                          phase: float = 0.0, 
                          **kwargs: Any) -> Tuple[int, int, int]:
        """Get an animated version of a color.
        
        Args:
            base_color: Original RGB color
            anim_type: Animation type
            progress: Animation progress (0.0 to 1.0)
            phase: Animation phase offset
            **kwargs: Additional animation parameters
            
        Returns:
            Animated RGB color
        """
        # Use RenderHelper's animation effect application
        style = kwargs.get('style', UIStyle.MECHANICAL)
        intensity = kwargs.get('intensity', 1.0)
        
        # Custom handling for new animation types
        if anim_type == AnimationStyle.PULSE:
            r, g, b = base_color
            # Pulsating effect (brightness oscillation)
            pulse = 1.0 + 0.5 * intensity * math.sin((progress + phase) * 6.0 * math.pi)
            return (
                int(min(255, r * pulse)), 
                int(min(255, g * pulse)), 
                int(min(255, b * pulse))
            )
        
        # Delegate to RenderHelper for other animation types
        return RenderHelper.apply_animation_effect(base_color, style, progress, phase)
