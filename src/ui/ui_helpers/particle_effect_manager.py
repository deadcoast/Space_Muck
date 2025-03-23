"""
particle_effect_manager.py

This module provides management for particle effects in UI components.
"""

# Standard library imports
import math
import random
from typing import Dict, Optional, Tuple, Any

# Third-party library imports
import pygame

# Local application imports
# Local ui imports
from ui.ui_helpers.ascii_particle_system import ASCIIParticle, ASCIIParticleSystem
from ui.ui_helpers.animation_helper import AnimationStyle


class ParticleEffectManager:
    """Manages particle effects for UI components."""
    
    def __init__(self):
        """Initialize the particle effect manager."""
        self.particle_systems: Dict[str, ASCIIParticleSystem] = {}
        self.max_particles = 200  # Default maximum particles across all systems
    
    def create_particle_system(self, system_id: str, max_particles: int = 50) -> ASCIIParticleSystem:
        """Create a new particle system.
        
        Args:
            system_id: Unique identifier for the particle system
            max_particles: Maximum number of particles for this system
            
        Returns:
            New particle system instance
        """
        if system_id in self.particle_systems:
            return self.particle_systems[system_id]
            
        system = ASCIIParticleSystem(max_particles)
        self.particle_systems[system_id] = system
        return system
    
    def remove_particle_system(self, system_id: str) -> bool:
        """Remove a particle system.
        
        Args:
            system_id: ID of the particle system to remove
            
        Returns:
            True if the system was removed, False otherwise
        """
        if system_id in self.particle_systems:
            del self.particle_systems[system_id]
            return True
        return False
    
    def update(self, dt: float) -> None:
        """Update all particle systems.
        
        Args:
            dt: Time delta in seconds
        """
        # Update each particle system
        for system in self.particle_systems.values():
            system.update(dt)
            
        # Remove empty systems
        empty_systems = [sid for sid, system in self.particle_systems.items() 
                         if len(system.particles) == 0 and len(system.emitters) == 0]
        for sid in empty_systems:
            self.remove_particle_system(sid)
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render all particle systems.
        
        Args:
            surface: Surface to render on
            font: Font to use for rendering
        """
        for system in self.particle_systems.values():
            system.render(surface, font)

    def create_effect(self, 
                     system_id: str, 
                     effect_type: AnimationStyle, 
                     x: int, 
                     y: int, 
                     width: int = 1, 
                     height: int = 1, 
                     duration: Optional[float] = 2.0, 
                     **kwargs: Any) -> str:
        """Create a particle effect.
        
        Args:
            system_id: Unique identifier for the particle system
            effect_type: Type of effect to create
            x: X-coordinate for the effect center
            y: Y-coordinate for the effect center
            width: Width of the effect area
            height: Height of the effect area
            duration: Duration of the effect in seconds (None for infinite)
            **kwargs: Additional effect-specific parameters
            
        Returns:
            ID of the created emitter
        """
        # Get or create particle system
        system = self.create_particle_system(system_id)
        
        # Set up emitter based on effect type
        if effect_type == AnimationStyle.SPARKLE:
            return self._create_sparkle_effect(system, x, y, width, height, duration, **kwargs)
        elif effect_type == AnimationStyle.GLITCH:
            return self._create_glitch_effect(system, x, y, width, height, duration, **kwargs)
        elif effect_type == AnimationStyle.DATA_STREAM:
            return self._create_data_stream_effect(system, x, y, width, height, duration, **kwargs)
        else:
            # Default to a basic particle effect (also used for PARTICLE type)
            return self._create_general_particle_effect(system, x, y, width, height, duration, **kwargs)
    
    def _create_sparkle_effect(self, 
                              system: ASCIIParticleSystem, 
                              x: int, 
                              y: int, 
                              width: int, 
                              height: int, 
                              duration: Optional[float], 
                              **kwargs: Any) -> str:
        """Create a sparkle effect.
        
        Args:
            system: Particle system to use
            x, y, width, height: Position and size of effect
            duration: Duration of the effect
            **kwargs: Additional parameters
            
        Returns:
            ID of the created emitter
        """
        # Extract sparkle-specific parameters
        density = kwargs.get('density', 0.3)
        color = kwargs.get('color', (255, 255, 255))
        chars = kwargs.get('chars', ['*', '+', '.', '·', '✧', '✦', '✳'])
        
        # Create particle factory function
        def create_sparkle_particle():
            # Random position within area
            px = random.uniform(-width/2, width/2)
            py = random.uniform(-height/2, height/2)
            
            # Random lifetime
            lifetime = random.uniform(0.3, 1.2)
            
            # Random character
            char = random.choice(chars)
            
            # Create particle with sparkle behavior
            particle = ASCIIParticle(px, py, char, (0, 0), lifetime, color)
            
            # Add sparkle fade-in/out behavior
            particle.alpha_curve = lambda t: min(1.0, 2 * min(t, 1.0 - t))
            
            return particle
        
        # Add emitter to system
        return system.add_emitter(
            x + width // 2,  # Center X
            y + height // 2,  # Center Y
            emission_rate=density * 10,  # Particles per second based on density
            particle_factory=create_sparkle_particle,
            duration=duration
        )
    
    def _create_glitch_effect(self, 
                             system: ASCIIParticleSystem, 
                             x: int, 
                             y: int, 
                             width: int, 
                             height: int, 
                             duration: Optional[float], 
                             **kwargs: Any) -> str:
        """Create a glitch effect.
        
        Args:
            system: Particle system to use
            x, y, width, height: Position and size of effect
            duration: Duration of the effect
            **kwargs: Additional parameters
            
        Returns:
            ID of the created emitter
        """
        # Extract glitch-specific parameters
        intensity = kwargs.get('intensity', 0.5)
        color = kwargs.get('color', (200, 100, 220))
        
        # Use technical/computer glitch characters
        chars = kwargs.get('chars', ['#', '@', '!', '&', '%', '$', '?', '=', '*', '+', '¦', 'x'])
        
        # Create particle factory function
        def create_glitch_particle():
            # Determine if horizontal or vertical glitch line
            is_horizontal = random.random() < 0.5
            
            if is_horizontal:
                # Horizontal glitch line
                px = random.uniform(-width/2, width/2)
                py = random.uniform(-height/2, height/2)
                vx = random.uniform(3.0, 8.0) * (1 if random.random() < 0.5 else -1)
                vy = 0
            else:
                # Vertical glitch line
                px = random.uniform(-width/2, width/2)
                py = random.uniform(-height/2, height/2)
                vx = 0
                vy = random.uniform(2.0, 5.0) * (1 if random.random() < 0.5 else -1)
            
            # Random lifetime
            lifetime = random.uniform(0.1, 0.4)
            
            # Random character
            char = random.choice(chars)
            
            # Create particle with glitch behavior
            particle = ASCIIParticle(px, py, char, (vx, vy), lifetime, color)
            
            # Glitch particles have digital, abrupt alpha
            particle.alpha_curve = lambda t: 1.0 if t < 0.8 else 0.0
            
            return particle
        
        # Add emitter to system
        return system.add_emitter(
            x + width // 2,  # Center X
            y + height // 2,  # Center Y
            emission_rate=intensity * 15,  # Particles per second based on intensity
            particle_factory=create_glitch_particle,
            duration=duration
        )
    
    def _get_velocity_for_direction(self, direction: str) -> Tuple[float, float]:
        """Get velocity components based on direction.
        
        Args:
            direction: Stream direction ('down', 'up', 'left', 'right')
            
        Returns:
            Tuple of (vx, vy) velocity components
        """
        vx, vy = 0, 0
        if direction == 'down':
            vy = random.uniform(2.0, 5.0)
        elif direction == 'up':
            vy = random.uniform(-2.0, -5.0)
        elif direction == 'right':
            vx = random.uniform(2.0, 5.0)
        elif direction == 'left':
            vx = random.uniform(-2.0, -5.0)
        return vx, vy
    
    def _get_stream_position(self, direction: str, width: int, height: int) -> Tuple[float, float]:
        """Get initial position for a data stream particle.
        
        Args:
            direction: Stream direction
            width: Effect width
            height: Effect height
            
        Returns:
            Tuple of (px, py) position coordinates
        """
        if direction in ('down', 'up'):
            # Horizontal position anywhere in width, vertical at edge
            px = random.uniform(-width/2, width/2)
            py = -height/2 if direction == 'down' else height/2
        else:
            # Vertical position anywhere in height, horizontal at edge
            px = -width/2 if direction == 'right' else width/2
            py = random.uniform(-height/2, height/2)
        return px, py

    def _create_data_stream_effect(self, 
                                  system: ASCIIParticleSystem, 
                                  x: int, 
                                  y: int, 
                                  width: int, 
                                  height: int, 
                                  duration: Optional[float], 
                                  **kwargs: Any) -> str:
        """Create a Matrix-like data stream effect.
        
        Args:
            system: Particle system to use
            x, y, width, height: Position and size of effect
            duration: Duration of the effect
            **kwargs: Additional parameters
            
        Returns:
            ID of the created emitter
        """
        # Extract data stream specific parameters
        density = kwargs.get('density', 0.5)
        color = kwargs.get('color', (0, 255, 100))
        direction = kwargs.get('direction', 'down')  # 'down', 'up', 'left', 'right'
        
        # Data characters
        chars = kwargs.get('chars', ['0', '1', '7', '9', '$', '%', '#', '&', '@', '!', ';', ':', '.'])
        
        # Get velocity components for this direction
        vx, vy = self._get_velocity_for_direction(direction)
        
        # Create particle factory function
        def create_data_stream_particle():
            # Get position based on stream direction
            px, py = self._get_stream_position(direction, width, height)
            
            # Longer lifetime for data streams
            lifetime = random.uniform(1.0, 3.0)
            
            # Random character with random changes
            char = random.choice(chars)
            
            # Create particle
            particle = ASCIIParticle(px, py, char, (vx, vy), lifetime, color)
            
            # Data stream particles change their character occasionally
            original_update = particle.update
            
            def updated_update(dt):
                if random.random() < 0.1:
                    particle.char = random.choice(chars)
                return original_update(dt)
            
            particle.update = updated_update
            
            # Fade alpha at the start and end
            particle.alpha_curve = lambda t: min(3 * t, 3 * (1 - t), 1.0)
            
            return particle
        
        # Add emitter to system
        return system.add_emitter(
            x,
            y,
            emission_rate=density * 8,  # Particles per second based on density
            particle_factory=create_data_stream_particle,
            duration=duration
        )
    
    def _create_general_particle_effect(self, 
                                       system: ASCIIParticleSystem, 
                                       x: int, 
                                       y: int, 
                                       width: int, 
                                       height: int, 
                                       duration: Optional[float], 
                                       **kwargs: Any) -> str:
        """Create a general particle effect.
        
        Args:
            system: Particle system to use
            x, y, width, height: Position and size of effect
            duration: Duration of the effect
            **kwargs: Additional parameters
            
        Returns:
            ID of the created emitter
        """
        # Extract general parameters
        intensity = kwargs.get('intensity', 0.5)
        color = kwargs.get('color', (255, 255, 255))
        chars = kwargs.get('chars', ['.', '*', '+', 'o', 'O', '·'])
        
        # Create particle factory function
        def create_general_particle():
            # Random position
            px = random.uniform(-width/4, width/4)
            py = random.uniform(-height/4, height/4)
            
            # Random velocity (exploding outward)
            angle = random.uniform(0, 2 * 3.14159)
            speed = random.uniform(1.0, 3.0)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            
            # Random lifetime
            lifetime = random.uniform(0.3, 1.0)
            
            # Random character
            char = random.choice(chars)
            
            # Create particle
            particle = ASCIIParticle(px, py, char, (vx, vy), lifetime, color)
            
            # Particles fade out at the end
            particle.alpha_curve = lambda t: min(2 * t, 1.0) if t < 0.5 else 1.0 - (t - 0.5) * 2
            
            return particle
        
        # Add emitter to system
        return system.add_emitter(
            x + width // 2,  # Center X
            y + height // 2,  # Center Y
            emission_rate=intensity * 20,  # Particles per second based on intensity
            particle_factory=create_general_particle,
            duration=duration
        )
