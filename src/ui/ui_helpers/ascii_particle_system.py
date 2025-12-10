"""
ascii_particle_system.py

Provides an ASCII-based particle system for creating dynamic visual effects
in the Space Muck's retro terminal interface.
"""

# Standard library imports
import random
import math
from typing import List, Tuple, Optional, Dict, Any, Callable

# Local application imports
from config import COLOR_TEXT, COLOR_BG, COLOR_HIGHLIGHT  # COLOR_ACCENT removed as it doesn't exist
from ui.ui_base.ui_style import UIStyle
from ui.ui_helpers.render_helper import RenderHelper


class ASCIIParticle:
    """Particle for ASCII-based visual effects"""
    
    def __init__(
        self,
        x: float,
        y: float,
        char: str,
        velocity: Tuple[float, float],
        lifetime: float,
        color: Tuple[int, int, int] = COLOR_TEXT,
        gravity: float = 0.0,
        decay: float = 0.0
    ):
        """Initialize a particle with position, movement, and visual properties
        
        Args:
            x: X position (can be fractional for sub-character positioning)
            y: Y position (can be fractional for sub-character positioning)
            char: ASCII character representing the particle
            velocity: (vx, vy) tuple for particle movement
            lifetime: How long the particle exists (in seconds)
            color: RGB color tuple for the particle
            gravity: Optional gravity effect (positive = down)
            decay: How much velocity decreases per second (0.0 = no decay)
        """
        self.x = x
        self.y = y
        self.char = char
        self.vx, self.vy = velocity
        self.lifetime = lifetime
        self.max_lifetime = lifetime  # Store for fade calculations
        self.color = color
        self.gravity = gravity
        self.decay = decay
        
    def update(self, dt: float) -> bool:
        """Update particle position and lifetime
        
        Args:
            dt: Time delta in seconds
            
        Returns:
            True if particle is still alive, False if expired
        """
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Apply gravity
        self.vy += self.gravity * dt
        
        # Apply velocity decay
        if self.decay > 0:
            decay_factor = 1.0 - (self.decay * dt)
            self.vx *= decay_factor
            self.vy *= decay_factor
        
        # Reduce lifetime
        self.lifetime -= dt
        return self.lifetime > 0
    
    def get_display_color(self) -> Tuple[int, int, int]:
        """Get the current display color, adjusted for lifetime fade
        
        Returns:
            RGB color tuple for current particle state
        """
        # Calculate alpha based on remaining lifetime
        alpha = min(1.0, self.lifetime / (self.max_lifetime * 0.7))
        
        # Blend with background color based on alpha
        r = int(self.color[0] * alpha + COLOR_BG[0] * (1.0 - alpha))
        g = int(self.color[1] * alpha + COLOR_BG[1] * (1.0 - alpha))
        b = int(self.color[2] * alpha + COLOR_BG[2] * (1.0 - alpha))
        
        return (r, g, b)


class ASCIIParticleSystem:
    """System for managing and rendering multiple ASCII particles"""
    
    def __init__(self, max_particles: int = 100):
        """Initialize the particle system
        
        Args:
            max_particles: Maximum number of particles allowed in the system
        """
        self.particles: List[ASCIIParticle] = []
        self.max_particles = max_particles
        self.emitters: List[Dict[str, Any]] = []
        self.time_accumulator = 0.0
        
    def add_particle(self, particle: ASCIIParticle) -> bool:
        """Add a particle to the system
        
        Args:
            particle: The particle to add
            
        Returns:
            True if particle was added, False if system is full
        """
        if len(self.particles) < self.max_particles:
            self.particles.append(particle)
            return True
        return False
        
    def add_emitter(
        self,
        x: float,
        y: float,
        emission_rate: float,
        particle_factory: Callable[[], ASCIIParticle],
        duration: Optional[float] = None,
        max_count: Optional[int] = None
    ) -> int:
        """Add a particle emitter to the system
        
        Args:
            x: X position of emitter
            y: Y position of emitter
            emission_rate: Particles per second
            particle_factory: Function that creates particles
            duration: Optional time limit for the emitter
            max_count: Optional maximum number of particles from this emitter
            
        Returns:
            Emitter ID for later reference
        """
        emitter_id = len(self.emitters)
        self.emitters.append({
            "x": x,
            "y": y,
            "rate": emission_rate,
            "factory": particle_factory,
            "duration": duration,
            "elapsed": 0.0,
            "count": 0,
            "max_count": max_count,
            "time_to_next": 1.0 / emission_rate if emission_rate > 0 else float('inf')
        })
        return emitter_id
        
    def remove_emitter(self, emitter_id: int) -> bool:
        """Remove an emitter from the system
        
        Args:
            emitter_id: ID of the emitter to remove
            
        Returns:
            True if emitter was removed, False if not found
        """
        if 0 <= emitter_id < len(self.emitters):
            self.emitters[emitter_id] = None  # Mark as removed
            return True
        return False
        
    def _update_particle(self, dt: float) -> None:
        """Update all existing particles
        
        Args:
            dt: Time delta in seconds
        """
        self.particles = [p for p in self.particles if p.update(dt)]
    
    @staticmethod
    def _check_emitter_expired(emitter: dict, dt: float) -> bool:
        """Check if an emitter has expired due to duration or max count
        
        Args:
            emitter: Emitter to check
            dt: Time delta in seconds
            
        Returns:
            True if emitter should be removed, False otherwise
        """
        # Check duration expiration
        if emitter["duration"] is not None:
            emitter["elapsed"] += dt
            if emitter["elapsed"] >= emitter["duration"]:
                return True

        # Check max count expiration
        return (emitter["max_count"] is not None and 
                emitter["count"] >= emitter["max_count"])
    
    def _spawn_particles(self, emitter: dict, emitter_idx: int) -> None:
        """Spawn new particles from an emitter
        
        Args:
            emitter: Emitter data dictionary
            emitter_idx: Index of emitter in the emitters list
        """
        # Spawn particles while timer allows and we have space
        while emitter["time_to_next"] <= 0 and len(self.particles) < self.max_particles:
            # Create and position new particle
            particle = emitter["factory"]()
            particle.x += emitter["x"]
            particle.y += emitter["y"]
            
            # Add to system and update counters
            self.particles.append(particle)
            emitter["count"] += 1
            emitter["time_to_next"] += 1.0 / emitter["rate"]
            
            # Check if reached max count
            if emitter["max_count"] is not None and emitter["count"] >= emitter["max_count"]:
                self.emitters[emitter_idx] = None
                break
    
    def update(self, dt: float) -> None:
        """Update all particles and emitters
        
        Args:
            dt: Time delta in seconds
        """
        # Update existing particles
        self._update_particle(dt)
        
        # Update emitters and spawn new particles
        for i, emitter in enumerate(self.emitters):
            if emitter is None:
                continue
                
            # Check if emitter has expired
            if self._check_emitter_expired(emitter, dt):
                self.emitters[i] = None
                continue
                
            # Update spawn timer and create particles if needed
            emitter["time_to_next"] -= dt
            if emitter["time_to_next"] <= 0:
                self._spawn_particles(emitter, i)
        
        # Clean up removed emitters
        self.emitters = [e for e in self.emitters if e is not None]
            
    def render(self, surface, font) -> None:
        """Render all particles
        
        Args:
            surface: Surface to render on
            font: Font to use for rendering
        """
        for particle in self.particles:
            # Convert particle position to integer coordinates
            x = int(particle.x)
            y = int(particle.y)
            
            # Get current display color
            color = particle.get_display_color()
            
            # Render the particle
            RenderHelper.draw_char(surface, font, x, y, particle.char, color)
            
    def clear(self) -> None:
        """Clear all particles and emitters"""
        self.particles.clear()
        self.emitters.clear()


# Factory functions for common particle effects

def create_explosion_emitter(
    x: float, 
    y: float, 
    style: UIStyle = UIStyle.MECHANICAL,
    size: float = 5.0
) -> Callable:
    """Create an explosion effect emitter
    
    Args:
        x: X position
        y: Y position
        style: UI style determining appearance
        size: Explosion size
        
    Returns:
        Function to create the emitter
    """
    def factory():
        particle_system = ASCIIParticleSystem(max_particles=50)
        
        def particle_factory() -> ASCIIParticle:
            # Determine particle traits based on style
            if style == UIStyle.SYMBIOTIC:
                chars = ['~', '*', '•', '°', '○']
                colors = [COLOR_HIGHLIGHT, (200, 200, 50), COLOR_TEXT]  # Replaced COLOR_ACCENT with yellow-ish color
            elif style == UIStyle.MECHANICAL:
                chars = ['#', '+', '*', '=', '/']
                colors = [(255, 150, 50), (200, 100, 30), (150, 50, 10)]
            elif style == UIStyle.ASTEROID:
                chars = ['@', '#', '%', '&', '$']
                colors = [(180, 180, 180), (150, 150, 150), (120, 120, 120)]
            elif style == UIStyle.QUANTUM:
                chars = ['⊕', '⊗', '§', 'Φ', '∞']
                colors = [(100, 200, 255), (50, 150, 255), (150, 100, 255)]
            else:  # Default/FLEET
                chars = ['*', '+', '.', '`', '\'']
                colors = [(255, 255, 100), (255, 200, 50), (255, 150, 0)]
                
            # Random attributes
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.0, 5.0) * size / 5.0
            lifetime = random.uniform(0.5, 1.2)
            
            return ASCIIParticle(
                x=0,  # Will be offsetted by emitter position
                y=0,  # Will be offsetted by emitter position
                char=random.choice(chars),
                velocity=(math.cos(angle) * speed, math.sin(angle) * speed),
                lifetime=lifetime,
                color=random.choice(colors),
                gravity=0.2,
                decay=0.9
            )
            
        particle_system.add_emitter(
            x=x,
            y=y,
            emission_rate=40,
            particle_factory=particle_factory,
            duration=0.2,
            max_count=int(30 * size / 5.0)
        )
        
        return particle_system
        
    return factory


def create_sparkle_effect(
    x: float,
    y: float,
    style: UIStyle = UIStyle.MECHANICAL,
    duration: float = 1.0
) -> ASCIIParticleSystem:
    """Create a sparkle effect at the specified position
    
    Args:
        x: X position
        y: Y position
        style: UI style determining appearance
        duration: Effect duration in seconds
        
    Returns:
        Configured particle system
    """
    particle_system = ASCIIParticleSystem(max_particles=20)
    
    def particle_factory() -> ASCIIParticle:
        # Determine particle traits based on style
        if style == UIStyle.SYMBIOTIC:
            chars = ['*', '·', '°', '⁕']
            color = COLOR_HIGHLIGHT
        elif style == UIStyle.MECHANICAL:
            chars = ['+', '*', '⚡', '➛']
            color = (255, 220, 100)
        elif style == UIStyle.ASTEROID:
            chars = ['∗', '⋄', '⁎', '⁑']
            color = (200, 200, 200)
        elif style == UIStyle.QUANTUM:
            chars = ['⊛', '⊕', '⊙', '◉']
            color = (100, 200, 255)
        else:  # Default/FLEET
            chars = ['*', '★', '☆', '⁂']
            color = (255, 255, 150)
            
        # Random movement
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.3, 1.0)
        
        return ASCIIParticle(
            x=random.uniform(-0.5, 0.5),
            y=random.uniform(-0.5, 0.5),
            char=random.choice(chars),
            velocity=(math.cos(angle) * speed, math.sin(angle) * speed),
            lifetime=random.uniform(0.3, duration),
            color=color,
            decay=0.7
        )
    
    particle_system.add_emitter(
        x=x,
        y=y,
        emission_rate=10,
        particle_factory=particle_factory,
        duration=duration
    )
    
    return particle_system


def create_data_stream_effect(
    x: float,
    y: float,
    width: int,
    height: int,
    style: UIStyle = UIStyle.QUANTUM,
    density: float = 0.3
) -> ASCIIParticleSystem:
    """Create a data stream effect in the specified area
    
    Args:
        x: X position
        y: Y position
        width: Width of effect area
        height: Height of effect area
        style: UI style determining appearance
        density: Character density (0.0-1.0)
        
    Returns:
        Configured particle system
    """
    particle_system = ASCIIParticleSystem(max_particles=width * height)
    
    def particle_factory() -> ASCIIParticle:
        # Set characters based on style
        if style == UIStyle.SYMBIOTIC:
            chars = list('~·.,:;!?+*∞')
            color_base = (100, 200, 100)
        elif style == UIStyle.MECHANICAL:
            chars = list('01+-*/=#%&')
            color_base = (200, 150, 50)
        elif style == UIStyle.ASTEROID:
            chars = list('@#%&$.,*+')
            color_base = (150, 150, 150)
        elif style == UIStyle.QUANTUM:
            chars = list('01φΦΨΩδγλπ')
            color_base = (100, 150, 255)
        else:  # Default/FLEET
            chars = list('><^v[]{}()')
            color_base = (150, 200, 255)
            
        # Randomize character
        char = random.choice(chars)
        
        # Calculate position
        pos_x = random.randint(0, width-1)
        
        # Color variation
        color_var = random.randint(-30, 30)
        color = (
            max(0, min(255, color_base[0] + color_var)),
            max(0, min(255, color_base[1] + color_var)),
            max(0, min(255, color_base[2] + color_var))
        )
        
        return ASCIIParticle(
            x=pos_x,
            y=0,  # Start at top
            char=char,
            velocity=(0, random.uniform(3.0, 8.0)),  # Move downward
            lifetime=random.uniform(1.0, 3.0),
            color=color
        )
    
    # Create multiple emitters across the width
    for i in range(width):
        if random.random() < density:
            rate = random.uniform(0.2, 2.0)
            particle_system.add_emitter(
                x=x,
                y=y,
                emission_rate=rate,
                particle_factory=particle_factory
            )
    
    return particle_system
