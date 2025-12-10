"""
animation_demo.py

This module demonstrates the enhanced animation effects for ASCII UI components.
"""

# Standard library imports
import math
import random
import time
# No type imports needed

# Third-party library imports
import pygame

# Local application imports
# Local imports only
from ui.ui_base.ui_style import UIStyle
from ui.ui_base.ascii_ui import ASCIIBox, ASCIIButton, ASCIIPanel, ASCIIProgressBar
from ui.ui_helpers.animation_helper import AnimationStyle
from ui.ui_helpers.ui_component_enhancer import UIComponentEnhancer


class AnimationDemo:
    """Demo for showcasing the enhanced animation features."""
    
    def __init__(self, surface: pygame.Surface, font: pygame.font.Font):
        """Initialize the animation demo.
        
        Args:
            surface: Surface to render on
            font: Font to use for rendering
        """
        self.surface = surface
        self.font = font
        self.running = False
        self.enhancer = UIComponentEnhancer.get_instance()
        self.clock = pygame.time.Clock()
        
        # Track time for animations
        self.start_time = 0
        self.elapsed_time = 0
        
        # Create UI elements
        self.ui_elements = []
        self.create_ui_elements()
        
    def create_ui_elements(self) -> None:
        """Create the UI elements for the demo."""
        surface_width = self.surface.get_width()
        surface_height = self.surface.get_height()
        char_width = self.font.size(' ')[0]
        char_height = self.font.size(' ')[1]
        
        # Calculate grid dimensions in character coordinates
        grid_width = surface_width // char_width
        grid_height = surface_height // char_height
        
        # Create a main panel
        main_panel = ASCIIPanel(
            2, 2, 
            grid_width - 4, 
            grid_height - 4, 
            UIStyle.MECHANICAL,
            "Animation Styles Demo"
        )
        self.ui_elements.append(main_panel)
        
        # Create a row of boxes to demonstrate different animation styles
        box_width = 20
        box_height = 7
        spacing = 3
        boxes_per_row = (grid_width - 8) // (box_width + spacing)
        
        styles = [
            (AnimationStyle.PULSE, "PULSE", UIStyle.MECHANICAL),
            (AnimationStyle.DATA_STREAM, "DATA_STREAM", UIStyle.QUANTUM),
            (AnimationStyle.GLITCH, "GLITCH", UIStyle.SYMBIOTIC),
            (AnimationStyle.PARTICLE, "PARTICLE", UIStyle.FLEET),
            (AnimationStyle.SPARKLE, "SPARKLE", UIStyle.ASTEROID)
        ]
        
        # Create boxes for each animation style
        for i, (style, name, ui_style) in enumerate(styles):
            row = i // boxes_per_row
            col = i % boxes_per_row
            
            x = 5 + col * (box_width + spacing)
            y = 5 + row * (box_height + spacing)
            
            box = ASCIIBox(
                x, y,
                box_width, box_height,
                ui_style,
                f"{name}",
                f"Animation style:\n{name}\n\nUI Style: {ui_style.name}"
            )
            
            # Apply animation to box
            self.enhancer.apply_animation_to_box(
                box, 
                style, 
                duration=-1.0,  # Infinite duration
                intensity=0.7
            )
            
            main_panel.add_child(box)
            self.ui_elements.append(box)
        
        # Create buttons with animation effects
        button_y = 5 + ((len(styles) - 1) // boxes_per_row + 1) * (box_height + spacing) + 2
        
        for i, (style, name, ui_style) in enumerate(styles):
            col = i % boxes_per_row
            
            x = 5 + col * (box_width + spacing)
            y = button_y
            
            button = ASCIIButton(
                x, y,
                f"{name} Button",
                lambda s=style: self.toggle_animation(s),
                ui_style,
                box_width, 3
            )
            
            # Apply animation to button
            self.enhancer.apply_animation_to_button(
                button, 
                style, 
                duration=-1.0,  # Infinite duration
                intensity=0.5
            )
            
            main_panel.add_child(button)
            self.ui_elements.append(button)
        
        # Create progress bars with different animation styles
        progress_y = button_y + 5
        
        for i, (style, name, ui_style) in enumerate(styles):
            col = i % boxes_per_row
            
            x = 5 + col * (box_width + spacing)
            y = progress_y
            
            progress_bar = ASCIIProgressBar(
                x, y,
                box_width, 3,
                ui_style,
                0.0,
                f"{name}"
            )
            
            # Apply animation to progress bar
            self.enhancer.apply_animation_to_progress_bar(
                progress_bar, 
                style, 
                duration=-1.0,  # Infinite duration
                intensity=0.6
            )
            
            main_panel.add_child(progress_bar)
            self.ui_elements.append(progress_bar)
            
            # Store reference to update progress
            if isinstance(progress_bar, ASCIIProgressBar):
                setattr(self, f"{name.lower()}_progress", progress_bar)
    
    @staticmethod
    def toggle_animation(style: AnimationStyle) -> None:
        """Toggle an animation on/off based on its style.
        
        Args:
            style: Animation style to toggle
        """
        # This is a placeholder for button click handling
        print(f"Toggled {style.name} animation")
    
    def update_progress_bars(self) -> None:
        """Update all progress bars for the demo."""
        t = self.elapsed_time
        
        # Update progress bars with different patterns
        if hasattr(self, "pulse_progress"):
            # Smooth oscillation
            self.pulse_progress.set_progress(0.5 + 0.5 * math.sin(t))
            
        if hasattr(self, "data_stream_progress"):
            # Stepped progress
            self.data_stream_progress.set_progress((t * 0.2) % 1.0)
            
        if hasattr(self, "glitch_progress"):
            # Random jumps
            if random.random() < 0.05:
                self.glitch_progress.set_progress(random.random())
            
        if hasattr(self, "particle_progress"):
            # Accelerating progress
            self.particle_progress.set_progress(
                (math.sin(t * 0.5) + 1) / 2
            )
            
        if hasattr(self, "sparkle_progress"):
            # Smooth climbing with resets
            progress = (t * 0.1) % 1.0
            self.sparkle_progress.set_progress(progress)
    
    def _handle_input_events(self) -> None:
        """Handle keyboard and mouse input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    
            # Handle mouse events
            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                self._handle_mouse_event(event)
    
    def _handle_mouse_event(self, event) -> None:
        """Process mouse events and dispatch to UI elements.
        
        Args:
            event: The pygame event to process
        """
        # Convert mouse position to character coordinates
        mouse_x, mouse_y = event.pos
        char_x = mouse_x // self.font.size(' ')[0]
        char_y = mouse_y // self.font.size(' ')[1]
        
        for element in self.ui_elements:
            if hasattr(element, "handle_mouse_event"):
                element.handle_mouse_event(event.type, (char_x, char_y))
    
    def _update_game_state(self, dt: float) -> None:
        """Update the game state.
        
        Args:
            dt: Time delta in seconds
        """
        # Update animations and effects
        self.enhancer.update(dt)
        self.update_progress_bars()
    
    def _render_frame(self) -> None:
        """Render a frame of the animation demo."""
        # Clear screen
        self.surface.fill((0, 0, 0))
        
        # Draw UI elements
        for element in self.ui_elements:
            element.draw(self.surface, self.font)
        
        # Draw particle effects and other enhancements
        self.enhancer.render(self.surface, self.font)
        
        # Update display
        pygame.display.flip()
    
    def run(self) -> None:
        """Run the animation demo."""
        self.running = True
        self.start_time = time.time()
        
        while self.running:
            # Handle events
            self._handle_input_events()
            
            # Calculate time
            current_time = time.time()
            dt = current_time - self.start_time - self.elapsed_time
            self.elapsed_time = current_time - self.start_time
            
            # Update game state
            self._update_game_state(dt)
            
            # Render the frame
            self._render_frame()
            
            # Cap framerate
            self.clock.tick(60)


def run_animation_demo():
    """Initialize and run the animation demo."""
    # Initialize pygame
    pygame.init()
    
    # Create display surface
    screen = pygame.display.set_mode((1024, 768))
    pygame.display.set_caption("Space Muck - ASCII Animation Demo")
    
    # Create font
    font = pygame.font.SysFont("consolas", 16)
    
    # Create and run demo
    demo = AnimationDemo(screen, font)
    demo.run()
    
    # Clean up
    pygame.quit()


if __name__ == "__main__":
    run_animation_demo()
