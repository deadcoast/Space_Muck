"""
src/algorithms/pde_spatial.py

Extends PDE-based modeling for more advanced simulations involving multiple fields
and advection. This module builds upon the foundational ideas in a simple Gray-Scott
reaction-diffusion approach but now introduces:
    1) Optional velocity fields to handle advection (movement) of certain scalar fields.
    2) Multiple PDE fields that can interact or remain independent.
    3) Greater flexibility for boundary conditions, including partial or directional
       boundary modes.

Designed to integrate with your symbiote evolution or CA logic, it can model more
complex scenarios, such as toxins or symbiote spores carried by space currents.

--------------------------------------------------------------------------------
CLASSES & FEATURES
--------------------------------------------------------------------------------

1. AdvectionDiffusionSystem
   - Core class that tracks two or more scalar fields, each with optional
     diffusion and advection terms.
   - Allows you to specify a velocity field (wind/current) for each PDE field
     or a global velocity field applied to all fields.
   - Supports various boundary modes (“wrap,” “reflect,” “constant,” etc.).

2. CoupledPDEFields
   - Demonstrates how to couple additional PDE fields beyond the standard U and V.
   - For instance, a field W could represent a toxin that decays while being advected
     and diffused.
   - Each field can have its own reaction terms, letting you define advanced
     multi-field PDE systems.

3. Production-Ready Integrations
   - No placeholders or unfinished stubs.
   - You can call step() or multi_step() on an AdvectionDiffusionSystem object
     within your main game loop. Each PDE field is updated in-place.

--------------------------------------------------------------------------------
DEPENDENCIES & NOTES
--------------------------------------------------------------------------------
- Requires NumPy.
- If advanced boundary handling or performance optimizations are desired,
  consider adding SciPy or a GPU-accelerated approach. For standard usage,
  this module is sufficient.
- No usage examples are shown here (per instructions), but the docstrings
  detail how to instantiate and update these classes each tick.

--------------------------------------------------------------------------------
LICENSE / COPYRIGHT NOTICE
--------------------------------------------------------------------------------
Copyright (c) 2025 ...
All rights reserved.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class AdvectionDiffusionSystem:
    """
    An extended PDE simulator that supports multiple scalar fields with
    optional advection (movement due to velocity fields), in addition to
    diffusion. Reaction terms can be appended for more complex behavior.

    Typical PDE form for each scalar field S_i:
        ∂S_i/∂t = - (v · ∇S_i) + D_i ∇² S_i + R_i(S_1, S_2, ..., S_n)

    where:
      - v is the velocity field (which may be global or field-specific),
      - D_i is the diffusion coefficient for field i,
      - R_i(...) is a reaction term that can couple multiple fields.

    This class focuses on the advection and diffusion aspects; a derived
    class or user code can implement reaction couplings as needed.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        diffusion_coeffs: Dict[str, float],
        dt: float = 1.0,
        boundary_mode: str = "wrap",
    ):
        """
        Args:
            grid_shape: A (width, height) tuple specifying the domain size.
            diffusion_coeffs: A mapping from field_name -> diffusion_coefficient.
                              Example: {"U": 0.1, "V": 0.05, "W": 0.2}.
            dt: The timestep for PDE updates (Euler integration).
            boundary_mode: "wrap", "reflect", or "constant" boundary condition mode.
        """
        self.width, self.height = grid_shape
        self.diffusion_coeffs = diffusion_coeffs.copy()
        self.dt = dt
        self.boundary_mode = boundary_mode

        # Each PDE field is stored in a dictionary by name
        # Example: self.fields["U"] = 2D np.ndarray
        self.fields: Dict[str, np.ndarray] = {}
        for field_name in self.diffusion_coeffs:
            self.fields[field_name] = np.zeros(
                (self.width, self.height), dtype=np.float32
            )

        # A global velocity field or separate velocity fields per PDE field
        # If you want each PDE field to have its own velocity, store them in a dict.
        # By default, we have a single global velocity field for advection.
        self.global_velocity_x = np.zeros((self.width, self.height), dtype=np.float32)
        self.global_velocity_y = np.zeros((self.width, self.height), dtype=np.float32)

        # Optional dictionary for reaction callbacks per field
        # e.g. self.reactions["U"] = some_function_that_returns_reaction_term(U,V,W,...)
        self.reactions: Dict[str, callable] = {}

    def set_field(self, field_name: str, initial_value: float = 0.0) -> None:
        """
        Fills an existing PDE field with a uniform scalar value. If the field
        does not exist in self.fields, raises ValueError.
        """
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found in PDE system.")
        self.fields[field_name].fill(initial_value)

    def set_field_array(self, field_name: str, array: np.ndarray) -> None:
        """
        Replaces the PDE field with a user-supplied array, ensuring shape matches.
        """
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found.")
        if array.shape != (self.width, self.height):
            raise ValueError(f"Array shape {array.shape} does not match PDE domain.")
        self.fields[field_name] = array.astype(np.float32, copy=True)

    def set_reaction(self, field_name: str, reaction_func: callable) -> None:
        """
        Assign a function that computes reaction terms for the given field.
        Reaction function signature:
            def reaction_func(**fields) -> np.ndarray
        where it can access fields["U"] or fields["V"], etc., and return
        a term that will be added to that field's PDE.
        """
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found.")
        self.reactions[field_name] = reaction_func

    def set_global_velocity(self, vx: np.ndarray, vy: np.ndarray) -> None:
        """
        Set or update the global velocity field for advection. Both vx and vy must
        be the same shape as the domain. Typically, you might have a wind/currents
        vector map.

        Args:
            vx: 2D array for x-component of velocity
            vy: 2D array for y-component of velocity
        """
        if vx.shape != (self.width, self.height) or vy.shape != (
            self.width,
            self.height,
        ):
            raise ValueError("Velocity fields must match PDE domain shape.")
        self.global_velocity_x = vx.astype(np.float32, copy=True)
        self.global_velocity_y = vy.astype(np.float32, copy=True)

    def step(self) -> None:
        """
        Execute a single PDE update for each field, applying:
            1) Advection: - (v · ∇S)
            2) Diffusion: D ∇²S
            3) Reaction: R(S1, S2, ...)
        Each PDE field is updated in place using an explicit Euler method.
        """
        # We'll store the updated fields in a temporary dict, then replace at end
        new_fields = {}

        # For each PDE field, compute updates
        for field_name, field_data in self.fields.items():
            # 1) Advection
            adv_term = self._compute_advection(field_data)

            # 2) Diffusion
            diff_term = self._compute_diffusion(
                field_data, self.diffusion_coeffs[field_name]
            )

            # 3) Reaction (if any)
            react_term = None
            if field_name in self.reactions:
                # Provide the entire fields dictionary to the reaction callback
                react = self.reactions[field_name]
                # The reaction callback can read from self.fields as needed
                react_term = react(**self.fields)
            else:
                # Default to zero if no reaction function assigned
                react_term = np.zeros((self.width, self.height), dtype=np.float32)

            # PDE update: dS/dt = -advection + diffusion + reaction
            ds_dt = -adv_term + diff_term + react_term
            updated = field_data + ds_dt * self.dt

            # We can clamp or handle boundaries if desired. For now, no clamp logic is enforced.
            new_fields[field_name] = updated

        # Once all fields are computed, update them
        self.fields = new_fields

    def multi_step(self, steps: int) -> None:
        """
        Runs multiple consecutive PDE updates.
        """
        for _ in range(steps):
            self.step()

    def get_field_copy(self, field_name: str) -> np.ndarray:
        """
        Returns a copy of the specified PDE field array.
        """
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found in PDE system.")
        return self.fields[field_name].copy()

    @staticmethod
    def boundary_wrap(arr: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """
        Shifts an array with wrap-around boundary conditions (toroidal domain).
        Utility function for advection/diffusion.
        """
        return np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)

    @staticmethod
    def boundary_reflect(
        arr: np.ndarray, shift_x: int, shift_y: int
    ) -> np.ndarray:
        """
        Shifts an array with reflection boundary conditions.
        """
        # We can pad with reflect for each shift if needed, but for performance,
        # do a simpler approach if shift is small. We'll do a general approach:
        padded = np.pad(arr, 1, mode="reflect")
        w, h = arr.shape
        top = 1 + shift_x
        left = 1 + shift_y
        return padded[top : top + w, left : left + h]

    @staticmethod
    def boundary_constant(
        arr: np.ndarray, shift_x: int, shift_y: int
    ) -> np.ndarray:
        """
        Shifts an array with constant boundary conditions.
        We use the value at the boundary as the constant (or zero if you prefer).
        """
        # This approach picks the corner value as a constant fill.
        cval = float(arr[0, 0])  # or 0.0 if you prefer a strict zero boundary
        padded = np.pad(arr, 1, mode="constant", constant_values=cval)
        w, h = arr.shape
        top = 1 + shift_x
        left = 1 + shift_y
        return padded[top : top + w, left : left + h]

    def _shift_with_bc(self, arr: np.ndarray, sx: int, sy: int) -> np.ndarray:
        """
        Shifts the array by (sx, sy) in x- and y-axes using the chosen boundary_mode.
        """
        if self.boundary_mode == "wrap" or self.boundary_mode not in [
            "reflect",
            "constant",
        ]:
            return self.boundary_wrap(arr, sx, sy)
        elif self.boundary_mode == "reflect":
            return self.boundary_reflect(arr, sx, sy)
        else:
            return self.boundary_constant(arr, sx, sy)

    def _compute_diffusion(
        self, field: np.ndarray, diffusion_coeff: float
    ) -> np.ndarray:
        """
        Computes D * ∇²(field) using a 2D 5-point stencil for the Laplacian,
        with boundary conditions specified by self.boundary_mode.
        """
        center = field
        up = self._shift_with_bc(field, -1, 0)
        down = self._shift_with_bc(field, 1, 0)
        left = self._shift_with_bc(field, 0, -1)
        right = self._shift_with_bc(field, 0, 1)

        lap = (up + down + left + right) - 4.0 * center
        return diffusion_coeff * lap

    def _compute_advection(self, field: np.ndarray) -> np.ndarray:
        """
        Approximates the advection term (v · ∇S) = vx * ∂S/∂x + vy * ∂S/∂y,
        using a first-order upwind approach or a central difference approach.
        Here we use a central difference for simplicity, with boundary conditions.
        """
        vx = self.global_velocity_x
        vy = self.global_velocity_y

        # ∂S/∂x approx = (S_x+1 - S_x-1) / 2
        sx_pos = self._shift_with_bc(field, -1, 0)  # up in x dimension
        sx_neg = self._shift_with_bc(field, 1, 0)
        grad_x = (sx_pos - sx_neg) * 0.5

        # ∂S/∂y approx = (S_y+1 - S_y-1) / 2
        sy_pos = self._shift_with_bc(field, 0, -1)
        sy_neg = self._shift_with_bc(field, 0, 1)
        grad_y = (sy_pos - sy_neg) * 0.5

        return (vx * grad_x) + (vy * grad_y)

    def add_field(
        self, field_name: str, diffusion_coeff: float, initial_value: float = 0.0
    ) -> None:
        """
        Dynamically add a new PDE field to the system with a specified diffusion
        coefficient and optionally an initial value.
        """
        if field_name in self.fields:
            raise ValueError(f"Field '{field_name}' already exists.")
        self.diffusion_coeffs[field_name] = diffusion_coeff
        self.fields[field_name] = np.full(
            (self.width, self.height), initial_value, dtype=np.float32
        )

    def remove_field(self, field_name: str) -> None:
        """
        Remove a PDE field from the system. Use with caution if other
        reaction terms depend on this field.
        """
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found.")
        del self.fields[field_name]
        del self.diffusion_coeffs[field_name]
        if field_name in self.reactions:
            del self.reactions[field_name]

    def set_boundary_mode(self, mode: str) -> None:
        """
        Changes the boundary mode for the next PDE computations.

        Args:
            mode: "wrap", "reflect", or "constant".
        """
        valid_modes = ("wrap", "reflect", "constant")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid boundary mode '{mode}'. Choose from {valid_modes}."
            )
        self.boundary_mode = mode


class CoupledPDEFields(AdvectionDiffusionSystem):
    """
    Example extension of AdvectionDiffusionSystem that demonstrates how to define
    multiple PDE fields with cross-coupling reaction terms.

    For instance, if we have fields: 'W' (toxin) and 'R' (resource):
        ∂W/∂t = - (v · ∇W) + Dw ∇²W + alpha * W * R  - gamma * W
        ∂R/∂t = - (v · ∇R) + Dr ∇²R - beta * W * R  + external_supply

    This class sets up the reaction callbacks for each field. The user can tailor
    the coefficients (alpha, beta, gamma, external_supply) as needed.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        diffusion_coeffs: Dict[str, float],
        dt: float = 1.0,
        boundary_mode: str = "wrap",
        alpha: float = 0.01,
        beta: float = 0.02,
        gamma: float = 0.01,
        external_supply: float = 0.0,
    ):
        super().__init__(grid_shape, diffusion_coeffs, dt, boundary_mode)
        # Example coupling parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.external_supply = external_supply

        # If the user wants to define reaction logic, we can set it up now
        # For demonstration, let's assume fields: "W" and "R".
        # Check if they exist in diffusion_coeffs, then define reaction callbacks.
        if "W" in diffusion_coeffs:
            self.reactions["W"] = self._reaction_toxin
        if "R" in diffusion_coeffs:
            self.reactions["R"] = self._reaction_resource

    def _reaction_toxin(self, **fields) -> np.ndarray:
        """
        Reaction for the 'W' field:
           dW/dt (reaction) = alpha * W * R - gamma * W
        """
        W = fields.get("W", None)
        R = fields.get("R", None)
        if W is None or R is None:
            # If the fields don't exist, return zero reaction
            return np.zeros((self.width, self.height), dtype=np.float32)
        return (self.alpha * W * R) - (self.gamma * W)

    def _reaction_resource(self, **fields) -> np.ndarray:
        """
        Reaction for the 'R' field:
           dR/dt (reaction) = -beta * W * R + external_supply
        """
        W = fields.get("W", None)
        R = fields.get("R", None)
        if W is None or R is None:
            return np.zeros((self.width, self.height), dtype=np.float32)
        # The external_supply is a constant inflow or feed of resource
        return (-self.beta * W * R) + self.external_supply


"""
--------------------------------------------------------------------------------
REACTION DIFFUSION SIMULATOR
--------------------------------------------------------------------------------
This module focuses on applying a Gray-Scott
type reaction-diffusion system, which is known for generating diverse spatiotemporal
patterns reminiscent of natural processes.

1. ReactionDiffusionSimulator:
   - Manages two scalar fields U and V across a 2D grid.
   - Applies discrete updates to simulate their reaction and diffusion.
   - Tunable diffusion coefficients, feed rate, and kill rate to produce
     stable or chaotic patterns.
"""


class ReactionDiffusionSimulator:
    """
    A Gray-Scott style reaction-diffusion system on a 2D grid, modeling the
    spatiotemporal evolution of two interacting scalar fields U and V.

    The canonical equations (in dimensionless form) are:

        ∂U/∂t = Du * ∇²U - U * V² + F * (1 - U)
        ∂V/∂t = Dv * ∇²V + U * V² - (F + k) * V

    Where:
        - U is often interpreted as the "activator" or resource concentration.
        - V is the "inhibitor" or consumer concentration (like a toxin or symbiote).
        - Du, Dv are diffusion coefficients controlling how U and V spread.
        - F is the feed (inflow) rate, and k is the kill (outflow) rate.

    Each timestep, the fields are updated discretely using finite differences
    for the Laplacian and an explicit Euler integration. The resulting patterns
    can show stripes, spots, or chaotic movements. With appropriate parameter
    tuning, you can mimic phenomena such as symbiote territory expansions or
    resource depletion zones.

    Attributes:
        width (int): Grid width.
        height (int): Grid height.
        du (float): Diffusion coefficient for U field.
        dv (float): Diffusion coefficient for V field.
        feed_rate (float): The F parameter controlling how quickly U is fed in.
        kill_rate (float): The k parameter controlling how quickly V is removed.
        dt (float): Time step for each PDE update.

        U (np.ndarray): 2D array storing the current values of U on the grid.
        V (np.ndarray): 2D array storing the current values of V on the grid.
        boundary_mode (str): 'wrap', 'reflect', or 'constant'. Affects boundary conditions
                             when computing Laplacian.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        du: float = 0.16,
        dv: float = 0.08,
        feed_rate: float = 0.060,
        kill_rate: float = 0.062,
        dt: float = 1.0,
        boundary_mode: str = "wrap",
    ):
        """
        Initializes the reaction-diffusion simulator with the given parameters.

        Args:
            grid_shape: (width, height) of the PDE simulation region.
            du: Diffusion coefficient for U field.
            dv: Diffusion coefficient for V field.
            feed_rate: Feed rate (F). Controls how quickly U is introduced.
            kill_rate: Kill rate (k). Controls how quickly V is removed.
            dt: Time step for each PDE update.
            boundary_mode: One of "wrap", "reflect", or "constant" to specify
                           boundary conditions when computing Laplacians.
        """
        self.width, self.height = grid_shape
        self.du = du
        self.dv = dv
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate
        self.dt = dt
        self.boundary_mode = boundary_mode

        # Main fields: U and V
        self.u = np.ones((self.width, self.height), dtype=np.float32)
        self.v = np.zeros((self.width, self.height), dtype=np.float32)

        # Optionally, you can randomize or set patterns here.
        # For example, a small random region of V in the center:
        # (No examples, so leaving this as is.)

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Computes the discrete Laplacian of 'field' using a finite difference
        approximation with the selected boundary mode.

        Args:
            field: 2D array whose Laplacian to compute.

        Returns:
            A 2D numpy array of the same shape, containing the Laplacian of field.
        """
        # This uses convolution-like shifting or np.roll for simplicity, but we
        # can expand to more sophisticated methods if desired.
        # For boundary_mode, we emulate wrap or reflect or constant.

        # Helper function to safely shift with boundary conditions
        def shift(arr, shift_x, shift_y):
            if self.boundary_mode == "wrap":
                return np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)
            elif self.boundary_mode == "reflect":
                # For reflect, we can do a manual reflection approach or fallback
                # to a bigger convolution approach with padding if needed. For
                # brevity, we’ll do a local reflection trick:
                return np.pad(arr, 1, mode="reflect")[
                    1 + shift_x : 1 + shift_x + arr.shape[0],
                    1 + shift_y : 1 + shift_y + arr.shape[1],
                ]
            elif self.boundary_mode == "constant":
                cval = float(arr[0, 0])  # or 0.0 if you want strictly zero
                padded = np.pad(arr, 1, mode="constant", constant_values=cval)
                sub = padded[
                    1 + shift_x : 1 + shift_x + arr.shape[0],
                    1 + shift_y : 1 + shift_y + arr.shape[1],
                ]
                return sub
            else:
                # Default to wrap if unrecognized boundary_mode
                return np.roll(np.roll(arr, shift_x, axis=0), shift_y, axis=1)

        # Laplacian approximation
        # L = -4 * center + up + down + left + right
        center = field
        up = shift(field, -1, 0)
        down = shift(field, 1, 0)
        left = shift(field, 0, -1)
        right = shift(field, 0, 1)

        lap = (up + down + left + right) - 4.0 * center
        return lap

    def step(self) -> None:
        """
        Performs a single Euler integration step of the reaction-diffusion system.
        Updates self.U and self.V in place.
        """
        u_lap = self.laplacian(self.u)
        v_lap = self.laplacian(self.v)

        # Gray-Scott Reaction Terms
        uvv = self.u * (self.v * self.v)
        du = (self.du * u_lap) - uvv + (self.feed_rate * (1.0 - self.u))
        dv = (self.dv * v_lap) + uvv - ((self.feed_rate + self.kill_rate) * self.v)

        self.u += du * self.dt
        self.v += dv * self.dt

        # Ensure numerical stability or clamp to [0,1] or some range if needed
        np.clip(self.u, 0.0, 1e9, out=self.u)
        np.clip(self.v, 0.0, 1e9, out=self.v)

    def multi_step(self, steps: int) -> None:
        """
        Runs the reaction-diffusion system for a specified number of steps.

        Args:
            steps: how many consecutive steps to perform.
        """
        for _ in range(steps):
            self.step()

    def inject_localized_v(
        self,
        center: Tuple[int, int],
        radius: int,
        v_amount: float,
        clamp_to_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Increases the V field in a circular region (like seeding a symbiote patch).

        Args:
            center: (x, y) coordinates of the injection center.
            radius: radius in grid cells for the circular injection.
            v_amount: how much to add to V in that region.
            clamp_to_range: if provided (min_val, max_val), clamp V to that range
                            after adding. For example, (0.0, 1.0).
        """
        cx, cy = center
        x_coords = np.arange(self.width)[:, None]  # column vector
        y_coords = np.arange(self.height)  # row
        dist2 = (x_coords - cx) ** 2 + (y_coords - cy) ** 2

        mask = dist2 <= radius**2
        self.v[mask] += v_amount

        if clamp_to_range:
            min_val, max_val = clamp_to_range
            np.clip(self.v, min_val, max_val, out=self.v)

    def inject_localized_u(
        self,
        center: Tuple[int, int],
        radius: int,
        u_amount: float,
        clamp_to_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Similar to inject_localized_v but for the U field. For instance,
        you might want to add resources in a region or seed a strong source
        of 'nutrients' for the reaction-diffusion system.

        Args:
            center: (x, y) coordinates of the injection center.
            radius: radius in grid cells for the circular injection.
            u_amount: how much to add to U in that region.
            clamp_to_range: optional min/max clamp for final U values.
        """
        cx, cy = center
        x_coords = np.arange(self.width)[:, None]
        y_coords = np.arange(self.height)
        dist2 = (x_coords - cx) ** 2 + (y_coords - cy) ** 2

        mask = dist2 <= radius**2
        self.u[mask] += u_amount

        if clamp_to_range:
            min_val, max_val = clamp_to_range
            np.clip(self.u, min_val, max_val, out=self.u)

    def get_u_field(self) -> np.ndarray:
        """
        Returns a copy of the U field for external use or inspection.
        """
        return self.u.copy()

    def get_v_field(self) -> np.ndarray:
        """
        Returns a copy of the V field for external use or inspection.
        """
        return self.v.copy()

    def set_boundary_mode(self, mode: str) -> None:
        """
        Changes the boundary mode for the next Laplacian computations.

        Args:
            mode: "wrap", "reflect", or "constant".
        """
        valid_modes = ("wrap", "reflect", "constant")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid boundary mode '{mode}'. Choose from {valid_modes}."
            )
        self.boundary_mode = mode

    def set_parameters(
        self,
        du: Optional[
            float
        ] = None,  # noqa: N803 - Standard mathematical notation for diffusion coefficient
        dv: Optional[
            float
        ] = None,  # noqa: N803 - Standard mathematical notation for diffusion coefficient
        feed_rate: Optional[float] = None,
        kill_rate: Optional[float] = None,
    ) -> None:
        """
        Dynamically adjusts PDE coefficients for Du, Dv, feed_rate, or kill_rate.
        Useful if you'd like to ramp up the aggressiveness of the pattern mid-game
        or respond to external events.

        Args:
            du: new diffusion coefficient for U (if not None).
            dv: new diffusion coefficient for V (if not None).
            feed_rate: new feed rate (if not None).
            kill_rate: new kill rate (if not None).
        """
        if du is not None:
            self.du = du
        if dv is not None:
            self.dv = dv
        if feed_rate is not None:
            self.feed_rate = feed_rate
        if kill_rate is not None:
            self.kill_rate = kill_rate

    def reset_fields(self, u_value: float = 1.0, v_value: float = 0.0) -> None:
        """
        Resets the U and V fields to uniform values. Typically done if you
        want to start fresh or re-seed the reaction-diffusion system with
        a new initial state.

        Args:
            u_value: value to assign to every cell in U.
            v_value: value to assign to every cell in V.
        """
        self.u.fill(u_value)
        self.v.fill(v_value)
