"""
CrewMember class: Represents a crew member that can be assigned to ship functions.
"""

# Standard library imports
import logging
import random
import uuid

# Local application imports
from typing import Any, Dict, List, Optional

# Third-party library imports


# Skill level constants
SKILL_LEVELS = {
    0: "Untrained",
    1: "Novice",
    2: "Trained",
    3: "Skilled",
    4: "Expert",
    5: "Master",
}

# Station types
STATION_TYPES = [
    "navigation",
    "engineering",
    "weapons",
    "science",
    "medical",
    "command",
]

# Trait effects on different stations
TRAIT_EFFECTS = {
    "focused": {
        "navigation": 0.1,
        "engineering": 0.1,
        "weapons": 0.2,
        "science": 0.2,
        "medical": 0.0,
        "command": 0.0,
    },
    "analytical": {
        "navigation": 0.1,
        "engineering": 0.2,
        "weapons": 0.0,
        "science": 0.2,
        "medical": 0.1,
        "command": 0.0,
    },
    "adaptable": {
        "navigation": 0.1,
        "engineering": 0.1,
        "weapons": 0.1,
        "science": 0.1,
        "medical": 0.1,
        "command": 0.1,
    },
    "intuitive": {
        "navigation": 0.2,
        "engineering": 0.0,
        "weapons": 0.1,
        "science": 0.0,
        "medical": 0.2,
        "command": 0.1,
    },
    "methodical": {
        "navigation": 0.0,
        "engineering": 0.2,
        "weapons": 0.0,
        "science": 0.2,
        "medical": 0.1,
        "command": 0.1,
    },
    "charismatic": {
        "navigation": 0.0,
        "engineering": 0.0,
        "weapons": 0.0,
        "science": 0.0,
        "medical": 0.1,
        "command": 0.3,
    },
}

# Name pools for random generation
FIRST_NAMES = [
    "Alex",
    "Morgan",
    "Taylor",
    "Jordan",
    "Casey",
    "Riley",
    "Quinn",
    "Avery",
    "Skyler",
    "Dakota",
    "Reese",
    "Finley",
    "Harley",
    "Emerson",
    "Phoenix",
    "Kai",
    "Zephyr",
    "Nova",
    "Orion",
    "Vega",
    "Lyra",
    "Atlas",
    "Cygnus",
    "Rigel",
]

LAST_NAMES = [
    "Smith",
    "Chen",
    "Patel",
    "Kim",
    "Nguyen",
    "Garcia",
    "Rodriguez",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Miller",
    "Davis",
    "Wilson",
    "Anderson",
    "Starling",
    "Voidwalker",
    "Nebula",
    "Comet",
    "Stardust",
    "Celestial",
    "Pulsar",
]


class CrewMember:
    """Represents a crew member that can be assigned to ship functions."""

    def __init__(
        self,
        name: Optional[str] = None,
        skills: Optional[Dict[str, int]] = None,
        traits: Optional[List[str]] = None,
        experience: int = 0,
        current_station: Optional[str] = None,
        id: Optional[str] = None,
    ):
        """Initialize a crew member.

        Args:
            name: Crew member's name
            skills: Dictionary of skills and their levels (0-5)
            traits: List of character traits
            experience: Experience points
            current_station: Currently assigned station
            id: Unique identifier
        """
        self.id = id or str(uuid.uuid4())
        self.name = name or self._generate_random_name()

        # Initialize skills (0-5 scale)
        self.skills = {
            "navigation": 0,
            "engineering": 0,
            "weapons": 0,
            "science": 0,
            "medical": 0,
            "command": 0,
        }

        # If skills are provided, update the default skills
        if skills:
            for skill, level in skills.items():
                if skill in self.skills:
                    self.skills[skill] = max(0, min(5, level))  # Clamp to 0-5

        # Generate random skills if none provided
        if not skills:
            self._generate_random_skills()

        # Initialize traits
        self.traits = traits or self._generate_random_traits()

        # Experience and level
        self.experience = experience
        self.level = self._calculate_level()

        # Current assignment
        self.current_station = current_station
        self.fatigue = 0  # 0-100 scale, increases with continuous work
        self.morale = 100  # 0-100 scale, affects performance

        # Status
        self.status = "available"  # available, working, resting, injured, etc.

        logging.info(f"Crew member {self.name} initialized with ID {self.id}")

    @staticmethod
    def _generate_random_name() -> str:
        """Generate a random name for the crew member.

        Returns:
            Random name string
        """
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        return f"{first_name} {last_name}"

    def _generate_random_skills(self) -> None:
        """Generate random skill levels for the crew member."""
        # Determine primary skill
        primary_skill = random.choice(list(self.skills.keys()))

        # Assign skill levels
        for skill in self.skills:
            if skill == primary_skill:
                # Primary skill is higher
                self.skills[skill] = random.randint(2, 3)
            else:
                # Secondary skills are lower
                self.skills[skill] = random.randint(0, 2)

    @staticmethod
    def _generate_random_traits() -> List[str]:
        """Generate random traits for the crew member.

        Returns:
            List of trait strings
        """
        available_traits = list(TRAIT_EFFECTS.keys())
        # Each crew member has 1-2 traits
        num_traits = random.randint(1, 2)
        return random.sample(available_traits, num_traits)

    def _calculate_level(self) -> int:
        """Calculate the crew member's level based on experience.

        Returns:
            Current level
        """
        # Simple level calculation: level = 1 + experience / 1000
        return 1 + self.experience // 1000

    def assign_to_station(self, station: str) -> Dict[str, Any]:
        """Assign the crew member to a station.

        Args:
            station: Station to assign to

        Returns:
            Dict with assignment results
        """
        try:
            if station not in STATION_TYPES and station is not None:
                return {"success": False, "message": f"Invalid station: {station}"}

            old_station = self.current_station
            self.current_station = station
            self.status = "working" if station else "available"

            return {
                "success": True,
                "message": (
                    f"Assigned {self.name} to {station}"
                    if station
                    else f"Unassigned {self.name}"
                ),
                "old_station": old_station,
                "new_station": station,
            }
        except Exception as e:
            logging.error(f"Error assigning crew member to station: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def get_skill_level_name(self, skill: str) -> str:
        """Get the name of the skill level.

        Args:
            skill: Skill to check

        Returns:
            Skill level name
        """
        if skill not in self.skills:
            return "Unknown"

        level = self.skills.get(skill, 0)
        return SKILL_LEVELS.get(level, "Unknown")

    def get_effective_skill(self, station: str) -> float:
        """Calculate the effective skill level for a station, including trait bonuses.

        Args:
            station: Station to calculate effective skill for

        Returns:
            Effective skill level (float)
        """
        if station not in self.skills:
            return 0.0

        base_skill = self.skills.get(station, 0)

        # Apply trait bonuses
        trait_bonus = 0.0
        for trait in self.traits:
            if trait in TRAIT_EFFECTS:
                trait_bonus += TRAIT_EFFECTS[trait].get(station, 0.0)

        # Apply morale and fatigue effects
        morale_factor = self.morale / 100.0  # 0.0 to 1.0
        fatigue_penalty = self.fatigue / 200.0  # 0.0 to 0.5 at max fatigue

        effective_skill = (
            base_skill * (1.0 + trait_bonus) * morale_factor * (1.0 - fatigue_penalty)
        )

        return round(effective_skill, 1)

    def rest(self, hours: int = 8) -> Dict[str, Any]:
        """Rest the crew member to reduce fatigue and increase morale.

        Args:
            hours: Number of hours to rest

        Returns:
            Dict with rest results
        """
        old_fatigue = self.fatigue
        old_morale = self.morale

        # Reduce fatigue
        fatigue_reduction = min(hours * 5, self.fatigue)
        self.fatigue = max(0, self.fatigue - fatigue_reduction)

        # Increase morale
        morale_increase = min(hours * 2, 100 - self.morale)
        self.morale = min(100, self.morale + morale_increase)

        # Update status
        if self.status in ["resting", "injured"]:
            self.status = "available"

        return {
            "success": True,
            "message": f"{self.name} rested for {hours} hours",
            "fatigue_reduced": fatigue_reduction,
            "morale_increased": morale_increase,
            "old_fatigue": old_fatigue,
            "new_fatigue": self.fatigue,
            "old_morale": old_morale,
            "new_morale": self.morale,
        }

    def work_shift(self, hours: int = 8) -> Dict[str, Any]:
        """Work a shift at the assigned station, gaining experience but increasing fatigue.

        Args:
            hours: Number of hours to work

        Returns:
            Dict with work results
        """
        if not self.current_station:
            return {
                "success": False,
                "message": f"{self.name} is not assigned to a station",
            }

        if self.status == "injured":
            return {
                "success": False,
                "message": f"{self.name} is injured and cannot work",
            }

        # Increase fatigue
        old_fatigue = self.fatigue
        fatigue_increase = hours * 5
        self.fatigue = min(100, self.fatigue + fatigue_increase)

        # Decrease morale if overworked
        old_morale = self.morale
        morale_change = 0
        if self.fatigue > 70:
            morale_change = -hours
            self.morale = max(0, self.morale + morale_change)

        # Gain experience
        old_experience = self.experience
        experience_gain = hours * (1 + self.skills.get(self.current_station, 0))
        old_level = self._update_experience_and_level(experience_gain)
        leveled_up = self.level > old_level

        # Update status
        self.status = "exhausted" if self.fatigue > 90 else "working"
        return {
            "success": True,
            "message": f"{self.name} worked at {self.current_station} for {hours} hours",
            "fatigue_increased": fatigue_increase,
            "morale_changed": morale_change,
            "experience_gained": experience_gain,
            "old_fatigue": old_fatigue,
            "new_fatigue": self.fatigue,
            "old_morale": old_morale,
            "new_morale": self.morale,
            "old_experience": old_experience,
            "new_experience": self.experience,
            "old_level": old_level,
            "new_level": self.level,
            "leveled_up": leveled_up,
        }

    def train_skill(self, skill: str, hours: int = 4) -> Dict[str, Any]:
        """Train a specific skill to improve it.

        Args:
            skill: Skill to train
            hours: Number of hours to train

        Returns:
            Dict with training results
        """
        if skill not in self.skills:
            return {"success": False, "message": f"Invalid skill: {skill}"}

        old_skill = self.skills[skill]

        # Check if skill is already at max level
        if old_skill >= 5:
            return {
                "success": False,
                "message": f"{self.name} is already a Master in {skill}",
            }

        # Calculate skill increase based on hours and current level
        # Higher levels require more training
        skill_increase = hours / (10 * (old_skill + 1))

        # Skill levels are integers, so we track partial progress
        # and only increase the level when it reaches the next integer
        new_skill_float = old_skill + skill_increase
        new_skill = int(new_skill_float)

        # Update skill if it increased
        if new_skill > old_skill:
            self.skills[skill] = new_skill
            skill_improved = True
        else:
            skill_improved = False

        # Gain some experience from training
        experience_gain = hours * 5
        old_level = self._update_experience_and_level(experience_gain)
        return {
            "success": True,
            "message": f"{self.name} trained {skill} for {hours} hours",
            "skill": skill,
            "old_skill": old_skill,
            "new_skill": self.skills[skill],
            "skill_improved": skill_improved,
            "experience_gained": experience_gain,
            "old_level": old_level,
            "new_level": self.level,
            "leveled_up": self.level > old_level,
        }

    def _update_experience_and_level(self, experience_gain):
        """Update crew member's experience and recalculate level.

        Args:
            experience_gain: Amount of experience to add

        Returns:
            Previous level before update
        """
        self.experience += experience_gain
        result = self.level
        self.level = self._calculate_level()
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert the crew member to a dictionary for serialization.

        Returns:
            Dict representation of the crew member
        """
        return {
            "id": self.id,
            "name": self.name,
            "skills": self.skills.copy(),
            "traits": self.traits.copy(),
            "experience": self.experience,
            "level": self.level,
            "current_station": self.current_station,
            "fatigue": self.fatigue,
            "morale": self.morale,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrewMember":
        """Create a crew member from a dictionary.

        Args:
            data: Dict representation of a crew member

        Returns:
            CrewMember instance
        """
        return cls(
            name=data.get("name"),
            skills=data.get("skills"),
            traits=data.get("traits"),
            experience=data.get("experience", 0),
            current_station=data.get("current_station"),
            id=data.get("id"),
        )
