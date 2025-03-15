"""
Crew Management System: Manages the player's crew members and their assignments.
"""

# Standard library imports
import logging

# Local application imports
from typing import Any, Dict, List, Optional

# Use absolute imports for consistency
from src.entities.crew_member import STATION_TYPES, CrewMember

# Third-party library imports


class CrewManagementSystem:
    """System for managing crew members and their assignments to ship stations."""

    def __init__(self):
        """Initialize the crew management system."""
        self.crew_members = {}  # Dict of crew_id: CrewMember
        self.station_assignments = {station: [] for station in STATION_TYPES}
        self.max_crew_size = 10  # Maximum number of crew members
        self.max_per_station = {  # Maximum crew per station
            "navigation": 2,
            "engineering": 3,
            "weapons": 2,
            "science": 2,
            "medical": 2,
            "command": 1,
        }

        logging.info("Crew Management System initialized")

    def add_crew_member(self, crew_member: CrewMember) -> Dict[str, Any]:
        """Add a crew member to the roster.

        Args:
            crew_member: CrewMember to add

        Returns:
            Dict with operation results
        """
        try:
            if len(self.crew_members) >= self.max_crew_size:
                return {
                    "success": False,
                    "message": f"Crew roster full (max: {self.max_crew_size})",
                }

            self.crew_members[crew_member.id] = crew_member

            return {
                "success": True,
                "message": f"Added {crew_member.name} to the crew roster",
                "crew_member": crew_member,
            }
        except Exception as e:
            logging.error(f"Error adding crew member: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def remove_crew_member(self, crew_id: str) -> Dict[str, Any]:
        """Remove a crew member from the roster.

        Args:
            crew_id: ID of the crew member to remove

        Returns:
            Dict with operation results
        """
        try:
            if crew_id not in self.crew_members:
                return {
                    "success": False,
                    "message": f"Crew member with ID {crew_id} not found",
                }

            crew_member = self.crew_members[crew_id]

            # Unassign from any station
            if crew_member.current_station:
                self._unassign_from_station(crew_id, crew_member.current_station)

            # Remove from roster
            del self.crew_members[crew_id]

            return {
                "success": True,
                "message": f"Removed {crew_member.name} from the crew roster",
                "crew_member": crew_member,
            }
        except Exception as e:
            logging.error(f"Error removing crew member: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def assign_to_station(self, crew_id: str, station: str) -> Dict[str, Any]:
        """Assign a crew member to a station.

        Args:
            crew_id: ID of the crew member to assign
            station: Station to assign to

        Returns:
            Dict with operation results
        """
        try:
            if crew_id not in self.crew_members:
                return {
                    "success": False,
                    "message": f"Crew member with ID {crew_id} not found",
                }

            if station not in STATION_TYPES:
                return {"success": False, "message": f"Invalid station: {station}"}

            crew_member = self.crew_members[crew_id]

            # Check if station is full
            if len(self.station_assignments[station]) >= self.max_per_station[station]:
                return {
                    "success": False,
                    "message": f"Station {station} is full (max: {self.max_per_station[station]})",
                }

            # Unassign from current station if assigned
            if crew_member.current_station:
                self._unassign_from_station(crew_id, crew_member.current_station)

            # Assign to new station
            result = crew_member.assign_to_station(station)
            if result["success"]:
                self.station_assignments[station].append(crew_id)

            return result
        except Exception as e:
            logging.error(f"Error assigning crew member to station: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def unassign_from_station(self, crew_id: str) -> Dict[str, Any]:
        """Unassign a crew member from their current station.

        Args:
            crew_id: ID of the crew member to unassign

        Returns:
            Dict with operation results
        """
        try:
            if crew_id not in self.crew_members:
                return {
                    "success": False,
                    "message": f"Crew member with ID {crew_id} not found",
                }

            crew_member = self.crew_members[crew_id]

            if not crew_member.current_station:
                return {
                    "success": False,
                    "message": f"{crew_member.name} is not assigned to a station",
                }

            station = crew_member.current_station
            result = self._unassign_from_station(crew_id, station)

            # Update crew member's status
            crew_member.assign_to_station(None)

            return result
        except Exception as e:
            logging.error(f"Error unassigning crew member: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

    def _unassign_from_station(self, crew_id: str, station: str) -> Dict[str, Any]:
        """Internal method to unassign a crew member from a station.

        Args:
            crew_id: ID of the crew member to unassign
            station: Station to unassign from

        Returns:
            Dict with operation results
        """
        if station not in self.station_assignments:
            return {"success": False, "message": f"Invalid station: {station}"}

        if crew_id not in self.station_assignments[station]:
            return {
                "success": False,
                "message": f"Crew member not assigned to {station}",
            }

        crew_member = self.crew_members[crew_id]
        self.station_assignments[station].remove(crew_id)

        return {
            "success": True,
            "message": f"Unassigned {crew_member.name} from {station}",
            "crew_member": crew_member,
            "station": station,
        }

    def get_crew_member(self, crew_id: str) -> Optional[CrewMember]:
        """Get a crew member by ID.

        Args:
            crew_id: ID of the crew member to get

        Returns:
            CrewMember or None if not found
        """
        return self.crew_members.get(crew_id)

    def get_all_crew_members(self) -> List[CrewMember]:
        """Get all crew members.

        Returns:
            List of all CrewMember objects
        """
        return list(self.crew_members.values())

    def get_station_crew(self, station: str) -> List[CrewMember]:
        """Get all crew members assigned to a station.

        Args:
            station: Station to get crew for

        Returns:
            List of CrewMember objects assigned to the station
        """
        if station not in self.station_assignments:
            return []

        return [
            self.crew_members[crew_id] for crew_id in self.station_assignments[station]
        ]

    def get_station_efficiency(self, station: str) -> float:
        """Calculate the efficiency of a station based on assigned crew.

        Args:
            station: Station to calculate efficiency for

        Returns:
            Efficiency value (0.0-5.0)
        """
        if station not in STATION_TYPES:
            return 0.0

        crew = self.get_station_crew(station)

        if not crew:
            return 0.0

        # Calculate average effective skill
        total_skill = sum(member.get_effective_skill(station) for member in crew)
        avg_skill = total_skill / len(crew)

        # Apply diminishing returns for multiple crew members
        # More crew is better, but with diminishing returns
        crew_bonus = min(1.0, (len(crew) / self.max_per_station[station]) * 0.5)

        return avg_skill * (1.0 + crew_bonus)

    def get_all_station_efficiencies(self) -> Dict[str, float]:
        """Get efficiency values for all stations.

        Returns:
            Dict of station: efficiency
        """
        return {
            station: self.get_station_efficiency(station) for station in STATION_TYPES
        }

    def update_crew_status(self, hours_passed: float = 1.0) -> Dict[str, Any]:
        """Update crew status based on time passed.

        Args:
            hours_passed: Number of hours passed

        Returns:
            Dict with update results
        """
        results = {
            "fatigue_increased": [],
            "morale_decreased": [],
            "experience_gained": [],
            "level_ups": [],
        }

        for crew_id, crew_member in self.crew_members.items():
            self._process_crew_work_shift(crew_id, crew_member, hours_passed, results)

        return results

    def _process_crew_work_shift(
        self,
        crew_id: str,
        crew_member: CrewMember,
        hours_passed: float,
        results: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Process a single crew member's work shift and update results.

        Args:
            crew_id: ID of the crew member
            crew_member: The crew member object
            hours_passed: Number of hours passed
            results: Results dictionary to update
        """
        # Skip crew members who aren't working or are injured
        if not crew_member.current_station or crew_member.status == "injured":
            return

        # Simulate work shift
        work_result = crew_member.work_shift(hours_passed)

        # Track results
        self._track_fatigue_increase(crew_id, crew_member, work_result, results)
        self._track_morale_decrease(crew_id, crew_member, work_result, results)
        self._track_experience_gain(crew_id, crew_member, work_result, results)
        self._track_level_up(crew_id, crew_member, work_result, results)

    def _track_fatigue_increase(
        self,
        crew_id: str,
        crew_member: CrewMember,
        work_result: Dict[str, Any],
        results: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Track fatigue increases in results."""
        if work_result["fatigue_increased"] > 0:
            results["fatigue_increased"].append(
                {
                    "crew_id": crew_id,
                    "name": crew_member.name,
                    "amount": work_result["fatigue_increased"],
                }
            )

    def _track_morale_decrease(
        self,
        crew_id: str,
        crew_member: CrewMember,
        work_result: Dict[str, Any],
        results: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Track morale decreases in results."""
        if work_result.get("morale_changed", 0) < 0:
            results["morale_decreased"].append(
                {
                    "crew_id": crew_id,
                    "name": crew_member.name,
                    "amount": abs(work_result["morale_changed"]),
                }
            )

    def _track_experience_gain(
        self,
        crew_id: str,
        crew_member: CrewMember,
        work_result: Dict[str, Any],
        results: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Track experience gains in results."""
        if work_result["experience_gained"] > 0:
            results["experience_gained"].append(
                {
                    "crew_id": crew_id,
                    "name": crew_member.name,
                    "amount": work_result["experience_gained"],
                }
            )

    def _track_level_up(
        self,
        crew_id: str,
        crew_member: CrewMember,
        work_result: Dict[str, Any],
        results: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Track level ups in results."""
        if work_result["leveled_up"]:
            results["level_ups"].append(
                {
                    "crew_id": crew_id,
                    "name": crew_member.name,
                    "new_level": crew_member.level,
                }
            )

    def rest_crew(
        self, crew_id: Optional[str] = None, hours: int = 8
    ) -> Dict[str, Any]:
        """Rest a crew member or all unassigned crew.

        Args:
            crew_id: ID of specific crew member to rest, or None for all unassigned
            hours: Number of hours to rest

        Returns:
            Dict with rest results
        """
        results = {"rested_crew": [], "fatigue_reduced": [], "morale_increased": []}

        if crew_id:
            # Rest specific crew member
            return self._rest_specific_crew(crew_id, hours, results)
        else:
            # Rest all unassigned crew
            self._rest_all_unassigned_crew(hours, results)

        return results

    def _rest_specific_crew(
        self, crew_id: str, hours: int, results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Rest a specific crew member.

        Args:
            crew_id: ID of the crew member to rest
            hours: Number of hours to rest
            results: Results dictionary to update

        Returns:
            Updated results dictionary
        """
        if crew_id not in self.crew_members:
            return {
                "success": False,
                "message": f"Crew member with ID {crew_id} not found",
            }

        crew_member = self.crew_members[crew_id]

        # Unassign if assigned
        if crew_member.current_station:
            self.unassign_from_station(crew_id)

        # Rest
        rest_result = crew_member.rest(hours)

        if rest_result["success"]:
            self._update_rest_results(results, crew_id, crew_member, rest_result)

        return results

    def _rest_all_unassigned_crew(
        self, hours: int, results: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Rest all unassigned crew members.

        Args:
            hours: Number of hours to rest
            results: Results dictionary to update
        """
        for crew_id, crew_member in self.crew_members.items():
            if not crew_member.current_station or crew_member.status == "exhausted":
                # Rest
                rest_result = crew_member.rest(hours)

                if rest_result["success"]:
                    self._update_rest_results(
                        results, crew_id, crew_member, rest_result
                    )

    def _update_rest_results(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        crew_id: str,
        crew_member: CrewMember,
        rest_result: Dict[str, Any],
    ) -> None:
        """Update results with rest outcome for a crew member.

        Args:
            results: Results dictionary to update
            crew_id: ID of the crew member
            crew_member: The crew member object
            rest_result: Result of the rest operation
        """
        results["rested_crew"].append({"crew_id": crew_id, "name": crew_member.name})

        results["fatigue_reduced"].append(
            {
                "crew_id": crew_id,
                "name": crew_member.name,
                "amount": rest_result["fatigue_reduced"],
            }
        )

        results["morale_increased"].append(
            {
                "crew_id": crew_id,
                "name": crew_member.name,
                "amount": rest_result["morale_increased"],
            }
        )

    def train_crew(self, crew_id: str, skill: str, hours: int = 4) -> Dict[str, Any]:
        """Train a crew member in a specific skill.

        Args:
            crew_id: ID of the crew member to train
            skill: Skill to train
            hours: Number of hours to train

        Returns:
            Dict with training results
        """
        if crew_id not in self.crew_members:
            return {
                "success": False,
                "message": f"Crew member with ID {crew_id} not found",
            }

        crew_member = self.crew_members[crew_id]

        # Unassign if assigned
        if crew_member.current_station:
            self.unassign_from_station(crew_id)

        # Train
        return crew_member.train_skill(skill, hours)

    def recruit_random_crew(self, count: int = 1) -> Dict[str, Any]:
        """Recruit random crew members.

        Args:
            count: Number of crew members to recruit

        Returns:
            Dict with recruitment results
        """
        results = {"success": True, "recruited": [], "message": ""}

        # Check if we can recruit more crew
        available_slots = self.max_crew_size - len(self.crew_members)
        if available_slots <= 0:
            return {
                "success": False,
                "message": f"Crew roster full (max: {self.max_crew_size})",
                "recruited": [],
            }

        # Adjust count if needed
        count = min(count, available_slots)

        # Recruit crew
        for _ in range(count):
            new_crew = CrewMember()
            add_result = self.add_crew_member(new_crew)

            if add_result["success"]:
                results["recruited"].append(new_crew)

        # Set message
        if results["recruited"]:
            names = ", ".join(crew.name for crew in results["recruited"])
            results["message"] = (
                f"Recruited {len(results['recruited'])} new crew members: {names}"
            )
        else:
            results["success"] = False
            results["message"] = "Failed to recruit any crew members"

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert the crew management system to a dictionary for serialization.

        Returns:
            Dict representation of the crew management system
        """
        return {
            "crew_members": {
                crew_id: crew.to_dict() for crew_id, crew in self.crew_members.items()
            },
            "station_assignments": self.station_assignments.copy(),
            "max_crew_size": self.max_crew_size,
            "max_per_station": self.max_per_station.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrewManagementSystem":
        """Create a crew management system from a dictionary.

        Args:
            data: Dict representation of a crew management system

        Returns:
            CrewManagementSystem instance
        """
        system = cls()

        # Load crew members
        crew_data = data.get("crew_members", {})
        for crew_id, crew_dict in crew_data.items():
            system.crew_members[crew_id] = CrewMember.from_dict(crew_dict)

        # Load station assignments
        system.station_assignments = data.get(
            "station_assignments", {station: [] for station in STATION_TYPES}
        )

        # Load other attributes
        system.max_crew_size = data.get("max_crew_size", 10)
        system.max_per_station = data.get("max_per_station", system.max_per_station)

        return system
