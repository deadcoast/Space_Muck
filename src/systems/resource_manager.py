"""
Resource Management System: Handles resource tracking, distribution, and operations.

This module provides functionality for managing game resources, including
tracking resource levels, handling resource flows, and managing resource operations.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

# Resource Types and Categories
class ResourceType(Enum):
    ENERGY = auto()
    MATTER = auto()
    FLUID = auto()
    DATA = auto()

# Resource States
class ResourceState(Enum):
    STABLE = auto()
    DEPLETING = auto()
    GROWING = auto()
    CRITICAL = auto()

# Resource Flow Patterns
FLOW_PATTERNS = {
    "direct": "One-to-one resource transfer",
    "split": "One-to-many resource distribution",
    "merge": "Many-to-one resource collection",
    "cycle": "Circular resource flow pattern",
}

# Resource Thresholds
RESOURCE_THRESHOLDS = {
    "critical_low": 0.1,  # 10% of capacity
    "low": 0.25,         # 25% of capacity
    "optimal": 0.75,     # 75% of capacity
    "high": 0.9,         # 90% of capacity
}

@dataclass
class ResourceFlow:
    """Represents a resource flow between source and destination."""
    source_id: str
    dest_id: str
    resource_type: ResourceType
    rate: float
    priority: int = 1
    active: bool = True

class ResourceManager:
    """
    Central manager for handling all resource-related operations.
    """
    
    def __init__(self) -> None:
        """Initialize the resource manager."""
        # Resource tracking
        self.resources: Dict[str, Dict[ResourceType, float]] = {}
        self.capacities: Dict[str, Dict[ResourceType, float]] = {}
        self.states: Dict[str, Dict[ResourceType, ResourceState]] = {}
        
        # Flow management
        self.flows: List[ResourceFlow] = []
        self.flow_history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, amount)
        
        # System state
        self.active = True
        self.paused = False
        self.update_interval = 1.0  # seconds
        self.last_update = 0.0
        
        logging.info("ResourceManager initialized")

    def register_resource_node(
        self,
        node_id: str,
        resource_types: Dict[ResourceType, float],
        capacities: Optional[Dict[ResourceType, float]] = None
    ) -> bool:
        """
        Register a new resource node in the system.

        Args:
            node_id: Unique identifier for the resource node
            resource_types: Initial resource amounts by type
            capacities: Optional capacities by resource type

        Returns:
            bool: True if registration successful
        """
        if node_id in self.resources:
            logging.warning(f"Resource node {node_id} already registered")
            return False

        self.resources[node_id] = resource_types
        self.capacities[node_id] = capacities or {
            rtype: amount * 2 for rtype, amount in resource_types.items()
        }
        self.states[node_id] = {
            rtype: ResourceState.STABLE for rtype in resource_types
        }

        logging.info(f"Registered resource node {node_id}")
        return True

    def remove_resource_node(self, node_id: str) -> bool:
        """
        Remove a resource node from the system.

        Args:
            node_id: ID of node to remove

        Returns:
            bool: True if removal successful
        """
        if node_id not in self.resources:
            logging.warning(f"Resource node {node_id} not found")
            return False

        # Clean up all references
        del self.resources[node_id]
        del self.capacities[node_id]
        del self.states[node_id]

        # Remove associated flows
        self.flows = [
            flow for flow in self.flows
            if flow.source_id != node_id and flow.dest_id != node_id
        ]

        logging.info(f"Removed resource node {node_id}")
        return True

    def add_resource_flow(
        self,
        source_id: str,
        dest_id: str,
        resource_type: ResourceType,
        rate: float,
        priority: int = 1
    ) -> bool:
        """
        Add a new resource flow between nodes.

        Args:
            source_id: Source node ID
            dest_id: Destination node ID
            resource_type: Type of resource to flow
            rate: Flow rate (units per second)
            priority: Flow priority (higher = more important)

        Returns:
            bool: True if flow added successfully
        """
        # Validate nodes exist
        if source_id not in self.resources or dest_id not in self.resources:
            logging.error(f"Invalid node ID in flow: {source_id} -> {dest_id}")
            return False

        # Check resource type availability
        if resource_type not in self.resources[source_id]:
            logging.error(f"Resource type {resource_type} not available at source {source_id}")
            return False

        flow = ResourceFlow(source_id, dest_id, resource_type, rate, priority)
        self.flows.append(flow)
        
        logging.info(f"Added resource flow: {source_id} -> {dest_id} ({resource_type})")
        return True

    def remove_resource_flow(
        self,
        source_id: str,
        dest_id: str,
        resource_type: ResourceType
    ) -> bool:
        """
        Remove a resource flow.

        Args:
            source_id: Source node ID
            dest_id: Destination node ID
            resource_type: Type of resource flowing

        Returns:
            bool: True if flow removed successfully
        """
        for flow in self.flows:
            if (flow.source_id == source_id and 
                flow.dest_id == dest_id and 
                flow.resource_type == resource_type):
                self.flows.remove(flow)
                logging.info(f"Removed resource flow: {source_id} -> {dest_id}")
                return True

        logging.warning(f"Flow not found: {source_id} -> {dest_id}")
        return False

    def update(self, dt: float) -> None:
        """
        Update resource states and process flows.

        Args:
            dt: Time delta since last update
        """
        if not self.active or self.paused:
            return

        self.last_update += dt
        if self.last_update < self.update_interval:
            return

        # Process all active flows
        for flow in self.flows:
            if not flow.active:
                continue

            # Calculate transfer amount
            transfer = flow.rate * self.update_interval
            source = self.resources[flow.source_id]
            dest = self.resources[flow.dest_id]

            # Check if transfer is possible
            available = source.get(flow.resource_type, 0)
            capacity = self.capacities[flow.dest_id].get(flow.resource_type, 0)
            current = dest.get(flow.resource_type, 0)
            
            # Adjust transfer based on constraints
            transfer = min(transfer, available)
            transfer = min(transfer, capacity - current)

            if transfer > 0:
                # Execute transfer
                source[flow.resource_type] -= transfer
                dest[flow.resource_type] = dest.get(flow.resource_type, 0) + transfer

                # Record in history
                flow_id = f"{flow.source_id}->{flow.dest_id}"
                if flow_id not in self.flow_history:
                    self.flow_history[flow_id] = []
                self.flow_history[flow_id].append((self.last_update, transfer))

        # Update resource states
        for node_id, resources in self.resources.items():
            for rtype, amount in resources.items():
                capacity = self.capacities[node_id][rtype]
                ratio = amount / capacity

                # Update state based on thresholds
                if ratio <= RESOURCE_THRESHOLDS["critical_low"]:
                    self.states[node_id][rtype] = ResourceState.CRITICAL
                elif ratio <= RESOURCE_THRESHOLDS["low"]:
                    self.states[node_id][rtype] = ResourceState.DEPLETING
                elif ratio >= RESOURCE_THRESHOLDS["high"]:
                    self.states[node_id][rtype] = ResourceState.GROWING
                else:
                    self.states[node_id][rtype] = ResourceState.STABLE

        self.last_update = 0

    def get_resource_amount(
        self,
        node_id: str,
        resource_type: ResourceType
    ) -> Optional[float]:
        """
        Get the current amount of a resource at a node.

        Args:
            node_id: Node to check
            resource_type: Resource type to check

        Returns:
            float: Current amount or None if not found
        """
        return self.resources.get(node_id, {}).get(resource_type)

    def get_resource_state(
        self,
        node_id: str,
        resource_type: ResourceType
    ) -> Optional[ResourceState]:
        """
        Get the current state of a resource at a node.

        Args:
            node_id: Node to check
            resource_type: Resource type to check

        Returns:
            ResourceState: Current state or None if not found
        """
        return self.states.get(node_id, {}).get(resource_type)

    def pause(self) -> None:
        """Pause resource processing."""
        self.paused = True
        logging.info("ResourceManager paused")

    def resume(self) -> None:
        """Resume resource processing."""
        self.paused = False
        logging.info("ResourceManager resumed")

    def shutdown(self) -> None:
        """Shutdown the resource manager."""
        self.active = False
        logging.info("ResourceManager shut down")
