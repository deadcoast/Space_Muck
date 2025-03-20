"""
ml_aggression_model.py

Integrates machine learning predictions from the PlayerFeedingBehaviorModel
with symbiote aggression calculations. This creates a more intelligent
response system where symbiotes can anticipate player behavior and plan
their strategies accordingly.
"""

import numpy as np
from typing import Dict

from src.algorithms.economy_decision import PlayerFeedingBehaviorModel


class PredictiveAggressionModel:
    """
    Uses ML-based predictions to make informed decisions about
    symbiote aggression levels based on anticipated player behavior.
    
    This model can:
    1) Adjust aggression based on predicted feeding probability
    2) Track historical player behaviors to improve predictions
    3) Apply different aggression strategies for different player clusters
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        aggression_baseline: float = 0.5,
        aggression_max: float = 1.0,
        feeding_threshold: float = 0.4
    ):
        """
        Initialize the predictive aggression model.
        
        Args:
            n_clusters: Number of player behavior clusters
            aggression_baseline: Default aggression level
            aggression_max: Maximum possible aggression
            feeding_threshold: Probability threshold below which to increase aggression
        """
        self.behavior_model = PlayerFeedingBehaviorModel(n_clusters=n_clusters)
        self.aggression_baseline = aggression_baseline
        self.aggression_max = aggression_max
        self.feeding_threshold = feeding_threshold
        
        # Cluster-specific aggression multipliers
        # Different player types may receive different responses
        self.cluster_aggression_multipliers = np.ones(n_clusters)
        
        # Track feeding history
        self.player_feeding_history = {}
        
    def record_player_data(self, player_id: str, features: np.ndarray, fed: bool) -> None:
        """
        Record player behavioral data for future prediction.
        
        Args:
            player_id: Unique identifier for the player
            features: Feature vector representing player state
            fed: Whether the player fed the symbiote in this state
        """
        if player_id not in self.player_feeding_history:
            self.player_feeding_history[player_id] = {
                'features': [],
                'fed': []
            }
        
        self.player_feeding_history[player_id]['features'].append(features)
        self.player_feeding_history[player_id]['fed'].append(1 if fed else 0)
        
    def train_model(self) -> None:
        """
        Train the feeding behavior model using collected player data.
        """
        # Combine all player data for training
        all_features = []
        all_labels = []
        
        for player_data in self.player_feeding_history.values():
            if player_data['features'] and player_data['fed']:
                all_features.extend(player_data['features'])
                all_labels.extend(player_data['fed'])
        
        if not all_features:
            return  # Not enough data to train
            
        # Convert to numpy arrays
        x_train = np.array(all_features)
        y_train = np.array(all_labels)
        
        # Perform clustering to identify player types
        self.behavior_model.cluster_players(x_train)
        
        # Train feeding prediction model
        self.behavior_model.train_feed_prediction(x_train, y_train)
        
    def set_cluster_aggression_profile(self, cluster_id: int, multiplier: float) -> None:
        """
        Set a custom aggression multiplier for a specific player cluster.
        
        Args:
            cluster_id: Cluster ID to modify
            multiplier: Aggression multiplier for this cluster
        """
        if 0 <= cluster_id < len(self.cluster_aggression_multipliers):
            self.cluster_aggression_multipliers[cluster_id] = multiplier
            
    def predict_aggression(self, current_features: np.ndarray) -> float:
        """
        Calculate appropriate aggression level based on predicted feeding behavior.
        
        Args:
            current_features: Current state features vector
            
        Returns:
            Recommended aggression level between 0 and aggression_max
        """
        # Reshape for single sample prediction
        features = current_features.reshape(1, -1)

        # Get probability of feeding
        feed_probability = self.behavior_model.predict_feeding_proba(features)[0]

        # Determine player's cluster
        cluster = self.behavior_model.cluster_labels[0] if self.behavior_model.cluster_labels is not None else 0
        cluster_multiplier = self.cluster_aggression_multipliers[cluster]

        # Higher aggression if low feeding probability, scaled by cluster multiplier
        if feed_probability < self.feeding_threshold:
            # Increase aggression inversely proportional to feeding probability
            aggression_factor = 1.0 + (self.feeding_threshold - feed_probability) / self.feeding_threshold
            return min(
                self.aggression_max,
                self.aggression_baseline * aggression_factor * cluster_multiplier,
            )
        else:
            # Feeding is likely, reduce aggression
            aggression_factor = feed_probability / self.feeding_threshold
            return max(
                0,
                self.aggression_baseline
                * (2.0 - aggression_factor)
                * cluster_multiplier,
            )
    
    def get_player_cluster(self, features: np.ndarray) -> int:
        """
        Determine which behavioral cluster a player belongs to.
        
        Args:
            features: Player state features
            
        Returns:
            Cluster ID for the player
        """
        # Need to reshape for single sample clustering
        features = features.reshape(1, -1)
        return self.behavior_model.cluster_players(features)[0]
    
    def get_cluster_feeding_statistics(self) -> Dict[int, float]:
        """
        Get feeding statistics by cluster.
        
        Returns:
            Dictionary mapping cluster IDs to average feeding rates
        """
        if self.behavior_model.cluster_labels is None:
            return {}
            
        # Get all player data
        all_features = []
        all_labels = []
        
        for player_data in self.player_feeding_history.values():
            if player_data['features'] and player_data['fed']:
                all_features.extend(player_data['features'])
                all_labels.extend(player_data['fed'])
                
        if not all_features:
            return {}
            
        # Convert to numpy arrays
        x_data = np.array(all_features)
        y_data = np.array(all_labels)
        
        # Cluster the data
        clusters = self.behavior_model.cluster_players(x_data)
        
        # Calculate average feeding rate per cluster
        cluster_stats = {}
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            if np.any(mask):
                cluster_stats[int(cluster_id)] = float(np.mean(y_data[mask]))
                
        return cluster_stats
