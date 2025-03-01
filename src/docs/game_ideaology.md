# Game Logic

1. The Symbiote Races(cells) are Space Monsters who eat asteroids.
2. The user plays a human miner who starts in a single mining space ship.
3. The Symbiote Races should behave similar to The Game of Life Cells, where it's Random. BUT, in this game, we introduce a feeding and currency system to accellerate their growth, but also keep them from eating the mining ships.
4. We need to introduce mathematics to increase their growth or their mutations with the Currency system (mining asteroids). Mining Asteroids should provide minerals, and the user should have to either:
   (a) Sell the Minerals to Buy Mining Ship Upgrades or Fleet Upgrades, Automated Mining drones, etc.
   (b) Feed the Mined Minerals to the Space Monster Symbiote Races to keep them away from his ship so he may grow his economy to outpace the Symbiote Space Monster Races Mutations.
5. The user should feel like he is both controlling the miner, and the space monster symbiotes by feeding them minerals to keep them away.
6. The Space Symbiote Monster Races should have a dedicated and sophisticated algorithm to run the mutations, just like The Game Of Life can produce very unique and obscure results, this game should too. But the players choices should be able to accelerate it.

## Game Flow and Pace

1. The player makes money mining minerals from asteroids, the player can choose to invest in their mining ships and fleet by SELLING the minerals they mine. But he must be wise with his money.
2. The Symbiote Races eat minerals from asteroids, SO, if the player does not have enough money to feed the symbiote random mutating beings, they will eat the players mining ships.
3. It should be a struggle of spending enough money to build up the mining fleet and the fleet out growing the randomly mutating Symbiote Races.
4. The Symbiote races should mutate, eat asteroids automatically, and mutate on their own just like in The Game of Life. This creates the challenge to outgrow the Symbiotes hunger and mutations.
5. The Catch is, the player can keep them away from eating their mining ships or destroying them, by feeding them minerals they have mined from asteroids. The Selling of minerals should be a manual choice for money, or to feed the Space Monster Symbiotes.
6. Will the player chose greed and wealth or will the player choose to feed the symbiote monsters to keep them away long enough that their economy can outpace the Symbiote Space Monster Races?

## -------------------------------------

## Core Game Balance Changes

## 1. Reduce initial asteroid density significantly

```python
def initialize_patterns(self) -> None:
    """Initialize the grid with sparse but valuable asteroid patterns"""
    # Clear grid
    self.grid.fill(0)
    self.rare_grid.fill(0)
    self.energy_grid.fill(0)
    
    # Create just a few seed patterns
    patterns = [
        # Small R-pentomino (chaotic growth)
        [(20, 20), (21, 19), (21, 20), (21, 21), (22, 21)]
    ]
    
    # Place patterns at random locations
    for pattern in patterns:
        offset_x = random.randint(10, self.width - 50)
        offset_y = random.randint(10, self.height - 50)
        
        for x, y in pattern:
            nx, ny = x + offset_x, y + offset_y
            if 0 <= nx < self.width and 0 <= ny < self.height:
                # More valuable asteroids (10x value)
                self.grid[ny, nx] = random.randint(100, 500)
                # Some energy to start
                self.energy_grid[ny, nx] = random.uniform(0.7, 1.0)
    
    # Add sparse random cells with high value
    for _ in range(self.width * self.height // 800):  # Only 0.125% random fill
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        self.grid[y, x] = random.randint(100, 500)
        if random.random() < self.rare_chance:
            self.rare_grid[y, x] = 1
            self.grid[y, x] = int(self.grid[y, x] * self.rare_bonus_multiplier)
        self.energy_grid[y, x] = random.uniform(0.3, 1.0)
```

## 2. Add symbiote hunger system to MinerEntity class

```python
class MinerEntity:
    def __init__(self, 
                 race_id: int,
                 color: Tuple[int, int, int],
                 birth_set: set = {3},
                 survival_set: set = {2, 3},
                 initial_density: float = 0.01) -> None:
        # Existing initialization...
        
        # Add hunger system
        self.hunger = 0.0  # Ranges from 0.0 (satiated) to 1.0 (starving)
        self.hunger_rate = 0.01  # How fast hunger increases
        self.aggression = 0.2  # Base aggression level
        self.fed_this_turn = False  # Track if race was fed this turn

    def update_hunger(self, minerals_consumed: int) -> None:
        """Update hunger based on minerals consumed. Return aggression level."""
        # Hunger increases over time
        self.hunger += self.hunger_rate
        
        # Feeding reduces hunger
        if minerals_consumed > 0:
            self.fed_this_turn = True
            self.hunger -= (minerals_consumed / 1000.0)
            
        # Cap hunger between 0 and 1
        self.hunger = max(0.0, min(1.0, self.hunger))
        
        # Calculate current aggression based on hunger
        return self.aggression + (self.hunger * 0.8)
```

## 3. Add feeding mechanic to the Player class

```python
def feed_symbiotes(self, field: AsteroidField, minerals: int) -> int:
    """Feed minerals to symbiotes to reduce their aggression"""
    if self.currency < minerals:
        minerals = self.currency  # Can't feed more than you have
        
    minerals_per_race = minerals // max(1, len(field.races))
    total_fed = 0
    
    for race in field.races:
        race.update_hunger(minerals_per_race)
        race.fed_this_turn = True
        total_fed += minerals_per_race
    
    self.currency -= total_fed
    return total_fed
```

## 4. Update the symbiote behavior to attack player ships if hungry

```python
def update_entities(self) -> Dict[int, int]:
    """Update alien symbiote races with hunger-based aggression system"""
    try:
        # Existing code...
        
        # Reset fed status for all races
        for race in self.races:
            race.fed_this_turn = False
        
        # Process each race's hunger and behavior
        for race in self.races:
            # Update hunger if not fed
            aggression = race.update_hunger(0)
            
            # If starving (high aggression), symbiotes may attack player structures
            # This will be processed in the Player.update_fleet method
            
        # Rest of existing update_entities code...
        
    except Exception as e:
        logging.critical(f"Error in update_entities: {str(e)}")
        import traceback
        logging.critical(traceback.format_exc())
        return {}
```

## 5. Add player fleet/mining ships system

```python
class Player:
    def __init__(self) -> None:
        # Existing initialization...
        
        # Mining fleet
        self.mining_ships = 1  # Start with one ship
        self.max_mining_ships = 10  # Maximum number of ships
        self.ship_positions = [(self.x, self.y)]  # Track ship positions
        self.ship_health = [100]  # Each ship has health
        self.ship_cost = 500  # Cost to build a new ship
        
    def update_fleet(self, field: AsteroidField) -> Dict[str, int]:
        """
        Update the mining fleet status and check for symbiote attacks
        Returns dictionary with damage and lost ships information
        """
        results = {
            "damage_taken": 0,
            "ships_lost": 0,
            "minerals_mined": 0
        }
        
        # Process each ship
        for i in range(len(self.ship_positions) - 1, -1, -1):
            ship_x, ship_y = self.ship_positions[i]
            
            # Check for nearby symbiotes that could attack
            attack_chance = 0.0
            for race in field.races:
                # Only hungry symbiotes attack
                if race.hunger > 0.6 and not race.fed_this_turn:
                    # Check cells around ship
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            nx, ny = ship_x + dx, ship_y + dy
                            if 0 <= nx < field.width and 0 <= ny < field.height:
                                if field.entity_grid[ny, nx] == race.race_id:
                                    # Calculate attack probability based on hunger and distance
                                    dist = max(1, abs(dx) + abs(dy))
                                    attack_chance += race.hunger * (4 - dist) * 0.05
            
            # Check if ship is attacked
            if random.random() < attack_chance:
                # Ship takes damage
                damage = random.randint(5, 20)
                self.ship_health[i] -= damage
                results["damage_taken"] += damage
                
                # Check if ship is destroyed
                if self.ship_health[i] <= 0:
                    self.ship_positions.pop(i)
                    self.ship_health.pop(i)
                    self.mining_ships -= 1
                    results["ships_lost"] += 1
                    
                    # Show attack notification
                    logging.info(f"Ship at ({ship_x}, {ship_y}) destroyed by symbiotes!")
        
        # Update main ship position
        if self.mining_ships > 0:
            self.ship_positions[0] = (self.x, self.y)
        
        # Each ship mines nearby asteroids
        total_mined = 0
        for ship_x, ship_y in self.ship_positions:
            for dy in range(-self.mining_range, self.mining_range + 1):
                for dx in range(-self.mining_range, self.mining_range + 1):
                    nx, ny = ship_x + dx, ship_y + dy
                    if 0 <= nx < field.width and 0 <= ny < field.height:
                        if field.grid[ny, nx] > 0:
                            value = field.grid[ny, nx]
                            if field.rare_grid[ny, nx] == 1:
                                self.total_rare_mined += 1
                                
                            total_mined += value
                            field.grid[ny, nx] = 0
                            field.rare_grid[ny, nx] = 0
                            
        # Calculate mining reward
        reward = int(total_mined * self.mining_efficiency)
        self.currency += reward
        self.total_mined += reward
        results["minerals_mined"] = reward
        
        return results
```

## 6. Add symbiote feeding to the game controls

```python
def handle_events(self) -> None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # Let the notification manager handle its events
        self.notifier.handle_event(event)
        
        if event.type == pygame.KEYDOWN:
            if self.state == STATE_PLAY:
                # Existing key handlers...
                
                # Add symbiote feeding controls
                elif event.key == pygame.K_f:
                    # Feed a small amount
                    minerals_fed = self.player.feed_symbiotes(self.field, 50)
                    self.notifier.add(f"Fed symbiotes: {minerals_fed} minerals")
                    
                elif event.key == pygame.K_g:
                    # Feed a large amount
                    minerals_fed = self.player.feed_symbiotes(self.field, 200)
                    self.notifier.add(f"Fed symbiotes: {minerals_fed} minerals")
                
                # Add fleet management
                elif event.key == pygame.K_b:
                    # Build a new mining ship if resources allow
                    if self.player.currency >= self.player.ship_cost and self.player.mining_ships < self.player.max_mining_ships:
                        self.player.currency -= self.player.ship_cost
                        self.player.mining_ships += 1
                        self.player.ship_positions.append((self.player.x, self.player.y))
                        self.player.ship_health.append(100)
                        self.notifier.add(f"New mining ship built! Fleet: {self.player.mining_ships}")
                    elif self.player.mining_ships >= self.player.max_mining_ships:
                        self.notifier.add("Maximum fleet size reached!", color=COLOR_HIGHLIGHT)
                    else:
                        self.notifier.add(f"Not enough credits! Need {self.player.ship_cost}", color=COLOR_HIGHLIGHT)
```

## 7. Update the Game's update method to include fleet updates and symbiote hunger

```python
def update(self) -> None:
    self.frame_counter += 1
    if self.frame_counter % self.update_interval == 0:
        self.field.update()
        
        # Update player's mining fleet
        fleet_results = self.player.update_fleet(self.field)
        
        if fleet_results["minerals_mined"] > 0:
            self.notifier.add(f"Mining fleet income: +{fleet_results['minerals_mined']} minerals", duration=60)
            
        if fleet_results["damage_taken"] > 0:
            self.notifier.add(f"Fleet attacked! Damage taken: {fleet_results['damage_taken']}", 
                             color=COLOR_HIGHLIGHT, duration=60)
            
        if fleet_results["ships_lost"] > 0:
            self.notifier.add(f"ALERT: {fleet_results['ships_lost']} ships lost to symbiote attacks!", 
                             color=(255, 0, 0), duration=120)
        
        # Report on symbiote races
        for race in self.field.races:
            race_count = np.sum(self.field.entity_grid == race.race_id)
            hunger_status = "Starving" if race.hunger > 0.8 else "Hungry" if race.hunger > 0.4 else "Satiated"
            
            self.notifier.add(f"Race {race.race_id}: {race_count} symbiotes, Status: {hunger_status}", 
                             duration=60,
                             color=race.color)
            
        # Auto-upgrade check if enabled
        self.auto_upgrade_logic()
    
    self.notifier.update()

# Additional imports for advanced mathematical modeling

import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.signal as signal
from sklearn.cluster import KMeans
import networkx as nx
from perlin_noise import PerlinNoise

# -------------------------------------
# Enhanced Symbiote Evolution System
# -------------------------------------
class MinerEntity:
    def __init__(self, 
                race_id: int,
                color: Tuple[int, int, int],
                birth_set: set = {3},
                survival_set: set = {2, 3},
                initial_density: float = 0.01) -> None:
        self.race_id = race_id
        self.color = color
        self.birth_set = birth_set
        self.survival_set = survival_set
        self.initial_density = initial_density
        self.mining_efficiency = 0.5  # Base efficiency
        
        # Advanced symbiote properties
        self.hunger = 0.3  # Ranges from 0.0 (satiated) to 1.0 (starving)
        self.hunger_rate = 0.01  # How fast hunger increases
        self.aggression = 0.2  # Base aggression level
        self.fed_this_turn = False  # Track if race was fed this turn
        self.evolution_stage = 0  # Evolutionary development stage
        self.evolution_points = 0
        self.evolution_threshold = 1000  # Points needed for evolution
        
        # Genome properties that affect behavior
        self.genome = {
            "metabolism_rate": random.uniform(0.8, 1.2),  # How fast they consume energy
            "expansion_drive": random.uniform(0.5, 1.5),  # How aggressively they expand
            "mutation_rate": random.uniform(0.01, 0.1),   # How quickly they evolve
            "intelligence": random.uniform(0.1, 0.5),     # How strategically they target resources
            "aggression_base": random.uniform(0.1, 0.3),  # Base aggression level
        }
        
        # Population metrics
        self.population = 0
        self.last_income = 0
        self.income_history = []
        self.population_history = []
        self.behavior_states = ["feeding", "expanding", "defensive", "aggressive"]
        self.current_behavior = "feeding"  # Default behavior
        
        # Spatial analysis data
        self.territory_center = None
        self.territory_radius = 0
        self.territory_density = 0
        
        # Each race has a unique behavioral trait
        if race_id == 1:  # Blue race - adaptive metabolism
            self.trait = "adaptive"
            self.genome["metabolism_rate"] *= 1.2
            self.genome["mutation_rate"] *= 1.3
        elif race_id == 2:  # Magenta race - aggressive expansion
            self.trait = "expansive" 
            self.genome["expansion_drive"] *= 1.5
            self.genome["aggression_base"] *= 1.2
        elif race_id == 3:  # Orange race - resource intelligence
            self.trait = "selective"
            self.genome["intelligence"] *= 1.5
            self.genome["metabolism_rate"] *= 0.8  # More efficient
    
    def update_hunger(self, minerals_consumed: int) -> float:
        """
        Update hunger based on minerals consumed and genome properties
        Returns the current aggression level
        """
        # Hunger increases over time, affected by metabolism rate
        self.hunger += self.hunger_rate * self.genome["metabolism_rate"]
        
        # Feeding reduces hunger
        if minerals_consumed > 0:
            self.fed_this_turn = True
            hunger_reduction = (minerals_consumed / 1000.0) * (1.0 / self.genome["metabolism_rate"])
            self.hunger -= hunger_reduction
            
            # Feeding can contribute to evolution
            self.evolution_points += minerals_consumed * 0.1
            
        # Cap hunger between 0 and 1
        self.hunger = max(0.0, min(1.0, self.hunger))
        
        # Calculate current aggression based on hunger and genome
        current_aggression = (self.genome["aggression_base"] + 
                            (self.hunger * 0.8) * self.genome["expansion_drive"])
        
        # Update behavior state based on hunger and population
        self._update_behavior_state()
        
        return current_aggression
    
    def _update_behavior_state(self) -> None:
        """Update the current behavior state based on various factors"""
        # Use a probability distribution to determine behavior state
        probabilities = [0.0, 0.0, 0.0, 0.0]  # For each behavior state
        
        # Feeding probability increases with hunger
        probabilities[0] = 0.4 * (self.hunger ** 2)  
        
        # Expansion probability based on population and genome
        if self.population > 0:
            expansion_factor = min(1.0, self.population / 500) * self.genome["expansion_drive"]
            probabilities[1] = 0.3 * expansion_factor
        
        # Defensive behavior when recently fed or low population
        if self.fed_this_turn or self.population < 50:
            probabilities[2] = 0.4
        else:
            probabilities[2] = 0.2
            
        # Aggressive behavior increases with hunger and population
        aggression_factor = self.hunger * self.genome["aggression_base"] * min(1.0, self.population / 300)
        probabilities[3] = 0.3 * aggression_factor
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
            
        # Select behavior based on weighted probability
        self.current_behavior = random.choices(
            self.behavior_states, 
            weights=probabilities, 
            k=1
        )[0]
    
    def evolve(self) -> None:
        """Evolve the race to the next stage, improving its abilities"""
        self.evolution_stage += 1
        self.evolution_points = 0
        
        # Increase evolution threshold for next stage
        self.evolution_threshold *= 1.5
        
        # Apply mutation to genome
        for gene in self.genome:
            # Random mutation with chance for significant changes
            if random.random() < self.genome["mutation_rate"]:
                # Normal small mutation
                mutation_factor = random.normalvariate(1.0, 0.1)
                self.genome[gene] *= mutation_factor
                
                # Small chance for major mutation
                if random.random() < 0.1:
                    major_factor = random.uniform(0.5, 2.0)
                    self.genome[gene] *= major_factor
        
        # Trait-specific evolution benefits
        if self.trait == "adaptive":
            self.mining_efficiency += 0.1
            self.genome["mutation_rate"] *= 1.1
            # Adaptive races can alter their birth/survival rules
            if random.random() < 0.3:
                possible_rules = {1, 2, 3, 4, 5}
                if len(self.birth_set) < 3:
                    new_rule = random.choice(list(possible_rules - self.birth_set))
                    self.birth_set.add(new_rule)
                if len(self.survival_set) < 4:
                    new_rule = random.choice(list(possible_rules - self.survival_set))
                    self.survival_set.add(new_rule)
                    
        elif self.trait == "expansive":
            self.genome["expansion_drive"] *= 1.15
            # Expansive races can expand more aggressively
            self.birth_set.add(random.choice([1, 2]))
            
        elif self.trait == "selective":
            self.mining_efficiency += 0.15
            self.genome["intelligence"] *= 1.2
            # Selective races get better at finding resources
            self.hunger_rate *= 0.9  # Lower hunger rate
    
    def analyze_territory(self, field: AsteroidField) -> Dict[str, Any]:
        """
        Perform spatial analysis of race territory using advanced algorithms
        Returns metrics about territory distribution and resource access
        """
        entity_locations = []
        
        # Find all entities of this race
        for y in range(field.height):
            for x in range(field.width):
                if field.entity_grid[y, x] == self.race_id:
                    entity_locations.append((x, y))
        
        if not entity_locations:
            return {
                "center": None,
                "radius": 0,
                "density": 0,
                "resource_access": 0,
                "fragmentation": 0
            }
        
        # Convert to numpy array for clustering
        points = np.array(entity_locations)
        
        # K-means clustering to find centers of population
        k = min(3, len(points))  # Use up to 3 clusters
        if k > 0:
            kmeans = KMeans(n_clusters=k).fit(points)
            clusters = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # Find main cluster (largest population)
            cluster_sizes = [np.sum(clusters == i) for i in range(k)]
            main_cluster_idx = np.argmax(cluster_sizes)
            main_center = centers[main_cluster_idx]
            
            # Calculate territory metrics
            self.territory_center = (int(main_center[0]), int(main_center[1]))
            
            # Calculate radius as distance to furthest entity in main cluster
            main_cluster_points = points[clusters == main_cluster_idx]
            distances = np.sqrt(((main_cluster_points - main_center) ** 2).sum(axis=1))
            self.territory_radius = int(np.max(distances))
            
            # Calculate density as entities per unit area
            area = max(1, np.pi * (self.territory_radius ** 2))
            self.territory_density = len(main_cluster_points) / area
            
            # Measure resource access (asteroids within territory)
            resource_access = 0
            for y in range(max(0, int(main_center[1] - self.territory_radius)), 
                         min(field.height, int(main_center[1] + self.territory_radius + 1))):
                for x in range(max(0, int(main_center[0] - self.territory_radius)),
                             min(field.width, int(main_center[0] + self.territory_radius + 1))):
                    if field.grid[y, x] > 0:
                        distance = np.sqrt((x - main_center[0])**2 + (y - main_center[1])**2)
                        if distance <= self.territory_radius:
                            resource_access += field.grid[y, x]
            
            # Measure fragmentation using network analysis
            # Create a graph where close entities are connected
            G = nx.Graph()
            for i, (x, y) in enumerate(entity_locations):
                G.add_node(i, pos=(x, y))
                
            # Connect entities that are close to each other
            for i in range(len(entity_locations)):
                for j in range(i+1, len(entity_locations)):
                    x1, y1 = entity_locations[i]
                    x2, y2 = entity_locations[j]
                    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    if distance < 10:  # Connection threshold
                        G.add_edge(i, j)
            
            # Calculate fragmentation as number of connected components
            if len(G.nodes) > 0:
                fragmentation = nx.number_connected_components(G) / len(G.nodes)
            else:
                fragmentation = 0
                
            return {
                "center": self.territory_center,
                "radius": self.territory_radius,
                "density": self.territory_density,
                "resource_access": resource_access,
                "fragmentation": fragmentation
            }
        
        return {
            "center": None,
            "radius": 0,
            "density": 0,
            "resource_access": 0,
            "fragmentation": 0
        }
