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

This is a fascinating game concept blending cellular automata with resource management and strategy. To provide the best possible algorithm and mathematical models for your game logic, lets clarify a few details:

### 1. Symbiote Mutation Mechanics

**(a) Should mutations be purely random like Conway’s Game of Life, or should they be influenced by external factors (e.g., mineral feeding, time, player actions)?**

1. **(a):** *They should be purely random, but accellerated by external factors such as user feeding them minerals they have mined. So they should grow in size or mutate faster, but it should still be true to a random mutation. I envision it to be truly random, with anomalies, just like Conways game to this day creates new and fascinating things. The Mutations should be influenced by the minerals they eat, but still ultimately be true to Conway.*

**(b) Do you envision different types of mutations (e.g., size, speed, aggression, reproduction rate)?**

 1. **(b):** *Think of it like this, life evolves over thousands of years over influence of what it can eat, what food it can reach, these should be factors in this. If a Symbiote Space Monster eats a ton of a certain mineral, it should influence, but not dictate its mutations. Survival of the fittest in the vastness of space for the symbioses. If they eat more of a rare mineral, it should be a bigger influence, make them more dangerous than say eating a ton of a common mineral. But still, ultimately, random as Conway had it. Like recreating Conways vision but with more of a Natural Selection influence.*

### 2. Currency and Resource System

**(a) Should there be different types of minerals that impact mutations differently?**

1. **(a):** *There should be many different types of minerals, ranging from; `(i): Common, (ii): Rare, (iii): Precious, (iv): Unmapped and the most rare, (v): Anomaly Tier Mineral.` Each with their own benefits. More common minerals eaten should provide a steady food source, but create less dangerous Symbioses. The more rare the minerals get, the bigger influence they should have of making them a `WORLD EATER` Class Symbiote. Think of this like Herbavores(prey), Carnivores(predators).*

**(b) How should the balance between selling minerals and feeding symbiotes be calculated (e.g., a formula for cost-benefit analysis)?**

1. **(b):**
   - *I Think a formula for cost benefit analysis is smart. Think of it like this, if a very powerful Symbiote comes across the user(human space miners), they may have to feed all of their rare minerals just to feed its hunger. The cost would be staying alive, losing everything, vs trying to build a large enough economy that you can overpower any Symbiotes mutation.*

### 3. **Player Fleet Progression**:
    
**(a) What kind of upgrades can be purchased (e.g., ship speed, mining efficiency, automated drones)?**

1. **(a):** *So far i have implemented:*

- *For Human Space Miners*
  - *Ship Upgrades: Mining Range, Mining Effeciency, Auto Mining Drone*
  - *Field Upgrades: Manual Asteroid Seeding (Explore for Asteroids to Mine)*
  - *Field Control: Add Birth Option (2 Neighbours) *a dead cell will spawn an asteroid if it has 2 dead neighbours*
  - *Field Control: Increase Regeneration Rate (1%) *increase chance for empty cells to spawn asteroids*
  - *Enhance Rare Asteroids 5% (Explore for Rare Asteroids) *increase chance for asteroids to be rare*

- *For Symbiote Space Monsters:*
  - *Discover Blue Symbiote (Adds the Blue Symbiote Monster to the game) *This allows the player to get a bit of a head start if they wish*
  - *Discouver Magenta Symbiote Monster (Same as above)*
  - *Discouver Orange Symbiote Monster (Same as above)*
  - *I am open to any suggestions and implementations through your deep research and algorithm for the Symbiote Space Monsters based on my Explanation in answers 1. and 2.*

**(b) Should upgrading the fleet make it easier to mine but also increase symbiote aggression in response?**

3. **(b):** *Upgrading should make the Symbiotes more aware that the human space miners(player) has food. Sort of like alligators waiting at the watering hole for deer to come to drink. The symbiotes should be more agressive the more riches the human miners have, especially if they dont get a good seed and are hungry. This should scale though. If the human miners sell all of their minerals, they dont have the food anymore, so the Symbiotes should become less agressive. BUT the risk of not having any minerals on hand, is that if a Symbiote does show up, you have nothing to feed them to deter them from destroying your mining fleet and or ship.*

### 4. **Game Pace and Balance**:
    
**(a) How fast should symbiotes mutate relative to player progression?**

4. **(a):** *The Symbiotes should mutate purely based on their ability to find food (minerals). If the human miners upgrade exploring and finding rare minerals too fast, the symbiotes surely come faster because the human space miners have food. It should be a risk and reward. Every game should be different, both because of Conways random Mutation aspect and the strategic aspect of adding purchasable upgrades, and mineral(food) tiers.*

**(b) Should mutations have a cap, or should they escalate indefinitely?**

4. **(b):** *No cap on mutations, the game should run as is. If a symbiote happens to spawn near Anomaly Tier minerals(food for them) annd become WORLD EATER class quite quickly, well that is life in space. It should be random, yet influenced by choice. Just as is human life and our existence came to be.*

### 5. **Mathematical Complexity**:
    
**(a) Are you looking for simple probability-based mechanics, or more complex models like differential equations, cellular automata with external input, or machine learning-based emergent behavior?**

5. **(a):** *Complex models like differential equations, cellular automata with external input. Machine Learning behaviour would be a big bonus aswell.
 - *The key focus and core engine here will ultimately be mathematics, you have all of my ideas and my game structure so far. It is now up to you, to truly ascend my game by creating a truly unique, purposeful, useful, and advanced algorithm, supported by mathematics for this game to be successful.*
 - *Your research must be thorough, you must think outside of the box, and you must create the mathematical foundations to this. You cannot do that by simply providing effecient algorithms or mathematics. You have to go above and beyond, creating something new catered for this exact purpose in the game ive described.*