# ASCII VISUAL UI LIBRARY

## Interface Components

### Converter Dashboard

The main dashboard provides an overview of all converters and active conversion processes:

```
+---------------------------------------------------------------+
|                    CONVERTER DASHBOARD                        |
+---------------+------------------------+----------------------+
| CONVERTERS    | ACTIVE PROCESSES       | PRODUCTION METRICS   |
| - Smelter #1  | - Iron Ore → Ingots    | Efficiency: 87%      |
| - Smelter #2  | - Copper Ore → Ingots  | Throughput: 45/min   |
| - Assembler   | - Ingots → Components  | Energy Use: 350kW    |
| - Refinery    | - Crude Oil → Fuel     | Queue: 3 processes   |
+---------------+------------------------+----------------------+
|                                                               |
|                     CHAIN VISUALIZATION                       |
|                                                               |
|  [ORE EXTRACTOR] → [SMELTER] → [ASSEMBLER] → [STORAGE]        |
|                                                               |
+-----------------------------------+---------------------------+
|              CONTROLS             |        EFFICIENCY         |
| [START] [PAUSE] [STOP] [OPTIMIZE] | Base: 0.8  Tech: 1.2      |
|                                   | Quality: 1.1  Env: 0.95   |
+-----------------------------------+---------------------------+
```

### Converter Details View

Detailed view of a selected ASCII UI converter with all its stats and controls:

```
+---------------------------------------------------------------+
|                  CONVERTER: ADVANCED SMELTER                  |
+-----------------------------------+--------------------------->
| STATUS: ACTIVE                    | TIER: 3                   |
| EFFICIENCY: 87%                   | ENERGY: 120kW/45kW        |
| UTILIZATION: 65%                  | UPTIME: 3h 45m            |
+-----------------------------------+--------------------------->
|                                                               |
|                ACTIVE CONVERSION PROCESSES                    |
|                                                               |
|[#142] Iron Ore → Iron Ingots (76% complete) [PAUSE] [STOP]    |
|[#143] Copper Ore → Copper Ingots (42% complete) [PAUSE] [STOP]|
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                   AVAILABLE RECIPES                           |
|                                                               |
| • Iron Ore → Iron Ingots (Base Eff: 0.9) [START]              |
| • Copper Ore → Copper Ingots (Base Eff: 0.85) [START]         |
| • Gold Ore → Gold Ingots (Base Eff: 0.7) [START]              |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                   EFFICIENCY FACTORS                          |
|                                                               |
| • Base Efficiency: 0.8                                        |
| • Quality Modifier: 1.1 (Good quality inputs)                 |
| • Technology Modifier: 1.2 (Level 4 tech)                     |
| • Environmental Modifier: 0.95 (Minor hazard)                 |
| → Applied Efficiency: 0.87 (0.8 × 1.1 × 1.2 × 0.95)           |
|                                                               |
+---------------------------------------------------------------+
```

### Chain Management Interface

ASCII UI Interface for creating and managing multi-step production chains:

```
+---------------------------------------------------------------+
|                  PRODUCTION CHAIN MANAGEMENT                  |
+---------------------------------------------------------------+
|                                                               |
|                     ACTIVE CHAINS                             |
|                                                               |
| [#24] Basic Electronics (3 steps, 45% complete)               |
|   › Step 1: Copper Ore → Copper Ingots [COMPLETED]            |
|   › Step 2: Iron Ore → Iron Plates [IN PROGRESS]              |
|   › Step 3: Copper + Iron → Electronic Components [PENDING]   |
|                                                               |
| [#25] Advanced Alloy (2 steps, 30% complete)                  |
|   › Step 1: Titanium Ore → Titanium Ingots [IN PROGRESS]      |
|   › Step 2: Titanium + Steel → Advanced Alloy [PENDING]       |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                     CHAIN CREATOR                             |
|                                                               |
|  Step 1: [SMELTER #1] [IRON ORE → IRON INGOTS] [▼]            |
|  Step 2: [ASSEMBLER] [IRON INGOTS → IRON PLATES] [▼]          |
|  Step 3: [FABRICATOR] [IRON PLATES → COMPONENTS] [▼]          |
|                                                               |
|  [+ ADD STEP]                [SAVE AS TEMPLATE] [START CHAIN]|
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                     CHAIN TEMPLATES                           |
|                                                               |
|  • Basic Electronics (3 steps) [LOAD] [DELETE]                |
|  • Advanced Alloy (2 steps) [LOAD] [DELETE]                   |
|  • Fuel Processing (4 steps) [LOAD] [DELETE]                  |
|                                                               |
+---------------------------------------------------------------+
```

### Efficiency Monitor

Detailed ASCII visualization of efficiency factors with historical tracking:

```
+---------------------------------------------------------------+
|                     EFFICIENCY MONITOR                        |
+---------------------------------------------------------------+
|                                                               |
|  OVERALL EFFICIENCY: 92%                                      |
|  ██████████████████▒▒▒▒ (TREND: +5% from last cycle)          |
|                                                               |
|  EFFICIENCY BREAKDOWN:                                        |
|  • Base Efficiency:         0.85  ███████████▒▒▒▒             |
|  • Quality Modifier:        1.15  ███████████████▒            |
|  • Technology Modifier:     1.25  ████████████████▒           |
|  • Environmental Modifier:  0.95  ████████████▒▒▒             |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|  EFFICIENCY HISTORY:                                          |
|  ↗                                                            |
|    ↗       ↗                                                  |
|  ↗   ↘   ↗   ↘   ↗                                            |
|     ↘         ↘                                               |
|  Last 24 hours (Avg: 87%)                                     |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|  EFFICIENCY OPTIMIZATION SUGGESTIONS:                         |
|  • Upgrade converter to Tier 4 (+10% base efficiency)         |
|  • Research "Advanced Metallurgy" (+15% tech modifier)        |
|  • Use higher quality Iron Ore (+8% quality modifier)         |
|  • Address nearby hazard (+5% environmental modifier)         |
|                                                               |
+---------------------------------------------------------------+
```
