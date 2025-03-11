# ASCII UI Library

In this file you will find examples of ASCII UI elements that can be used to create a CLI interface for the application.

**MOST UI ARE EXAMPLE TEMPLATES, AND WOULD REQUIRE PROPER INPUT DATA FOR THE RELATED IMPLEMENTATIONS**

## Starting Screen

```
 /-=============================================================\
||                                                             ||
||      ▗▄▄▖▗▄▄▖  ▗▄▖  ▗▄▄▖▗▄▄▄▖    ▗▖  ▗▖▗▖ ▗▖ ▗▄▄▖▗▖ ▗▖      ||
||     ▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌       ▐▛▚▞▜▌▐▌ ▐▌▐▌   ▐▌▗▞▘      ||
||      ▝▀▚▖▐▛▀▘ ▐▛▀▜▌▐▌   ▐▛▀▀▘    ▐▌  ▐▌▐▌ ▐▌▐▌   ▐▛▚▖       ||
||     ▗▄▄▞▘▐▌   ▐▌ ▐▌▝▚▄▄▖▐▙▄▄▖    ▐▌  ▐▌▝▚▄▞▘▝▚▄▄▖▐▌ ▐▌      ||
||                                                             ||
||=============================================================||
||                                                             ||
||    ┌─\│/─┐  ╔═╗┌┬┐┌─┐┬─┐┌┬┐  ╔═╗┌─┐┌┬┐┌─┐                   ||
||    │ ─ ─ │  ╚═╗ │ ├─┤├┬┘ │   ║ ╦├─┤│││├┤                    ||
||    └─/│\─┘  ╚═╝ ┴ ┴ ┴┴└─ ┴   ╚═╝┴ ┴┴ ┴└─┘                   ||
||                                                             ||
||    ┌─O┬──┐  ╦  ┌─┐┌─┐┌┬┐  ╔═╗┌─┐┌┬┐┌─┐                      ||
||    │ ┌┘  │  ║  │ │├─┤ ││  ║ ╦├─┤│││├┤                       ||
||    └─┴O──┘  ╩═╝└─┘┴ ┴─┴┘  ╚═╝┴ ┴┴ ┴└─┘                      ||
||                                                             ||
||    ┌─┌─┐─┐  ╔═╗┌─┐┌┬┐┬┌─┐┌┐┌┌─┐                             ||
||    │  ┌┘ │  ║ ║├─┘ │ ││ ││││└─┐                             ||
||    └─ o ─┘  ╚═╝┴   ┴ ┴└─┘┘└┘└─┘                             ||
||                                                             ||
||    ┌──┬──┐  ╔═╗ ┬ ┬┬┌┬┐                                     ||
||    │  │  │  ║═╬╗│ ││ │                                      ||
||    └──o──┘  ╚═╝╚└─┘┴ ┴                                      ||
||                                                             ||
\===============================================================/
```

## Menu UI Example

```
┌──────────────────── [ ERROR-HANDLER CLI ] ──────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │
│  │ RUN │ │ VAL │ │ LOG │ │ CFG │ │ FIX │ │ SYS │ │ HELP│    │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘    │
│                                                             │
│  ┏━━━━━━━━━━━━━━━━━━ COMMAND INPUT ━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃                                                       ┃  │
│  ┃  $ project-cli run --port 8080 --path ./myproject _   ┃  │
│  ┃                                                       ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                             │
│  ┏━━━━━━━━━━━━━━━━━━ ERROR OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃                                                       ┃  │
│  ┃  ⚠ ValidationError: Invalid project path              ┃  │
│  ┃  ✗ Directory './myproject' contains no Python files   ┃  │
│  ┃  ! Additional context available with --verbose        ┃  │
│  ┃                                                       ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                             │
│  ┏━━━━━━━━━━━━━━━━━━ SUGGESTIONS ━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃                                                       ┃  │
│  ┃  ▶ Try: project-cli validate --path ./myproject       ┃  │
│  ┃  ▶ Try: project-cli run --path ./correct/path         ┃  │
│  ┃  ▶ See: project-cli --help for more information       ┃  │
│  ┃                                                       ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                             │
│  System: Linux | User: admin | Version: 2.1.3 | Log: Active │
└─────────────────────────────────────────────────────────────┘
```

## Arrows and Progress Bar Example

```
╭───────────────────────────────────────╮
│                                       │
│  ◢█◤ validate_project_path            │
│  └─▶ ./projects/quantum-app           │
│                                       │
│  ◢█◤ validate_port                    │
│  └─▶ 3000   [████████░░] SECURE       │
│                                       │
│  ◢█◤ system_status                    │
│  └─▶ ⚠ ValidationError detected       │
│     ↳ ./quantum-app/config.py missing │
│                                       │
│  ┌╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾┐   │
│  ┤ ERROR REMEDY SUGGESTIONS       ├   │
│  └╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾╼╾┘   │
│                                       │
│ ◢█◤  Create missing config file       │
│  └─▶ Run: project fix --path ./app    │
│  └─▶ See: project --help              │
│                                       │
╰───────────────────────────────────────╯
```

## Circuit Board Interface Example

```
┌────────────────────────────────────────┐
│ ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐  │
│ │ERROR│───┤CHECK├───┤VALID├───┤FIXED│  │
│ └┬────┘   └┬────┘   └┬────┘   └──┬──┘  │
│  │         │         │           │     │
│ ┌▼─────────▼─────────▼───────────▼─┐   │
│ │                                  │   │
│ │  $ run --port 8080 --path ./app  │   │
│ │                                  │   │
│ └┬─────────────────────────────────┘   │
│  │                                     │
│ ┌▼────────────────────────────────┐    │
│ │ [X] ValidationError:            │    │
│ │     Port 8080 requires root     │◀┐  │
│ └┬────────────────────────────────┘ │  │
│  │                                  │  │
│ ┌▼─────────────────────────────────┐│  │
│ │ ┌───────────────────────────────┐││  │
│ │ │Try: sudo run --port 8080 --p..│││  │
│ │ └───────────────────────────────┘││  │
│ │ ┌───────────────────────────────┐││  │
│ │ │Try: run --port 3000 --path ./a│└┘  │
│ │ └───────────────────────────────┘    │
│ └──────────────────────────────────────┤
└────────────────────────────────────────┘
```

## System Terminal

```
╭───────────── SYSTEM TERMINAL ─────────────╮
│                                           │
│  ╭─────────────────────────────────────╮  │
│  │                                     │  │
│  │  NEBULA@SYSTEM:~$ _                 │  │
│  │                                     │  │
│  ╰─────────────────────────────────────╯  │
│                                           │
│  [COMMANDS]  [FILES]  [SYSTEM]  [HELP]    │
│                                           │
╰───────────────────────────────────────────╯
```

## Pause Menu Sudiebar Popout UI Example

```
      _____________________________________________
 .==='                  + Menu +                   '===.
 ( * )---------------------------------------------( * )
 '==='                                             '===' 
  |||          [1] New Game    [2] Main Menu        |||
  |||          [3] Options     [4] Exit             |||
 .===.                                             .===.
 ( * )---------------------------------------------( * )
 '==='_____________________________________________'===' 
```

## Sidebar UI Example

This menu can be used for multiple purposes. You can use it to display a list of options, a form, or a grid of items. You should remove the

```
┌───────────────────── CLI-TEMPLATE ──────────────────────────┐
│                                                             │
│  ┌─────────┐ ┌───────────────────────────────────────────┐  │
│  │ MENU    │ │                                           │  │
│  ├─────────┤ │  $ _                                      │  │
│  │ ▶ ITEM1 │ │                                           │  │
│  │ ○ ITEM2 │ ├───────────────────────────────────────────┤  │
│  │ ○ ITEM3 │ │                                           │  │
│  │ ○ ITEM4 │ │  [MESSAGE-TYPE]                           │  │
│  │ ○ ITEM5 │ │  Primary message text                     │  │
│  │ ○ ITEM6 │ │  Secondary message details                │  │
│  │ ○ ITEM7 │ │                                           │  │
│  ├─────────┤ ├───────────────────────────────────────────┤  │
│  │ STATUS  │ │                                           │  │
│  ├─────────┤ │  Content section:                         │  │
│  │ ● STAT1 │ │  1. Primary content line                  │  │
│  │ ○ STAT2 │ │     Details for first item                │  │
│  │ ● STAT3 │ │  2. Secondary content line                │  │
│  │ ● STAT4 │ │     Details for second item               │  │
│  │         │ │                                           │  │
│  └─────────┘ └───────────────────────────────────────────┘  │
│                                                             │
│ [KEY1:Action1] [KEY2:Action2] [KEY3:Action3] [KEY4:Action4] │
└─────────────────────────────────────────────────────────────┘
```

## Resource Example UI

```
┏━━━━━━━━━━━━━━━━ COLONY RESOURCES ━━━━━━━━━━━━━━━━┓
┃                                                  ┃
┃  ┏━━━━━━━━━┓ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  ┃
┃  ┃ OVERVIEW┃ ┃                                ┃  ┃
┃  ┣━━━━━━━━━┫ ┃  COLONY STATUS: DAY 127        ┃  ┃
┃  ┃ ◉ BASE  ┃ ┃  Population: 78/120            ┃  ┃
┃  ┃ ○ FOOD  ┃ ┃  Morale: HIGH                  ┃  ┃
┃  ┃ ○ WATER ┃ ┃                                ┃  ┃
┃  ┃ ○ POWER ┃ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  ┃
┃  ┃ ○ MEDS  ┃ ┃                                ┃  ┃
┃  ┃ ○ TECH  ┃ ┃  RESOURCE LEVELS               ┃  ┃
┃  ┣━━━━━━━━━┫ ┃  FOOD    ██████████░░░░ 67%    ┃  ┃
┃  ┃ ALERTS  ┃ ┃  WATER   ████████████ 100%     ┃  ┃
┃  ┣━━━━━━━━━┫ ┃  POWER   ███████░░░░░ 58%      ┃  ┃
┃  ┃ ⚠️ LOW   ┃ ┃  MEDICAL ███░░░░░░░░░ 32% ⚠️    ┃  ┃
┃  ┃   FOOD  ┃ ┃  TECH    ██████░░░░░░ 51%      ┃  ┃
┃  ┃ ✓ WATER ┃ ┃                                ┃  ┃
┃  ┃ ⚠️ POWER ┃ ┃  PROJECTED DEPLETION: 18 DAYS  ┃  ┃
┃  ┃ ! MEDS  ┃ ┃                                ┃  ┃
┃  ┗━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  ┃
┃                                                  ┃
┃  [1:Allocate] [2:Trade] [3:Craft] [4:Expand]     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## TECH TREE EXAMPLE UI

```
╭━━━━━━━━━━━━━━ ARCANE GRIMOIRE ━━━━━━━━━━━━━╮
┃                   ✧✦✧                      ┃
┃  ╭───────────╮  ╭───────────────────────╮  ┃
┃  │ SCHOOLS   │  │                       │  ┃
┃  ├───────────┤  │    SPELL WEAVING      │  ┃
┃  │ ▓ FIRE    │  │                       │  ┃
┃  │ ░ WATER   │  │    ╭─Fire─╮           │  ┃
┃  │ ░ EARTH   │  │    │ III  │           │  ┃
┃  │ ░ AIR     │  │    ╰──┬───╯           │  ┃
┃  │ ░ VOID    │  │       │               │  ┃
┃  ├───────────┤  │       ▼               │  ┃
┃  │ MANA      │  │    ╭──────╮           │  ┃
┃  ├───────────┤  │    │SHAPE │           │  ┃
┃  │ ████████░░│  │    ╰──┬───╯           │  ┃
┃  │ 82/100    │  │       │               │  ┃
┃  │           │  │       ▼               │  ┃
┃  │ REGEN: 4/s│  │    ╭─Spirit╮          │  ┃
┃  │           │  │    │  II   │          │  ┃
┃  ╰───────────╯  ╰───────────────────────╯  ┃
┃                                            ┃
┃  [TAB:School] [E:Cast] [R:Reset] [F:Focus] ┃
╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯
```
## Grid Example UI

```

```

## Radial Example UI

```
// 4. RADIAL MENU SYSTEM

             NORTH
               |
               |
       NW      |      NE
         \     |     /
          \    |    /
           \   |   /
            \  |  /
WEST ----------+---------- EAST
            /  |  \
           /   |   \
          /    |    \
         /     |     \
       SW      |      SE
               |
               |
             SOUTH

    [SELECT DESTINATION SECTOR]

   <STATUS: STEALTH MODE ACTIVE>
   <FUEL: 64%> <HULL: 87%>

   [TAB] TOGGLE VIEW  [SPACE] SELECT
   ```

## Ship Control Example UI

```
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
       TACTICAL OVERLAY      
▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
      ↑        N        ↑
   NW ╲        |        ╱ NE
       \       |       /
        \      |      /
W ────── + CENTER + ────── E
        /      |      \
       /       |       \
   SW ╱        |        ╲ SE
      ↓        S        ↓
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
      SYSTEM READOUTS      
▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
 WEAPONS [||||||||--] 84%
 SHIELDS [||||||----] 63%
 ENGINES [|||||||---] 75%
▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
     COMMAND PROTOCOLS     
▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
   [A]ttack [D]efend [M]ove  
   [S]can   [R]etreat [H]old  
```

## Building UI Example

● - ON
○ - OFF

```
      MINING CONTROL
      
      ╔════╦════╦════╗
      ║ ○  ║ ●  ║ ●  ║
╔═════╣ Cu ║ Fe ║ Au ╠══════╗
║     ║    ║    ║    ║      ║
║  C  ╚════╩════╩════╝   O  ║
║  O           ┌─┐       U  ║
║  O           │▲│       T  ║
║  L     ┌─────┴─┴─────┐ P  ║
║  A     │  DRILLING   │ U  ║
║  N     └─┬─────────┬─┘ T  ║
║  T      ┌┴┐       ┌┴┐     ║
║         │▼│       │▼│     ║
╚═════════╩═╩═══════╩═╩═════╝

   POWER: 87%  TEMP: 623K
   STATUS: NOMINAL

   [1-3]:SELECT  [+/-]:ADJUST

```

## FOUR PANEL SIDEBAR EAMPLE UI
```
// 9. WEATHER MONITORING SYSTEM

╭─────────────────────────────╮
│      WEATHER STATION        │
├─────────────────────────────┤
│                             │
│ DATE: 03-12 ┊ TIME: 14:32   │
│                             │
│ ┏━━━━━━━━━━━┓ ┏━━━━━━━━━━━┓ │
│ ┃ TEMP °C   ┃ ┃ HUMIDITY  ┃ │
│ ┃           ┃ ┃           ┃ │
│ ┃    24.6   ┃ ┃    68%    ┃ │
│ ┃           ┃ ┃           ┃ │
│ ┗━━━━━━━━━━━┛ ┗━━━━━━━━━━━┛ │
│                             │
│ ┏━━━━━━━━━━━┓ ┏━━━━━━━━━━━┓ │
│ ┃ PRESSURE  ┃ ┃ WIND      ┃ │
│ ┃           ┃ ┃           ┃ │
│ ┃ 1015 hPa  ┃ ┃  8 km/h   ┃ │
│ ┃           ┃ ┃     →     ┃ │
│ ┗━━━━━━━━━━━┛ ┗━━━━━━━━━━━┛ │
│                             │
│ FORECAST:     ☁️ →           │
│ ALERT LEVEL:  ■□□□          │
│                             │
╰─────────────────────────────╯
```

## Galaxy Map Example UI
```
// 10. ORBITAL TRACKING SYSTEM

            .
       .         .
  .                    .
                 +
     .   [PLANET]     .
                   
    ----------*----------
   /            ORBIT 1   \
  /                         \
 |                           |
 |              .            |
 |         .         .       |
 |    .                      |
 |            [SAT-1]        |
  \                         /
   \                       /
    -----------------------
     \                   /
      \                 /
       \               /
        \     .       /
         \           /
          \  [SAT-2]/
           \       /
            \     /
             \   /
              \ /
               *
           [STATION]

   TRACKING: SAT-1  STATUS: NOMINAL
   ORBIT: 100km     VELOCITY: 7.8km/s
   
   [T] TARGET  [C] COMMS  [O] ORBIT

```

```
// 11. COMBAT HUD

┌───────────────────────────────┐
│               ^               │
│       █       │       █       │
│     █         │         █     │
│    █          │          █    │
│   █           │           █   │
├────────────┬──┼──┬────────────┤
│ AMMO: 28/30│  │  │HEALTH: 87  │
├────────────┘  │  └────────────┤
│               │               │
│      █        │        █      │
│       █       │       █       │
│        █      │      █        │
│         ◯─────┼─────◯         │
│         TARGET LOCKED         │
└───────────────────────────────┘
 WEAPON: RIFLE    MODE: SINGLE
 DISTANCE: 53M    WIND: 3KM/H→
 ```

## Multi Panel UI Example

 Below is **one** large **multi-panel ASCII “game UI” layout** that you can **directly copy and adapt**. It features a **header banner**, **left character/stats panel**, **center gameplay/log area**, **right inventory/equipment panel**, and a **bottom status line**. The design uses layered borders, decorative corners, and various ASCII textures to look more “advanced” than a simple box. 

```
  .======================================================================================.
  ||         ______   __  __   ______   ____   ____   ______    ____   ____   _____      ||
  ||        |   ___| |  ||  | |   ___| |    | |    | |   ___|  /    | |    | |  ___|     ||
  ||        |  |__   |  ||  | |  |__   | |  | | |  | |  |__   / /|  | | |  | | |__       ||
  ||        |   __|  |  ||  | |   __|  | |  | | |  | |   __| / /_|  | | |  | |  __|      ||
  ||        |  |     |  ||  | |  |     | |__| | |__| |  |   / ___   | | |__| | |___      ||
  ||        |__|     |__||__| |__|     |____| |____| |__|  /_/   |__| |____| |_____|     ||
  ||                                                                                     ||
  ||=====================================================================================||
  ||                                                                                     ||
  ||   [F1] Help  [F2] Character Sheet  [F3] Inventory  [F4] Map  [ESC] Quit             ||
  '======================================================================================='
```

```
  .---------------------------------------------------------------------------------------.
  |  .---------------------------.        .--------------------------------------------.  |
  |  |  CHARACTER / PARTY INFO   |        |               INVENTORY / GEAR             |  |
  |  '---------------------------'        '--------------------------------------------'  |
  |                                                                                       |
  |  Name:  <HeroName>                       ┌────────────────────────────────────┐       |
  |  Class: <HeroClass>                      |  Weapon: Longsword (DMG+5)         |       |
  |  Level: 10      EXP: 23450 / 30000       |  Shield: Iron Buckler (DEF+3)      |       |
  |  HP: [##########------]  65 / 100        |  Armor: Leather Vest (DEF+2)       |       |
  |  MP: [#####--------------]  15 / 50      |                                    |       |
  |                                          |  Potion (x2)                       |       |
  |  Status Effects: Poisoned (2 turns)      |  Elixir (x1)                       |       |
  |                                          |  Key Items: Gate Key, Map Fragment |       |
  |                                          └────────────────────────────────────┘       |
  |                                                                                       |
  |  Attributes:                                                                          |
  |    STR: 14   DEX: 11  CON: 12  INT:  8                                                |
  |    WIS:  9   CHA: 13  LCK:  5                                                         |
  |                                                                                       |
  |---------------------------------------------------------------------------------------|
  |  .--------------------------------------------------------------------------------.   |
  |  |                                  GAMEPLAY LOG                                  |   |
  |  |--------------------------------------------------------------------------------|   |
  |  |  > You enter the dark ruins...                                                 |   |
  |  |  > An eerie wind whistles through the cracked stone.                           |   |
  |  |  > A venomous spider emerges from the shadows!                                 |   |
  |  |  > The spider attacks! You lose 4 HP.                                          |   |
  |  |  > [COMBAT MENU: Attack | Skill | Item | Flee ]                                |   |
  |  '--------------------------------------------------------------------------------'   |
  '---------------------------------------------------------------------------------------'

   [Time: 14:32]   [Location: Dark Ruins Entrance]   [FPS:60]   [Ping: 34ms]   [Gold: 123]
```

## Large ASCII MENU TWO

1. **Banner / Header**  
   - Large ASCII text block for a game title or logo at the top.  
   - A horizontal “ruler” of `=` signs separates the logo from the function key shortcuts (F1, F2, etc.).

2. **Main Upper Panels**  
   - Left side: “Character / Party Info.” This box has name, class, level, HP/MP bars, status effects, attributes, etc.  
   - Right side: “Inventory / Gear.” A separate box for equipment listings, potions, key items, etc.

3. **Gameplay / Log Panel**  
   - A wide rectangular box near the bottom for text updates, story events, chat logs, or combat messages.

4. **Bottom Status Line**  
   - A single line across the bottom that can display time, location, FPS, ping, or in-game currency.  
   - In many older text-based games (MUDs, roguelikes, etc.), the bottom line is prime real estate for vital stats or commands.

5. **Vertical & Horizontal Separators**  
   - We use ASCII lines made of `|`, `-`, `.`, `'`, and other characters for visual grouping.  
   - Notice the layout relies on heavy use of spacing for alignment.  

```
   .========================================================================================.
   ||    _______  _______  ___   __   __  _______  ______    _______  _______  ___   __    ||
   ||   |       ||       ||   | |  |_|  ||       ||    _ |  |       ||       ||   | |  |   ||
   ||   |  _____||    _  ||   | |       ||    ___||   | ||  |_     _||_     _||   | |  |   ||
   ||   | |_____ |   |_| ||   | |       ||   |___ |   |_||_   |   |    |   |  |   | |  |   ||
   ||   |_____  ||    ___||   | |       ||    ___||    __  |  |   |    |   |  |   | |__|   ||
   ||    _____| ||   |    |   | | ||_|| ||   |___ |   |  | |  |   |    |   |  |   |  __    ||
   ||   |_______||___|    |___| |_|   |_||_______||___|  |_|  |___|    |___|  |___| |__|   ||
   ||                                                                                      || 
   ||======================================================================================||
   || [1] NAVIGATION    [2] SCANNERS    [3] COMMS    [4] WEAPONS    [5] CREW    [ESC]EXIT  ||
   '========================================================================================'

   .-----------------------------------------------------------------------------------------.
   |   .--------------------------------------.   .--------------------------------------.   |
   |   | SHIP SYSTEM STATUS                   |   | LOCAL SECTOR - STAR MAP              |   |
   |   |--------------------------------------|   |--------------------------------------|   |
   |   | Hull Integrity:  [#######-----] 67%  |   |   Current System: ARGUS-PRIME        |   |
   |   | Shields:         [#########---] 82%  |   |   Planet: Borial-3                   |   |
   |   | Reactor Output:  [######------] 59%  |   |   Station: Durok-Delta               |   |
   |   | Sublight Engines:  ONLINE            |   |   Coordinates: X=128, Y=455, Z=12    |   |
   |   | FTL Drive:         CHARGING (33%)    |   |   Distances: Planet=45, Station=72   |   |
   |   '--------------------------------------'   |  .-----------------------------------.   |
   |                                              |  | VESSELS DETECTED:                  |  |
   |   CREW ROSTER:                               |  |   Friendly: 2 (Argus Patrol)       |  |
   |     - CAPT. A. Reynolds  (Bridge)            |  |   Hostile:  1 (Unknown)            |  |
   |     - LT. S. Davis       (Engine Room)       |  |   Neutral:  3 (Merchants)          |  |
   |     - ENS. T. Kahn       (Shields)           |  '------------------------------------'  |
   |     - ENS. R. Powell     (Weapons)           |                                          |
   '-----------------------------------------------------------------------------------------'

   [Stardate: 3057.4]   [Location: Sector 7-Gamma]   [Alert: GREEN]   [Credits: 5200]   [FPS: 60]
```

## Main MENU ASCII Screen

```
                                 ·         ✦        .
       .              °                      .                 ✧
            ✧                 ∘     .               .     ✦        .
   ✦              .    .              S P A C E - M U C K       .    .
         .               °          (A GAME OF GALAXY LIFE)     ✧          .
            .      ✦           .                     .        °        .
       .                  ·                ∘                     .   ✧
                            .      .       .      .      °     .
                             

         ╔══════════════════════════════════════════════════════════════╗
         ║ ░░   SIDEBAR - QUICK MODULES  ░░                             ║
  ╭──────╫──────────────────────────────────────────────────────────────╫──────────────────────────────────────────╮
  │      ║ ● NAVIGATION        ○ AUTO-CHART        ○ COMMS              ║                                          │
  │      ║ ○ CARGO BAY         ○ MINING LOG        ● DRONE OPS          ║  ┌────────────────────────────────────┐  │
  │      ║ ○ STATION DOCK      ○ CREW ASSIGN       ○ TASK LIST          ║  │   POP-OUT / MODAL PANEL            │  │
  │      ╚══════════════════════════════════════════════════════════════╝  │ ───────────────────────────────────│  │
  │       ░░  Toggle with [↑↓], Press [ENTER] to enable/disable            │   SELECT AN ACTION:                │  │
  │                                                                        │     [1] CONFIRM JUMP               │  │
  │                                                                        │     [2] REROUTE POWER              │  │
  │                                                                        │     [3] DUMP CARGO                 │  │
  │   ┌─────────────────────────────────────────────────────────────────┐  │     [4] CANCEL / RETURN            │  │
  │   │ SHIP CONTROLS                                                   │  │                                    │  │
  │   │ ╭─────────────────────────────╮   ╭───────────────────────────╮ │  │  Warnings:                         │  │
  │   │ │ REACTOR / POWER GRID        │   │ THRUSTERS / FTL DRIVE     │ │  │   - Exiting sector unscanned       │  │
  │   │ │ ─────────────────────────── │   │ ────────────────────────  │ │  │   - Low hull integrity (32%)       │  │
  │   │ │ Reactor Load:   68%  [####-]│   │ Sublight: ○ OFF / ● ON    │ │  └────────────────────────────────────┘  │
  │   │ │ Aux Output:     45%  [##---]│   │ FTL Drive:  ○ OFF  (CHRG) │ │                                          │
  │   │ │ Shields:        59%  [###--]│   │ Maneuver:  ● ON  [ACTIVE] │ │                                          │
  │   │ ╰─────────────────────────────╯   ╰───────────────────────────╯ │                                          │
  │   │   DRONE STATUS:  2 active (Survey #3, Harvest #1)               │                                          │
  │   │   Alert Level:   GREEN                                          │                                          │
  │   └─────────────────────────────────────────────────────────────────┘                                          │
  │                                                                                                               ┌╯
  │   ╭────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
  │   │  MINING & EXTRACTION                                                                                   │  │
  │   ╞════════════════════════════════════════════════════════════════════════════════════════════════════════╡  │
  │   │   SCAN: ASTEROID CLUSTER ALPHA-23                                                                      │  │
  │   │     - Iron Ore:       280 units   [###------] (28%)                                                    │  │
  │   │     - Titanium:       140 units   [##-------]  (14%)                                                   │  │
  │   │     - Rare Crystals:   12 units   [#--------]   (1.2%)                                                 │  │
  │   │     - Other Debris:   60 units    [#--------]   (6%)                                                   │  │
  │   │                                                                                                        │  │
  │   │   EXTRACTION:   ● ON  (Auto-Drill)                                                                     │  │
  │   │   PRIORITY:     [ Titanium ]                                                                           │  │
  │   │   CARGO SPACE:  71% used                                                                               │  │
  │   │   REFINING:     42%  [####-----]                                                                       │  │
  │   │   DISPOSAL:     ○ OFF                                                                                  │  │
  │   ╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯  │
  ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

         [TIME: 14:47]    [SECTOR: BETA-12]     [ALERT: GREEN]    [CREDITS: 7420]    [FPS: 60]
```