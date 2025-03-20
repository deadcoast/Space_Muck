# System Diagrams

## Key System Relationships

```
SYSTEM RELATIONSHIPS
Entity System <-> Context Providers <-> Event System
UI Components <-> Manager Systems <-> Resource System
Procedural Gen <-> Exploration System
```

```
+----------------+      +--------------------+      +----------------+
|                |      |                    |      |                |
| Entity System  <----->+ Context Providers  <----->+ Event System   |
|                |      |                    |      |                |
+-------^--------+      +---------^----------+      +-------^--------+
        |                        |                         |
        |                        |                         |
        v                        v                         v
+----------------+      +--------------------+      +----------------+
|                |      |                    |      |                |
| UI Components  <----->+ Manager Systems    <----->+ Resource System|
|                |      |                    |      |                |
+-------^--------+      +---------^----------+      +----------------+
        |                        |                         
        |                        |                         
        v                        v                         
+----------------+      +--------------------+      
|                |      |                    |      
| Procedural Gen <----->+ Exploration System |      
|                |      |                    |      
+----------------+      +--------------------+      
```