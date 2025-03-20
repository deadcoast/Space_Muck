# Converter Management UI

## Overview

The Converter Management Interface provides players with a comprehensive ASCII UI for monitoring and controlling resource conversion processes, including multi-step production chains. This document outlines the design, components, and implementation details for this interface.

## Design Goals

1. **Intuitive Production Chain Management**: Allow players to easily visualize and manage complex production chains in ASCII Aesthetics
2. **Real-time Efficiency Monitoring**: Provide clear visual feedback on conversion efficiency factors through clear ASCII UI
3. **Interactive Control**: Enable hands-on management of conversion processes
4. **Clear Resource Visualization**: Visualize resource flows and conversions through ASCII UI and live effects (Spinners, Progress bars, color changing, etc.)
5. **Chain Optimization Tools**: Help players optimize their production chains
6. **Accessibility**: Ensure the interface is usable with keyboard navigation and screen readers
7. **Real-time Updates**: Maintain live status updates through event system integration
8. **Template Management**: Enable saving and loading of common production chain configurations

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

### 1. ConverterDashboard Component

Main entry point for converter management:

```typescript
interface ConverterDashboardProps {
  converters: ConverterSummary[];
  activeProcesses: ConversionProcessSummary[];
  activeChains: ChainStatusSummary[];
  onConverterSelect: (converterId: string) => void;
  onProcessSelect: (processId: string) => void;
  onChainSelect: (chainId: string) => void;
}

const ConverterDashboard: React.FC<ConverterDashboardProps> = ({
  converters,
  activeProcesses,
  activeChains,
  onConverterSelect,
  onProcessSelect,
  onChainSelect,
}) => {
  // Implementation
};
```

## Implementation Status and Components

### Current Implementation

#### 1. Core UI Framework
- [x] ASCII Box and Panel components
- [x] Progress bar component
- [x] Button component
- [x] Color-coded text support
- [x] Basic keyboard navigation

#### 2. Converter Management
- [x] Converter dashboard with list view
- [x] Converter details view
- [x] Basic process management
- [x] Efficiency monitoring
- [x] Chain management interface

### Pending Implementation

#### 1. Enhanced Visualization
- [ ] ASCII flow diagrams for chains
- [ ] Resource flow animations
- [ ] Live status indicators
- [ ] Historical efficiency graphs

#### 2. Template System
- [ ] Chain template storage
- [ ] Template management UI
- [ ] Quick-load templates
- [ ] Template sharing

#### 3. Performance Metrics
- [ ] Throughput tracking
- [ ] Energy usage monitoring
- [ ] Utilization statistics
- [ ] Uptime tracking

#### 4. Accessibility Features
- [ ] Full keyboard navigation
- [ ] Screen reader support
- [ ] High contrast mode
- [ ] Configurable update rates

#### 5. Real-time Updates
- [ ] Event system integration
- [ ] Periodic refresh mechanism
- [ ] Status change notifications
- [ ] Performance optimization

## Implementation Priorities

### Phase 1: Core Functionality Enhancement
1. **Performance Metrics**
   - Implement throughput tracking
   - Add energy usage monitoring
   - Integrate utilization statistics
   Priority: High | Effort: Medium | Dependencies: None

2. **Real-time Updates**
   - Develop event system integration
   - Implement periodic refresh mechanism
   - Add status change notifications
   Priority: High | Effort: High | Dependencies: None

### Phase 2: Visual Improvements
1. **Chain Visualization**
   - Create ASCII flow diagrams
   - Implement resource flow animations
   - Add live status indicators
   Priority: Medium | Effort: High | Dependencies: Real-time Updates

2. **Historical Data**
   - Implement efficiency graphs
   - Add trend tracking
   - Create performance history views
   Priority: Medium | Effort: Medium | Dependencies: Performance Metrics

### Phase 3: User Experience
1. **Template System**
   - Implement template storage
   - Create template UI
   - Add sharing functionality
   Priority: Low | Effort: Medium | Dependencies: None

2. **Accessibility**
   - Enhance keyboard navigation
   - Add screen reader support
   - Implement high contrast mode
   Priority: High | Effort: High | Dependencies: None

## Development Roadmap

### Sprint 1: Core Metrics (Current)
- [ ] Implement base event system
- [ ] Add throughput tracking
- [ ] Integrate energy monitoring
- [ ] Create utilization metrics
- [ ] Set up periodic refresh system
Target Completion: Week 1

### Sprint 2: Visual Enhancements
- [ ] Design ASCII flow diagrams
- [ ] Create animation system for resource flows
- [ ] Implement live status indicators
- [ ] Add efficiency trend arrows
- [ ] Design historical graphs
Target Completion: Week 2

### Sprint 3: Chain Management
- [ ] Build chain template system
- [ ] Create template UI
- [ ] Implement save/load functionality
- [ ] Add chain validation
- [ ] Create step configuration UI
Target Completion: Week 3

### Sprint 4: Accessibility & Polish
- [ ] Enhance keyboard navigation
- [ ] Add screen reader support
- [ ] Implement high contrast mode
- [ ] Performance optimization
- [ ] Bug fixes and refinements
Target Completion: Week 4

## Progress Tracking

### Completed Features
- [x] Core UI Framework
  - ASCII Box and Panel components
  - Progress bar component
  - Button component
  - Color-coded text support
  - Basic keyboard navigation

- [x] Basic Converter Management
  - Converter dashboard with list view
  - Converter details view with stats
  - Process management controls
  - Chain management interface
  - Efficiency monitoring panel

### In Progress (Sprint 1)
- [ ] Event System Integration
  - Base event handler implementation
  - Metric collection system
  - Real-time update mechanism
  - Performance monitoring

### Blocked Features
1. ASCII Flow Diagrams
   - Blocked by: Event system for real-time updates
   - Required for: Chain visualization

2. Historical Graphs
   - Blocked by: Metric collection system
   - Required for: Efficiency monitoring

3. Template System
   - Blocked by: Chain validation system
   - Required for: Production optimization

### Known Issues
1. Performance
   - Refresh rate needs optimization
   - Memory usage in large chain displays
   - Event handler overhead

2. Accessibility
   - Keyboard navigation incomplete
   - Missing screen reader support
   - No high contrast mode

3. UI/UX
   - Chain visualization needs improvement
   - Missing visual feedback in some interactions
   - Template UI not user-friendly

## Testing and Quality Assurance

### Unit Tests
1. Core Components
   - ASCII Box rendering and boundaries
   - Progress bar calculations
   - Button state management
   - Color handling and contrast ratios

2. Event System
   - Event registration and deregistration
   - Event propagation and handling
   - Metric collection accuracy
   - Cache management and cleanup

3. Template System
   - Template serialization/deserialization
   - Validation rules
   - Version compatibility
   - Error handling

### Integration Tests
1. UI Component Integration
   - Component lifecycle management
   - State propagation between components
   - Resource cleanup

2. Performance Testing
   - Memory usage monitoring
   - CPU utilization tracking
   - Refresh rate optimization
   - Large dataset handling

3. Accessibility Testing
   - Screen reader compatibility
   - Keyboard navigation paths
   - Color contrast compliance
   - Input device support

### Acceptance Criteria
1. Performance Metrics
   - UI updates < 16ms (60 FPS)
   - Memory usage < 100MB
   - Event processing < 5ms

2. Accessibility Standards
   - WCAG 2.1 Level AA compliance
   - Full keyboard navigation
   - Screen reader support
   - High contrast mode

3. User Experience
   - < 3 clicks for common actions
   - Clear error messages
   - Consistent feedback
   - Intuitive navigation

## Code Review Guidelines

### General Standards
1. Code Organization
   - Follow PEP 8 style guide
   - Use type hints consistently
   - Document all public interfaces
   - Keep functions focused and small

2. Performance Considerations
   - Profile critical sections
   - Optimize render loops
   - Cache expensive computations
   - Use appropriate data structures

3. Error Handling
   - Use descriptive error messages
   - Implement proper exception handling
   - Add logging for debugging
   - Validate user input

### UI Component Standards
1. Component Structure
   - Clear separation of concerns
   - Consistent naming conventions
   - Proper event propagation
   - Resource cleanup

2. Accessibility
   - Screen reader compatibility
   - Keyboard navigation support
   - Color contrast requirements
   - Input device handling

3. Documentation
   - Clear component purpose
   - Usage examples
   - Parameter descriptions
   - Known limitations

### Contribution Process
1. Pre-submission
   - Run unit tests
   - Check performance metrics
   - Verify accessibility
   - Update documentation

2. Code Review
   - Functionality review
   - Performance analysis
   - Accessibility check
   - Documentation review

3. Post-merge
   - Update changelog
   - Monitor metrics
   - Gather feedback
   - Plan improvements

## Monitoring and Metrics

### Performance Metrics
1. Rendering Performance
   - Frame time (target: < 16ms)
   - Memory usage per component
   - Event processing latency
   - UI update frequency

2. User Interaction
   - Time to complete common actions
   - Error frequency and types
   - Feature usage patterns
   - Navigation paths

3. System Health
   - Component initialization time
   - Memory leak detection
   - Event queue size
   - Cache hit rates

### Monitoring Tools
1. Performance Profiler
   ```python
   class UIProfiler:
       def __init__(self):
           self.metrics: Dict[str, float] = {}
           self.start_time: float = 0.0

       def start_operation(self, name: str) -> None:
           self.start_time = time.time()

       def end_operation(self, name: str) -> None:
           duration = time.time() - self.start_time
           if name not in self.metrics:
               self.metrics[name] = duration
           else:
               self.metrics[name] = 0.9 * self.metrics[name] + 0.1 * duration
   ```

2. Usage Analytics
   ```python
   class UIAnalytics:
       def __init__(self):
           self.interactions: List[Dict[str, Any]] = []
           self.error_log: List[Dict[str, Any]] = []

       def log_interaction(self, component: str, action: str) -> None:
           self.interactions.append({
               'component': component,
               'action': action,
               'timestamp': time.time()
           })

       def log_error(self, component: str, error: Exception) -> None:
           self.error_log.append({
               'component': component,
               'error': str(error),
               'timestamp': time.time()
           })
   ```

### Reporting
1. Real-time Metrics
   - Component performance
   - Error rates
   - User activity
   - System health

2. Historical Analysis
   - Performance trends
   - Usage patterns
   - Error patterns
   - Resource utilization

3. Alerts
   - Performance degradation
   - High error rates
   - Resource exhaustion
   - System instability

## Implementation Guidelines

### Base Component Structure
```python
class UIComponent:
    def __init__(self):
        self.update_interval: float = 1.0  # seconds
        self.last_update: float = 0.0
        self.needs_refresh: bool = False
        self.accessibility_mode: bool = False
        self.high_contrast: bool = False
    
    def update(self, delta_time: float) -> None:
        self.last_update += delta_time
        if self.last_update >= self.update_interval:
            self.refresh()
            self.last_update = 0.0
    
    def refresh(self) -> None:
        if self.needs_refresh:
            self.redraw()
            self.needs_refresh = False
            
    def toggle_accessibility(self) -> None:
        self.accessibility_mode = not self.accessibility_mode
        self.needs_refresh = True
        
    def toggle_high_contrast(self) -> None:
        self.high_contrast = not self.high_contrast
        self.needs_refresh = True
```

### Event System
```python
class EventListener:
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.metrics_cache: Dict[str, Any] = {}
        self.last_update_time: float = 0.0
    
    def subscribe(self, event: str, handler: Callable) -> None:
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
    
    def notify(self, event: str, data: Any) -> None:
        if event in self.handlers:
            # Cache metrics data for history
            if event.startswith('metric_'):
                self.metrics_cache[event] = {
                    'value': data,
                    'timestamp': time.time()
                }
            # Notify handlers
            for handler in self.handlers[event]:
                handler(data)
                
    def get_metric_history(self, metric: str, duration: float) -> List[Dict[str, Any]]:
        current_time = time.time()
        return [data for data in self.metrics_cache.get(metric, [])
                if current_time - data['timestamp'] <= duration]
```

### Template System
```python
class ChainTemplate:
    def __init__(self):
        self.name: str = ""
        self.steps: List[ChainStep] = []
        self.metadata: Dict[str, Any] = {}
        self.version: str = "1.0"
        self.created_at: float = time.time()
        self.last_modified: float = time.time()
    
    def save(self) -> None:
        self.last_modified = time.time()
        template_data = {
            'name': self.name,
            'steps': [step.to_dict() for step in self.steps],
            'metadata': self.metadata,
            'version': self.version,
            'created_at': self.created_at,
            'last_modified': self.last_modified
        }
        # Save to file/database
        pass
    
    @classmethod
    def load(cls, name: str) -> 'ChainTemplate':
        # Load from file/database
        template = cls()
        # Populate template data
        return template
        
    def validate(self) -> bool:
        # Validate template integrity
        return all(step.is_valid() for step in self.steps)
```

### 2. ConverterDetailsView Component

Detailed view of a single converter:

```typescript
interface ConverterDetailsViewProps {
  converter: ConverterDetail;
  activeProcesses: ConversionProcessDetail[];
  availableRecipes: RecipeDetail[];
  onStartProcess: (recipeId: string) => void;
  onPauseProcess: (processId: string) => void;
  onStopProcess: (processId: string) => void;
  onBack: () => void;
}

const ConverterDetailsView: React.FC<ConverterDetailsViewProps> = ({
  converter,
  activeProcesses,
  availableRecipes,
  onStartProcess,
  onPauseProcess,
  onStopProcess,
  onBack,
}) => {
  // Implementation
};
```

### 3. ChainManagementInterface Component

Interface for managing production chains:

```typescript
interface ChainManagementInterfaceProps {
  activeChains: ChainStatus[];
  availableConverters: ConverterSummary[];
  availableRecipes: RecipeDetail[];
  savedTemplates: ChainTemplate[];
  onStartChain: (steps: ChainStep[]) => void;
  onPauseChain: (chainId: string) => void;
  onResumeChain: (chainId: string) => void;
  onCancelChain: (chainId: string) => void;
  onSaveTemplate: (template: ChainTemplate) => void;
  onLoadTemplate: (templateId: string) => void;
  onDeleteTemplate: (templateId: string) => void;
}

const ChainManagementInterface: React.FC<ChainManagementInterfaceProps> = ({
  activeChains,
  availableConverters,
  availableRecipes,
  savedTemplates,
  onStartChain,
  onPauseChain,
  onResumeChain,
  onCancelChain,
  onSaveTemplate,
  onLoadTemplate,
  onDeleteTemplate,
}) => {
  // Implementation
};
```

### 4. EfficiencyMonitor Component

Visualizes efficiency factors and history:

```typescript
interface EfficiencyMonitorProps {
  converter: ConverterDetail;
  efficiencyHistory: EfficiencyHistoryEntry[];
  optimizationSuggestions: OptimizationSuggestion[];
}

const EfficiencyMonitor: React.FC<EfficiencyMonitorProps> = ({
  converter,
  efficiencyHistory,
  optimizationSuggestions,
}) => {
  // Implementation
};
```

### 5. ChainVisualization Component

Visualizes production chains as a graph:

```typescript
interface ChainVisualizationProps {
  chain: ChainStatus;
  converters: Record<string, ConverterSummary>;
  recipes: Record<string, RecipeDetail>;
  interactive?: boolean;
  onNodeClick?: (nodeId: string, type: 'converter' | 'recipe') => void;
}

const ChainVisualization: React.FC<ChainVisualizationProps> = ({
  chain,
  converters,
  recipes,
  interactive = false,
  onNodeClick,
}) => {
  // Implementation using D3 or React Flow for visualization
};
```

## UI/UX Considerations

### Visual Design Elements

1. **Color Coding**:

   - Green: Optimal efficiency (>90%)
   - Yellow: Moderate efficiency (70-90%)
   - Orange: Low efficiency (50-70%)
   - Red: Poor efficiency (<50%)
   - Blue: Inactive or paused

2. **Icons**:

   - Different icons for converter types (smelter, assembler, refinery)
   - Status icons (active, paused, error)
   - Resource type icons

3. **Progress Indicators**:
   - Circular progress for individual processes
   - Linear progress for chains
   - Animated flows for resource transfers

### Interaction Patterns

1. **Drag and Drop**:

   - Drag recipes to converters to start processes
   - Drag and drop chain steps to reorder them
   - Drag resources between storage and converters

2. **Contextual Controls**:

   - Right-click menus for common actions
   - Hover tooltips with detailed information
   - Double-click to open detailed views

3. **Real-time Updates**:
   - Efficiency graphs update in real-time
   - Process progress bars animate smoothly
   - Resource levels change visually as they're consumed/produced

## Implementation Plan

### Phase 1: Core Components

1. Develop `ConverterDashboard` component
2. Implement `ConverterDetailsView` component
3. Create basic process management controls
4. Implement simple efficiency display

### Phase 2: Chain Management

1. Develop `ChainManagementInterface` component
2. Implement `ChainVisualization` component
3. Create chain step configuration UI
4. Add template saving/loading functionality

### Phase 3: Advanced Features

1. Implement `EfficiencyMonitor` with historical tracking
2. Add optimization suggestions
3. Create interactive chain visualization
4. Implement drag-and-drop functionality

### Phase 4: Polish and Integration

1. Apply consistent styling and theming
2. Add animations and transitions
3. Implement responsive design for different screen sizes
4. Integrate with game event system for real-time updates

## Accessibility Considerations

1. **Keyboard Navigation**:

   - All interactive elements should be keyboard accessible
   - Implement logical tab order
   - Add keyboard shortcuts for common actions

2. **Screen Reader Support**:

   - Add proper ARIA labels and roles
   - Ensure meaningful text alternatives for visual elements
   - Implement announcements for state changes

3. **Color Considerations**:
   - Ensure sufficient color contrast
   - Don't rely solely on color to convey information
   - Provide alternative visual indicators (patterns, icons)

## Performance Considerations

1. **Component Optimization**:

   - Use React.memo for components that render frequently
   - Implement virtualization for long lists
   - Use requestAnimationFrame for animations

2. **Data Management**:

   - Implement data caching for converter and recipe information
   - Use efficient data structures for quick lookups
   - Batch updates to minimize render cycles

3. **Rendering Optimization**:
   - Use CSS transitions instead of JS animations where possible
   - Optimize SVG rendering for chain visualizations
   - Implement progressive loading for complex views
