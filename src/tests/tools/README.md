# Space Muck Test Tools

This directory contains testing utilities and specialized test cases for the Space Muck project.

## Test Utilities

### Integration Tests
```bash
# Run integration tests
python -m src.tests.tools.integration_tests
```

### Performance Tests
```bash
# Run performance tests
python -m src.tests.tools.performance_tests
```

### Regression Tests
```bash
# Run regression tests
python -m src.tests.tools.regression_tests
```

### Unit Tests
```bash
# Run unit tests
python -m src.tests.tools.unit_tests
```

## Import Testing

The following scripts test various import scenarios to ensure proper module loading:

```bash
# Test basic imports
python -m src.tests.tools.test_basic_imports

# Test circular imports
python -m src.tests.tools.test_circular

# Test dependency configuration
python -m src.tests.tools.test_dependency_config

# Test import fixes
python -m src.tests.tools.test_import_fixes

# Test individual imports
python -m src.tests.tools.test_individual_imports

# Test runner configuration
python -m src.tests.tools.test_runner_config

# Test simple imports
python -m src.tests.tools.test_simple_imports

# Test single import
python -m src.tests.tools.test_single_import
```

## Component Tests

```bash
# Test combat system
python -m src.tests.tools.test_combat_system

# Test encounter generator
python -m src.tests.tools.test_encounter_generator
```

## Mock Data Testing

```bash
# Run mock data tests
python -m src.tests.tools.mock_data_test
```
