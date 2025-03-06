# Space Muck Verifications

This directory contains verification scripts for various Space Muck components. These scripts validate the behavior and output of game components in different scenarios.

## Running Verification Scripts

### Asteroid Generator Verifications
```bash
# Run asteroid generator verification
python -m src.tests.verifications.verify_asteroid_generator

# Run simplified asteroid generator verification
python -m src.tests.verifications.verify_asteroid_generator_simple
```

### Entity Verifications
```bash
# Verify base entity functionality
python -m src.tests.verifications.verify_base_entity

# Verify fleet functionality
python -m src.tests.verifications.verify_fleet

# Verify player functionality
python -m src.tests.verifications.verify_player
```

### Generator Verifications
```bash
# Verify base generator optimizations
python -m src.tests.verifications.verify_base_generator_optimizations

# Verify procedural generator
python -m src.tests.verifications.verify_procedural_generator

# Verify symbiote evolution generator
python -m src.tests.verifications.verify_symbiote_evolution_generator

# Verify simplified symbiote evolution generator
python -m src.tests.verifications.verify_symbiote_evolution_generator_simple
```

## Verification vs. Testing

Verification scripts differ from unit tests in the following ways:

1. **Purpose**: Verification scripts validate behavior in realistic scenarios, while unit tests focus on isolated component functionality.

2. **Output**: Verification scripts often produce visual or detailed textual output for manual inspection, while unit tests produce pass/fail results.

3. **Coverage**: Verification scripts typically cover end-to-end workflows, while unit tests focus on specific functions or methods.

Use verification scripts when you need to validate complex behavior or visualize the output of game components.
