# API Reference

Welcome to the AMULGENT API Reference documentation. This section provides comprehensive documentation for all public modules, classes, and functions in the AMULGENT package.

## Overview

AMULGENT (AI Multi-Agent System with Unified Logic, Governance, and Efficient Network Topology) provides a powerful API for building intelligent multi-agent systems with hierarchical reasoning capabilities.

## Package Structure

```
aimulgent/
├── agents/          # Agent implementations
│   ├── base.py      # Base agent class
│   ├── analysis.py  # Analysis agent
│   └── hrm_reasoning.py  # HRM reasoning agent
├── core/            # Core system components
│   ├── system.py    # Main system orchestration
│   ├── coordinator.py  # Agent coordination
│   └── config.py    # Configuration management
└── magic/           # Advanced features
```

## Quick Navigation

### [Agents](agents/base.md)

Agent implementations for various tasks:

- **[Base Agent](agents/base.md)** - Foundation for all agents
- **[Analysis Agent](agents/analysis.md)** - Data analysis and processing
- **[HRM Reasoning Agent](agents/hrm_reasoning.md)** - Hierarchical reasoning with token optimization

### [Core System](core/system.md)

Core system components:

- **[System](core/system.md)** - Main orchestration and workflow management
- **[Coordinator](core/coordinator.md)** - Inter-agent communication and task distribution
- **[Config](core/config.md)** - System configuration and parameter management

### [Magic](magic/index.md)

Advanced features and utilities.

## Key Features

### Hierarchical Reasoning Model (HRM)

The HRM component uses a multi-tier reasoning strategy that optimizes token usage and energy consumption:

```python
from aimulgent.agents.hrm_reasoning import HRMReasoning

# Initialize HRM Reasoning
hrm = HRMReasoning()

# Process with token optimization
result = hrm.reason(input_data, optimize_tokens=True)
```

**Reasoning Tiers:**

1. **Tier 1**: Fast Pattern Matching (Low Energy)
2. **Tier 2**: Heuristic Processing (Medium Energy)
3. **Tier 3**: Deep Analysis (High Energy)
4. **Tier 4**: Full LLM Reasoning (Maximum Energy)

Each tier is only activated when necessary, optimizing both token usage and energy consumption.

### Multi-Agent System

Coordinate multiple agents for complex tasks:

```python
from aimulgent.core.system import AIMultiAgentSystem
from aimulgent.core.config import Config

# Initialize the system
config = Config()
system = AIMultiAgentSystem(config)

# Process a task
result = system.process("Your task here")
```

### Workflow Visualization

Visualize data flow and system operations for transparency and security.

## Usage Examples

### Basic Usage

```python
from aimulgent.core.system import AIMultiAgentSystem
from aimulgent.core.config import Config

# Initialize the system
config = Config()
system = AIMultiAgentSystem(config)

# Process a task
result = system.process("Analyze this data")
print(result)
```

### HRM Reasoning Example

```python
from aimulgent.agents.hrm_reasoning import HRMReasoning

# Initialize HRM Reasoning
hrm = HRMReasoning()

# Process with token optimization
result = hrm.reason(
    input_data="Complex task requiring analysis",
    optimize_tokens=True
)

print(f"Tokens used: {result.tokens_used}")
print(f"Energy saved: {result.energy_saved}%")
print(f"Result: {result.output}")
```

### Configuration Example

```python
from aimulgent.core.config import Config

# Create configuration
config = Config(
    max_tokens=1000,
    temperature=0.7,
    enable_caching=True
)

# Update configuration
config.update(
    max_tokens=2000,
    enable_hrm=True
)
```

## API Conventions

### Docstring Format

All public APIs use Google-style docstrings:

```python
def function_name(arg1: str, arg2: int) -> dict:
    """Brief description of the function.
    
    Longer description with more details about what the function
    does and how it should be used.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When arg2 is negative
        TypeError: When arg1 is not a string
    
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        {'status': 'success'}
    """
    pass
```

### Type Hints

All public APIs use type hints for better IDE support and type checking:

```python
from typing import List, Dict, Optional, Union

def process_data(
    data: List[Dict[str, Union[str, int]]],
    options: Optional[Dict[str, bool]] = None
) -> Dict[str, any]:
    """Process data with optional configuration."""
    pass
```

### Error Handling

APIs use standard Python exceptions with descriptive messages:

```python
try:
    result = system.process(data)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

## API Stability

### Version Compatibility

AMULGENT follows semantic versioning:

- **Major version** (X.0.0): Breaking changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, backward compatible

### Deprecation Policy

Deprecated APIs:

1. Are marked with `@deprecated` decorator
2. Include migration instructions in docstrings
3. Remain available for at least one minor version
4. Issue warnings when used

```python
import warnings

@deprecated("Use new_function() instead")
def old_function():
    """This function is deprecated.
    
    .. deprecated:: 0.2.0
        Use :func:`new_function` instead.
    """
    warnings.warn(
        "old_function is deprecated, use new_function",
        DeprecationWarning
    )
```

## Performance Considerations

### Token Optimization

HRM reasoning automatically optimizes token usage:

- **Fast tier**: ~10-50 tokens
- **Medium tier**: ~100-500 tokens
- **Deep tier**: ~500-2000 tokens
- **Full LLM**: 2000+ tokens

### Caching

Enable caching for frequently accessed data:

```python
config = Config(enable_caching=True, cache_ttl=3600)
```

### Parallel Processing

Multiple agents can process tasks in parallel:

```python
system = AIMultiAgentSystem(
    config,
    max_workers=4  # Number of parallel workers
)
```

## Testing

### Unit Tests

All public APIs have comprehensive unit tests:

```bash
pytest tests/test_hrm_reasoning.py
pytest tests/test_system.py
```

### Integration Tests

Test complete workflows:

```bash
pytest tests/integration/
```

### Type Checking

Verify type hints:

```bash
mypy aimulgent/
```

## Contributing

When adding new APIs:

1. Add comprehensive docstrings
2. Include type hints
3. Write unit tests
4. Update this documentation
5. Add usage examples
6. Follow the code style guide

See the [Contributing Guide](../development/contributing.md) for more details.

## Support

For API questions and issues:

- [GitHub Issues](https://github.com/D0CT4/AMULGENT/issues)
- [GitHub Discussions](https://github.com/D0CT4/AMULGENT/discussions)
- Reference Issue #2 for documentation improvements

## License

AMULGENT is licensed under the MIT License. See [LICENSE](../about/license.md) for details.

---

**Next Steps:**

- Explore [Agents API](agents/base.md)
- Learn about [Core System](core/system.md)
- Check out [Examples](../examples/basic.md)
- Read the [User Guide](../user-guide/getting-started.md)
