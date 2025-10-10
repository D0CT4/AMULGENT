# AIMULGENT Python SDK

> Official Python client library for AIMULGENT multi-agent system

## Overview

The AIMULGENT Python SDK provides a convenient interface for interacting with the AIMULGENT system from Python applications. It supports code analysis, agent coordination, and system monitoring.

## Installation

```bash
pip install aimulgent-sdk
```

Or install from source:

```bash
git clone https://github.com/D0CT4/AMULGENT.git
cd AMULGENT/AIMULGENT/sdk/python
pip install -e .
```

## Quick Start

```python
from aimulgent_sdk import AIMULGENTClient

# Initialize client
client = AIMULGENTClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"  # Optional
)

# Analyze code
code = """
def example_function(x, y):
    return x + y
"""

result = await client.analyze_code(
    code=code,
    language="python"
)

print(f"Quality Score: {result['quality_score']}")
print(f"Issues Found: {len(result['issues'])}")

# Get system status
status = await client.get_system_status()
print(f"System Status: {status['status']}")
print(f"Active Agents: {status['agents_active']}")
```

## Features

### Code Analysis

```python
# Analyze code with options
result = await client.analyze_code(
    code=code,
    language="python",
    options={
        "check_security": True,
        "check_complexity": True,
        "check_style": True
    }
)

# Access results
for issue in result['issues']:
    print(f"[{issue['severity']}] {issue['message']}")
    print(f"  Line {issue['line']}: {issue['code']}")
```

### System Monitoring

```python
# Get health status
health = await client.get_health()
print(f"Status: {health['status']}")
print(f"Uptime: {health['uptime_seconds']}s")

# Get metrics
metrics = await client.get_metrics()
print(f"Tasks Processed: {metrics['tasks_total']}")
print(f"CPU Usage: {metrics['system_cpu_percent']}%")
```

### Agent Management

```python
# List available agents
agents = await client.list_agents()
for agent in agents:
    print(f"Agent: {agent['name']} - Status: {agent['status']}")

# Send task to specific agent
result = await client.send_agent_task(
    agent_id="analysis-agent-001",
    task_type="code_review",
    payload={"code": code}
)
```

## Configuration

### Environment Variables

```bash
export AIMULGENT_BASE_URL=http://localhost:8000
export AIMULGENT_API_KEY=your-api-key
export AIMULGENT_TIMEOUT=30
```

### Configuration File

Create `~/.aimulgent/config.json`:

```json
{
  "base_url": "http://localhost:8000",
  "api_key": "your-api-key",
  "timeout": 30,
  "retry_attempts": 3
}
```

## API Reference

### AIMULGENTClient

Main client class for interacting with AIMULGENT.

#### Methods

- `analyze_code(code: str, language: str, options: dict = None) -> dict`
  - Analyze code and return quality metrics

- `get_system_status() -> dict`
  - Get comprehensive system status

- `get_health() -> dict`
  - Get health check information

- `get_metrics() -> dict`
  - Get system metrics in Prometheus format

- `list_agents() -> list`
  - List all available agents

- `send_agent_task(agent_id: str, task_type: str, payload: dict) -> dict`
  - Send task to specific agent

## Error Handling

```python
from aimulgent_sdk.exceptions import (
    AIMULGENTError,
    ConnectionError,
    AuthenticationError,
    ValidationError
)

try:
    result = await client.analyze_code(code)
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except ValidationError as e:
    print(f"Invalid input: {e}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except AIMULGENTError as e:
    print(f"AIMULGENT error: {e}")
```

## Examples

See the [examples](./examples/) directory for complete usage examples:

- `basic_usage.py` - Basic code analysis
- `advanced_analysis.py` - Advanced analysis with options
- `monitoring.py` - System monitoring and metrics
- `agent_coordination.py` - Multi-agent workflows

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/D0CT4/AMULGENT.git
cd AMULGENT/AIMULGENT/sdk/python

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy aimulgent_sdk

# Format code
black aimulgent_sdk
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Support

- **Documentation**: [Full Documentation](https://github.com/D0CT4/AMULGENT/wiki)
- **Issues**: [GitHub Issues](https://github.com/D0CT4/AMULGENT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/D0CT4/AMULGENT/discussions)
