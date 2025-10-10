# AMULGENT

![Build Status](https://github.com/D0CT4/AMULGENT/workflows/Test/badge.svg)
![Lint Status](https://github.com/D0CT4/AMULGENT/workflows/Lint/badge.svg)
![Security Status](https://github.com/D0CT4/AMULGENT/workflows/Security/badge.svg)
![Coverage Status](https://github.com/D0CT4/AMULGENT/workflows/Coverage/badge.svg)
![Documentation Status](https://github.com/D0CT4/AMULGENT/workflows/Documentation/badge.svg)

## Overview

AMULGENT (AI Multi-Agent System with Unified Logic, Governance, and Efficient Network Topology) is an advanced workflow visualizer designed to build more secure systems with transparency in data uploaded for Small Language Models (SLMs) and Large Language Models (LLMs). The system implements Hierarchical Reasoning Model (HRM) reasoning to prioritize token usage and energy reduction, making it both efficient and environmentally conscious.

### Key Features

- **Hierarchical Reasoning Model (HRM)**: Intelligent token prioritization and energy-efficient processing
- **Multi-Agent System**: Coordinated agents for analysis, base operations, and HRM reasoning
- **Workflow Visualization**: Clear visualization of data flow and system operations
- **Security-First Design**: Transparent data handling with built-in security measures
- **Energy Optimization**: Reduced computational overhead through smart reasoning strategies

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/D0CT4/AMULGENT.git
cd AMULGENT

# Navigate to the AIMULGENT directory
cd AIMULGENT

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Development Installation

For development with all testing and linting tools:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install from pyproject.toml
pip install -e .
```

## Quick Start

### Basic Usage

```python
from aimulgent.core.system import AIMultiAgentSystem
from aimulgent.core.config import Config

# Initialize the system
config = Config()
system = AIMultiAgentSystem(config)

# Process a task
result = system.process("Your task here")
print(result)
```

### HRM Reasoning Example

```python
from aimulgent.agents.hrm_reasoning import HRMReasoning

# Initialize HRM Reasoning
hrm = HRMReasoning()

# Process with token optimization
result = hrm.reason(input_data, optimize_tokens=True)
print(f"Tokens used: {result.tokens_used}")
print(f"Energy saved: {result.energy_saved}%")
```

### Running the Demo

```bash
# Run the HRM demonstration
python demo_hrm.py
```

## Architecture

### System Architecture

```
AMULGENT/
├── aimulgent/              # Main package
│   ├── agents/            # Agent implementations
│   │   ├── base.py       # Base agent class
│   │   ├── analysis.py   # Analysis agent
│   │   └── hrm_reasoning.py  # HRM reasoning agent
│   ├── core/             # Core system components
│   │   ├── system.py     # Main system orchestration
│   │   ├── coordinator.py # Agent coordination
│   │   └── config.py     # Configuration management
│   ├── magic/            # Advanced features
│   └── main.py           # Entry point
├── core/                 # Additional core components
├── tests/                # Test suite
└── config/               # Configuration files
```

### Component Descriptions

#### Agents

1. **Base Agent** (`base.py`): Foundation for all agents with common functionality
2. **Analysis Agent** (`analysis.py`): Data analysis and processing
3. **HRM Reasoning Agent** (`hrm_reasoning.py`): Hierarchical reasoning with token optimization

#### Core System

1. **System** (`system.py`): Main orchestration and workflow management
2. **Coordinator** (`coordinator.py`): Inter-agent communication and task distribution
3. **Config** (`config.py`): System configuration and parameter management

#### Hierarchical Reasoning Model (HRM)

The HRM component uses a multi-tier reasoning strategy:

```
Tier 1: Fast Pattern Matching (Low Energy)
   ↓
Tier 2: Heuristic Processing (Medium Energy)
   ↓
Tier 3: Deep Analysis (High Energy)
   ↓
Tier 4: Full LLM Reasoning (Maximum Energy)
```

Each tier is only activated when necessary, optimizing both token usage and energy consumption.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aimulgent --cov-report=html

# Run specific test file
pytest tests/test_hrm_reasoning.py

# Run with verbose output
pytest -v
```

### Test Suite

The project includes comprehensive tests for:
- HRM Reasoning functionality
- Agent operations
- System coordination
- Integration scenarios
- Error handling and edge cases

## Development

### Code Quality

We maintain high code quality standards using:

```bash
# Format code with Black
black aimulgent/

# Sort imports with isort
isort aimulgent/

# Lint with flake8
flake8 aimulgent/

# Type checking with mypy
mypy aimulgent/

# Security checks
bandit -r aimulgent/
safety check
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pip install pre-commit
pre-commit install
```

## Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AMULGENT.git
   cd AMULGENT
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Ensure all tests pass
   - Follow the existing code style

4. **Run Quality Checks**
   ```bash
   pytest
   black aimulgent/
   flake8 aimulgent/
   mypy aimulgent/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Describe your changes
   - Reference any related issues (e.g., "Fixes #2")

### Contribution Guidelines

- **Code Style**: Follow PEP 8 guidelines
- **Testing**: Maintain or increase test coverage
- **Documentation**: Update docs for new features
- **Commits**: Write clear, descriptive commit messages
- **Issues**: Reference related issues in PR descriptions

### Reporting Issues

When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Test**: Automated testing with pytest
- **Lint**: Code quality checks with flake8, black, mypy, isort
- **Security**: Security scanning with bandit and safety
- **Coverage**: Code coverage reporting
- **Documentation**: Automated documentation builds

All workflows must pass before merging pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

## Acknowledgments

- Thanks to all contributors who help improve AMULGENT
- Inspired by modern AI workflow optimization techniques
- Built with focus on sustainability and energy efficiency

## Contact

For questions, suggestions, or issues:
- Create an issue: [GitHub Issues](https://github.com/D0CT4/AMULGENT/issues)
- Repository: [https://github.com/D0CT4/AMULGENT](https://github.com/D0CT4/AMULGENT)

## Related Documentation

- [HRM Enhancements](AIMULGENT/HRM_ENHANCEMENTS.md) - Detailed HRM feature documentation
- [AIMULGENT README](AIMULGENT/README.md) - Package-specific documentation

---

**Built with ❤️ for efficient and transparent AI systems**
