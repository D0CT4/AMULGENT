# Frequently Asked Questions (FAQ)

## Table of Contents
- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Usage & Configuration](#usage--configuration)
- [Troubleshooting](#troubleshooting)
- [Performance & Optimization](#performance--optimization)
- [Contributing & Development](#contributing--development)

---

## General Questions

### What is AMULGENT?
AMULGENT (AI Multi-Agent System with Unified Logic, Governance, and Efficient Network Topology) is an advanced workflow visualizer designed to build more secure systems with transparency in data uploaded for Small Language Models (SLMs) and Large Language Models (LLMs). It implements Hierarchical Reasoning Model (HRM) reasoning to prioritize token usage and energy reduction.

### What are the key benefits of using AMULGENT?
- **Energy Efficiency**: Up to 40% reduction in computational overhead
- **Cost Savings**: 30-50% reduction in API costs through intelligent token optimization
- **Security**: Built-in security measures and transparent data handling
- **Visualization**: Clear workflow visualization for better system understanding
- **Flexibility**: Multi-agent architecture for diverse use cases

### What programming languages does AMULGENT support?
AMULGENT is primarily written in Python (99.3%) with Docker support for containerized deployments.

### Is AMULGENT free to use?
Yes, AMULGENT is open-source and free to use under the MIT License.

### What's the difference between HRM and traditional LLM reasoning?
Traditional LLM reasoning sends every query directly to the language model, consuming tokens and energy regardless of query complexity. HRM uses a three-tier approach:
- **Tier 1**: Quick heuristic evaluation (< 10ms) for simple queries
- **Tier 2**: Rule-based reasoning (< 100ms) for moderate complexity
- **Tier 3**: Full LLM reasoning (only when necessary)

This reduces unnecessary LLM calls by 60-80% in typical workloads.

---

## Installation & Setup

### Q: What are the system requirements?
**Minimum Requirements:**
- Python 3.8 or higher
- 2GB RAM
- 500MB disk space

**Recommended:**
- Python 3.10+
- 4GB+ RAM
- 1GB disk space
- Docker (for containerized deployment)

### Q: I'm getting "ModuleNotFoundError: No module named 'aimulgent'"
**Solution:**
```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Q: How do I verify the installation?
```bash
# Run the demo
python -m aimulgent.demo_hrm

# Or import in Python
python -c "import aimulgent; print(aimulgent.__version__)"
```

### Q: Can I use AMULGENT with virtual environments?
Yes! We highly recommend using virtual environments:
```bash
# Create virtual environment
python -m venv venv

# Activate (Unix/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install AMULGENT
pip install -r requirements.txt
```

### Q: Docker installation fails. What should I do?
Common issues:
1. **Docker not installed**: Install Docker Desktop from docker.com
2. **Permission denied**: Run with `sudo` or add your user to the docker group
3. **Port already in use**: Change the port in `docker-compose.yml`

```bash
# Check Docker is running
docker --version

# Build and run
docker-compose up --build
```

---

## Usage & Configuration

### Q: How do I configure the HRM tier thresholds?
Edit the configuration in your code or config file:
```python
from aimulgent.core import HRMEngine

engine = HRMEngine(
    tier_thresholds=[0.3, 0.7],  # Tier 1 -> 2 at 30%, Tier 2 -> 3 at 70%
    enable_caching=True
)
```

Or in `config/default.yaml`:
```yaml
hrm:
  tier_thresholds: [0.3, 0.7]
  enable_caching: true
  cache_ttl: 3600
```

### Q: Can I use my own LLM provider (not OpenAI)?
Yes! AMULGENT supports multiple providers:
```python
from aimulgent.integrations import LLMProvider

# Anthropic Claude
provider = LLMProvider(
    provider_type="anthropic",
    api_key="your-api-key",
    model="claude-3-opus"
)

# HuggingFace
provider = LLMProvider(
    provider_type="huggingface",
    model="gpt2-large"
)

# Custom endpoint
provider = LLMProvider(
    provider_type="custom",
    endpoint="https://your-llm-api.com"
)
```

### Q: How do I visualize my workflow?
```python
from aimulgent.visualization import WorkflowVisualizer

visualizer = WorkflowVisualizer(system)
visualizer.generate_graph(
    output_path="output/workflow.html",
    show_metrics=True,
    interactive=True
)
```

### Q: Can I customize the agents?
Yes! Extend the base agent class:
```python
from aimulgent.agents import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
    
    def process(self, data):
        # Your custom logic here
        return processed_data
```

---

## Troubleshooting

### Q: The pre-commit hooks are failing
**Solution:**
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run hooks manually
pre-commit run --all-files

# If specific hooks fail, check the tool configurations
# in .pre-commit-config.yaml
```

### Q: Import errors with dependencies
**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# For development dependencies
pip install -e ".[dev]"

# Clear pip cache if needed
pip cache purge
```

### Q: Visualization not rendering properly
**Common causes:**
1. **Missing JavaScript libraries**: Ensure you're opening the HTML file in a modern browser
2. **CORS issues**: If loading from file://, some browsers block scripts. Use a local server:
```bash
python -m http.server 8000
# Open http://localhost:8000/output/workflow.html
```

### Q: High memory usage during processing
**Solutions:**
1. Enable batch processing:
```python
system.config.batch_size = 100
system.config.enable_streaming = True
```

2. Reduce cache size:
```python
engine.config.cache_max_size = 1000  # Reduce from default 10000
```

3. Use memory profiling:
```bash
pip install memory-profiler
python -m memory_profiler your_script.py
```

### Q: HRM reasoning is slower than expected
**Check these factors:**
1. **Network latency**: If using external LLM APIs
2. **Cache disabled**: Ensure `enable_caching=True`
3. **Inefficient tier thresholds**: Adjust thresholds to use lower tiers more often
4. **Debug mode enabled**: Disable in production

```python
# Optimize for speed
engine = HRMEngine(
    tier_thresholds=[0.2, 0.8],  # Use more tier 1 & 2
    enable_caching=True,
    cache_ttl=7200,  # 2 hours
    debug=False
)
```

### Q: Docker container exits immediately
**Debugging steps:**
```bash
# Check logs
docker-compose logs

# Run in foreground
docker-compose up

# Check container status
docker ps -a

# Enter container for debugging
docker-compose run --rm app /bin/bash
```

---

## Performance & Optimization

### Q: How can I improve token efficiency?
1. **Optimize tier thresholds**: Lower thresholds mean more queries handled by cheaper tiers
2. **Enable caching**: Reuse results for identical queries
3. **Batch processing**: Process multiple queries together
4. **Prompt optimization**: Use concise prompts

```python
# Example optimized configuration
engine = HRMEngine(
    tier_thresholds=[0.25, 0.75],
    enable_caching=True,
    cache_ttl=3600,
    batch_size=50
)
```

### Q: What's the expected performance improvement?
Based on our benchmarks:
- **Token reduction**: 30-70% compared to direct LLM calls
- **Energy savings**: 35-45% less computational overhead
- **Cost reduction**: $0.30-$0.70 per 1000 queries (varies by model)
- **Latency**: 40-60% faster for queries that can use Tier 1/2

### Q: How do I monitor system performance?
```python
from aimulgent.utils import MetricsCollector

metrics = MetricsCollector(system)
stats = metrics.get_stats()

print(f"Total queries: {stats['total_queries']}")
print(f"Tier 1 usage: {stats['tier1_pct']}%")
print(f"Tier 2 usage: {stats['tier2_pct']}%")
print(f"Tier 3 usage: {stats['tier3_pct']}%")
print(f"Tokens saved: {stats['tokens_saved']}")
print(f"Avg response time: {stats['avg_response_time_ms']}ms")
```

---

## Contributing & Development

### Q: How can I contribute to AMULGENT?
We welcome contributions! Here's how to get started:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Run linting: `pre-commit run --all-files`
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Q: How do I run the tests?
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=aimulgent --cov-report=html

# Run specific test file
pytest tests/test_hrm.py

# Run tests in parallel
pytest tests/ -n auto
```

### Q: What code style should I follow?
We use:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking
- **Google-style** docstrings

```bash
# Format code
black aimulgent/

# Lint
ruff check aimulgent/

# Type check
mypy aimulgent/
```

### Q: How do I add a new feature?
1. Check existing issues/PRs for similar work
2. Open an issue to discuss your idea
3. Follow the development setup in README.md
4. Write tests for your feature
5. Update documentation
6. Submit a PR with a clear description

### Q: Where can I get help?
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community support
- **Documentation**: Check docs/api/index.md
- **Examples**: See the examples/ directory

---

## Still Have Questions?

If your question isn't answered here:
1. Check the [full documentation](docs/api/index.md)
2. Search [existing GitHub issues](https://github.com/D0CT4/AMULGENT/issues)
3. Ask in [GitHub Discussions](https://github.com/D0CT4/AMULGENT/discussions)
4. Open a [new issue](https://github.com/D0CT4/AMULGENT/issues/new)

We're here to help! 🚀
