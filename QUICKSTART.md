# AMULGENT Quickstart Guide

Get up and running with AMULGENT in **under 5 minutes**! ⚡

---

## 🎯 What You'll Learn

By the end of this guide, you'll:
- ✅ Have AMULGENT installed and running
- ✅ Run your first HRM-powered query
- ✅ Visualize a workflow
- ✅ Understand basic configuration options

---

## ⚙️ Prerequisites

Make sure you have:
- **Python 3.8+** installed
- **pip** package manager
- **Git** (for cloning the repository)
- (Optional) **Docker** for containerized deployment

Check your Python version:
```bash
python --version  # Should show 3.8 or higher
```

---

## 📥 Step 1: Installation (60 seconds)

### Option A: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/D0CT4/AMULGENT.git

# Navigate to directory
cd AMULGENT

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import aimulgent; print('✓ Installation successful!')"
```

### Option B: Docker Install

```bash
# Clone and navigate
git clone https://github.com/D0CT4/AMULGENT.git
cd AMULGENT

# Build and run
docker-compose up -d

# Check status
docker-compose ps
```

---

## 🚀 Step 2: Run Your First Demo (30 seconds)

### HRM Reasoning Demo

```bash
python -m aimulgent.demo_hrm
```

**Expected Output:**
```
🚀 AMULGENT HRM Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "What is 2+2?"

✓ Tier 1 (Heuristic): Answered directly - 8ms
Result: 4
Tokens used: 0 (100% savings vs. LLM)
Energy saved: 0.001 kWh

Done! ✨
```

###  Workflow Visualization Demo

```bash
python -m aimulgent.demo_workflow_visualization
```

This will generate an HTML visualization at `output/workflow.html` - open it in your browser!

---

## 💡 Step 3: Your First Custom Script (90 seconds)

Create a file called `my_first_script.py`:

```python
from aimulgent.core import HRMEngine
from aimulgent.agents import AnalyzerAgent

# Initialize HRM engine
engine = HRMEngine(
    tier_thresholds=[0.3, 0.7],  # Adjust reasoning tiers
    enable_caching=True
)

# Create an analyzer agent
agent = AnalyzerAgent(engine)

# Process a query
query = "What are the benefits of renewable energy?"
result = agent.analyze(query)

# Print results
print(f"✓ Query: {query}")
print(f"✓ Tier Used: {result.tier}")
print(f"✓ Tokens: {result.token_count}")
print(f"✓ Response: {result.response[:100]}...")
print(f"✓ Energy Saved: {result.energy_saved_kwh:.4f} kWh")
```

Run it:
```bash
python my_first_script.py
```

---

## 🎨 Step 4: Visualize a Workflow (60 seconds)

Create `workflow_demo.py`:

```python
from aimulgent.core import System
from aimulgent.visualization import WorkflowVisualizer

# Create a system
system = System()

# Add workflow steps
system.add_step("data_input", source="user_query")
system.add_step("hrm_analysis", dependencies=["data_input"])
system.add_step("multi_agent_processing", dependencies=["hrm_analysis"])
system.add_step("result_output", dependencies=["multi_agent_processing"])

# Execute workflow
results = system.execute({
    "query": "Analyze customer sentiment from reviews"
})

# Generate visualization
visualizer = WorkflowVisualizer(system)
visualizer.generate_graph(
    output_path="my_workflow.html",
    interactive=True,
    show_metrics=True
)

print("✓ Visualization saved to: my_workflow.html")
print(f"✓ Processing time: {results.execution_time_ms}ms")
print(f"✓ Total tokens: {results.total_tokens}")
```

Run and view:
```bash
python workflow_demo.py
open my_workflow.html  # Or double-click the file
```

---

## ⚙️ Step 5: Basic Configuration (Optional)

### Configuration via Code

```python
from aimulgent.core import HRMEngine

engine = HRMEngine(
    tier_thresholds=[0.25, 0.75],  # Lower = more tier 1/2 usage
    enable_caching=True,           # Reuse results
    cache_ttl=3600,                # Cache for 1 hour
    debug=False                     # Disable debug logs
)
```

### Configuration via YAML

Create `config/my_config.yaml`:

```yaml
hrm:
  tier_thresholds: [0.3, 0.7]
  enable_caching: true
  cache_ttl: 3600
  
system:
  log_level: INFO
  output_dir: ./output
  
agents:
  analyzer:
    enabled: true
  base:
    enabled: true
  hrm:
    enabled: true
```

Load it in your script:
```python
from aimulgent.core import System

system = System.from_config("config/my_config.yaml")
```

---

## 📊 Understanding HRM Tiers

AMULGENT uses a 3-tier reasoning model to optimize token usage:

| Tier | Method | Speed | Use Case | Token Cost |
|------|---------|-------|----------|------------|
| **Tier 1** | Heuristic | < 10ms | Simple queries, math, facts | Free |
| **Tier 2** | Rule-based | < 100ms | Pattern matching, templates | Free |
| **Tier 3** | Full LLM | Variable | Complex reasoning, creativity | Standard LLM pricing |

**Example Tier Selection:**
- "What is 5 + 3?" → Tier 1 (8ms, 0 tokens)
- "What's the capital of France?" → Tier 2 (45ms, 0 tokens)
- "Write a poem about the ocean" → Tier 3 (2500ms, 250 tokens)

---

## 🎯 Common Use Cases

### 1. Customer Support Chatbot
```python
from aimulgent.integrations import ChatbotInterface

bot = ChatbotInterface(engine)
response = bot.answer("How do I reset my password?")
# Tier 2: Rule-based → instant answer, 0 tokens
```

### 2. Data Analysis Pipeline
```python
from aimulgent.agents import AnalyzerAgent

agent = AnalyzerAgent(engine)
insights = agent.analyze_dataset("sales_data.csv")
# Mixed tiers: Tier 1/2 for simple stats, Tier 3 for trends
```

### 3. Content Moderation
```python
from aimulgent.utils import ContentModerator

moderator = ContentModerator(engine)
is_safe, score = moderator.check("User submitted content here")
# Tier 1/2: Pattern matching for common violations
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'aimulgent'"
**Solution:**
```bash
pip install -e .  # Install in development mode
```

### Issue: Demo not running
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Must be 3.8+
```

### Issue: Docker container not starting
**Solution:**
```bash
# Check Docker is running
docker --version

# Rebuild container
docker-compose down
docker-compose up --build
```

---

## 📚 Next Steps

Now that you're up and running, explore these resources:

1. **Read the Full Documentation**: [README.md](README.md)
2. **Explore Examples**: Check out the `examples/` directory
3. **API Reference**: See [docs/api/index.md](docs/api/index.md)
4. **FAQ**: Common questions answered in [FAQ.md](FAQ.md)
5. **Contributing**: Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🎉 You're Ready!

You've now:
- ✅ Installed AMULGENT
- ✅ Run demos
- ✅ Created custom scripts
- ✅ Visualized workflows
- ✅ Understood HRM tiers

**Start building amazing, energy-efficient AI systems!** 🚀

---

## 💬 Need Help?

- **Questions?** Check [FAQ.md](FAQ.md)
- **Issues?** Open a [GitHub Issue](https://github.com/D0CT4/AMULGENT/issues)
- **Discussion?** Join [GitHub Discussions](https://github.com/D0CT4/AMULGENT/discussions)

Happy coding! 🎯
