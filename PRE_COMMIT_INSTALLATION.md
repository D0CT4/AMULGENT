# Pre-commit Hooks Installation and Usage Guide

This document provides comprehensive instructions for installing, configuring, and using pre-commit hooks in the AMULGENT project.

## Table of Contents

- [What are Pre-commit Hooks?](#what-are-pre-commit-hooks)
- [Why Use Pre-commit Hooks?](#why-use-pre-commit-hooks)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [CI/CD Integration](#cicd-integration)

## What are Pre-commit Hooks?

Pre-commit hooks are scripts that run automatically before each commit. They help catch issues early by:
- Checking code style and formatting
- Running linters and static analysis
- Detecting security vulnerabilities
- Running automated tests
- Preventing commits of sensitive information

## Why Use Pre-commit Hooks?

AMULGENT uses pre-commit hooks to maintain high code quality:

1. **Consistency**: Ensures all code follows the same standards
2. **Early Detection**: Catches issues before they reach the repository
3. **Automation**: Reduces manual code review overhead
4. **Security**: Prevents accidental commits of secrets or vulnerabilities
5. **Quality**: Maintains test coverage and documentation standards

## Installation

### Prerequisites

Ensure you have the following installed:

```bash
# Python 3.8 or higher
python --version

# pip (Python package manager)
pip --version

# Git
git --version
```

### Step 1: Install Pre-commit

Install the pre-commit package:

```bash
# Install pre-commit globally
pip install pre-commit

# Or install in a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pre-commit
```

### Step 2: Clone the Repository

If you haven't already:

```bash
git clone https://github.com/D0CT4/AMULGENT.git
cd AMULGENT
```

### Step 3: Install Pre-commit Hooks

Install the git hooks from the `.pre-commit-config.yaml` file:

```bash
pre-commit install
```

You should see:
```
pre-commit installed at .git/hooks/pre-commit
```

### Step 4: Install Additional Tools

Some hooks require additional tools. Install them:

```bash
# Install development dependencies
cd AIMULGENT
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt

# Install additional security tools
pip install safety bandit
```

### Step 5: Verify Installation

Verify the hooks are installed correctly:

```bash
pre-commit --version
pre-commit run --all-files
```

## Configuration

### Configuration File

The `.pre-commit-config.yaml` file in the root directory contains all hook configurations. Key components:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        name: Format code with Black
        args: ['--line-length=88']
```

### Enabled Hooks

AMULGENT uses the following hooks:

#### Code Formatting
- **black**: Automatic Python code formatting
- **isort**: Import statement sorting
- **prettier**: JSON/YAML/Markdown formatting

#### Linting
- **flake8**: Style guide enforcement
- **pylint**: Advanced code analysis
- **pydocstyle**: Docstring style checking
- **mypy**: Static type checking

#### Security
- **bandit**: Security issue detection
- **safety**: Dependency vulnerability checking
- **detect-secrets**: Secret detection

#### Testing
- **pytest**: Automated test execution
- **pytest-cov**: Test coverage checking

#### General Checks
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: YAML syntax validation
- **check-json**: JSON syntax validation
- **detect-private-key**: Prevent committing private keys
- **check-merge-conflict**: Detect merge conflict markers

### Customizing Configuration

To modify hook behavior:

1. Edit `.pre-commit-config.yaml`
2. Update hook arguments:
   ```yaml
   - id: black
     args: ['--line-length=100']  # Change from 88 to 100
   ```
3. Re-install hooks:
   ```bash
   pre-commit install
   ```

## Usage

### Automatic Execution

Hooks run automatically on `git commit`:

```bash
git add .
git commit -m "Your commit message (Fixes #2)"
```

If any hook fails, the commit is blocked and you'll see error messages.

### Manual Execution

#### Run All Hooks

Test all hooks on all files:

```bash
pre-commit run --all-files
```

#### Run Specific Hook

Run a single hook:

```bash
pre-commit run black --all-files
pre-commit run flake8 --all-files
pre-commit run pytest-check --all-files
```

#### Run on Staged Files Only

Run hooks on staged files:

```bash
git add file.py
pre-commit run
```

### Skipping Hooks

**Warning**: Only skip hooks when absolutely necessary!

#### Skip All Hooks

```bash
git commit --no-verify -m "Emergency fix"
```

#### Skip Specific Hooks

Set the `SKIP` environment variable:

```bash
SKIP=flake8,pylint git commit -m "WIP: Skip linters"
```

## Troubleshooting

### Common Issues

#### Issue 1: "pre-commit: command not found"

**Solution**:
```bash
pip install pre-commit
# Or ensure pre-commit is in your PATH
export PATH="$HOME/.local/bin:$PATH"
```

#### Issue 2: Hooks not running

**Solution**:
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Verify installation
ls -la .git/hooks/pre-commit
```

#### Issue 3: "black: command not found"

**Solution**:
```bash
# Install development dependencies
pip install black isort flake8 pylint mypy pytest
```

#### Issue 4: Pytest failures

**Solution**:
```bash
# Run tests manually to see details
cd AIMULGENT
pytest tests/ -v

# Fix failing tests, then commit
```

#### Issue 5: "safety check failed"

**Solution**:
```bash
# Update vulnerable dependencies
safety check
pip install --upgrade <vulnerable-package>

# Or update all dependencies
pip list --outdated
pip install --upgrade <package-name>
```

#### Issue 6: Too slow on large commits

**Solution**:
```bash
# Run hooks only on changed files (default)
git commit -m "Message"

# Or disable slow hooks temporarily
SKIP=pytest-check,pytest-coverage git commit -m "Message"
```

### Getting Help

If you encounter issues:

1. Check hook output for specific errors
2. Run the failing hook manually:
   ```bash
   pre-commit run <hook-id> --all-files --verbose
   ```
3. Check the pre-commit documentation: https://pre-commit.com
4. Open an issue: https://github.com/D0CT4/AMULGENT/issues

## Advanced Usage

### Updating Hooks

Update hook versions automatically:

```bash
# Update to latest versions
pre-commit autoupdate

# Update specific repo
pre-commit autoupdate --repo https://github.com/psf/black
```

### Running Hooks in Different Stages

Hooks can run at different stages:

- **commit**: Run on `git commit` (default)
- **push**: Run on `git push`
- **merge-commit**: Run on merge commits
- **manual**: Only run manually

Example configuration:

```yaml
- repo: local
  hooks:
    - id: pytest-coverage
      name: Check test coverage
      entry: pytest
      language: system
      pass_filenames: false
      stages: [push]  # Only run on push
```

Install push hooks:

```bash
pre-commit install --hook-type pre-push
```

### Creating Custom Hooks

Add custom hooks to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: custom-check
      name: Custom validation
      entry: python scripts/custom_check.py
      language: system
      pass_filenames: false
      always_run: true
```

### Performance Optimization

#### Use Parallel Execution

Pre-commit runs hooks in parallel by default. Adjust parallelism:

```bash
# Run with specific number of parallel processes
PRE_COMMIT_CONCURRENCY=4 pre-commit run --all-files
```

#### Cache Hook Environments

Pre-commit caches hook environments. Clear cache if needed:

```bash
pre-commit clean
pre-commit gc
```

#### Skip Slow Hooks in Development

Create a Git alias for quick commits during development:

```bash
git config alias.qc '!git commit --no-verify'

# Use: git qc -m "Quick WIP commit"
```

### Integration with IDEs

#### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.linting.pylintEnabled": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

1. Install plugins: Black, Flake8, Pylint
2. Configure external tools:
   - Settings → Tools → External Tools
   - Add pre-commit as external tool
3. Configure to run on save

## CI/CD Integration

### GitHub Actions

The pre-commit hooks are integrated with GitHub Actions. See `.github/workflows/` for CI configuration.

### Pre-commit.ci

AMULGENT uses pre-commit.ci for automatic hook running on pull requests:

- Hooks run automatically on PRs
- Auto-fixes are committed
- Status checks appear on PRs

Configuration in `.pre-commit-config.yaml`:

```yaml
ci:
  autofix_commit_msg: '[pre-commit.ci] auto fixes'
  autofix_prs: true
  autoupdate_schedule: weekly
  skip:
    - pytest-check
    - pytest-coverage
    - safety-check
```

## Best Practices

1. **Always install hooks** after cloning the repository
2. **Run hooks manually** before committing large changes:
   ```bash
   pre-commit run --all-files
   ```
3. **Keep hooks updated**:
   ```bash
   pre-commit autoupdate
   ```
4. **Don't skip hooks** unless absolutely necessary
5. **Fix issues immediately** rather than committing with `--no-verify`
6. **Add tests** for new features to maintain coverage
7. **Document changes** with proper docstrings
8. **Reference issues** in commit messages (e.g., "Fixes #2")

## Hook Reference

### Quick Command Reference

```bash
# Installation
pre-commit install                    # Install commit hooks
pre-commit install --hook-type pre-push  # Install push hooks

# Running
pre-commit run --all-files           # Run all hooks on all files
pre-commit run <hook-id>             # Run specific hook
pre-commit run --files file.py       # Run on specific files

# Maintenance
pre-commit autoupdate                # Update hook versions
pre-commit clean                     # Clean cached environments
pre-commit uninstall                 # Uninstall hooks

# Debugging
pre-commit run --verbose             # Verbose output
pre-commit run --show-diff-on-failure  # Show diffs on failure
SKIP=hook1,hook2 git commit          # Skip specific hooks
```

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com)
- [Supported Hooks](https://pre-commit.com/hooks.html)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [AMULGENT Contributing Guide](https://github.com/D0CT4/AMULGENT#contributing)
- [AMULGENT Security Policy](https://github.com/D0CT4/AMULGENT/blob/main/SECURITY.md)

## Support

If you need help with pre-commit hooks:

1. Check this guide and the troubleshooting section
2. Review the [pre-commit documentation](https://pre-commit.com)
3. Open an issue: [GitHub Issues](https://github.com/D0CT4/AMULGENT/issues)
4. Reference Issue #2 for pre-commit related discussions

---

**Happy coding with automated quality checks! 🚀**
