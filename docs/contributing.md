# Contributing to VibeRL

Thank you for your interest in contributing to VibeRL! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/0xWelt/VibeRL.git
cd VibeRL

# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest -n 8

# Run linting
ruff check .
ruff format .

# Run type checking
mypy viberl/
```

## Code Style

We use:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pre-commit** for automated checks

### Code Formatting

```bash
# Format code
uv run ruff format src/

# Check formatting
uv run ruff format src/ --check

# Fix linting issues
uv run ruff check src/ --fix
```

### Type Annotations

All new code should include type annotations. We use strict type checking:

```python
# Good
def train_agent(agent: Agent, env: gym.Env, episodes: int) -> dict[str, float]:
    """Train an agent on an environment."""
    ...

# Bad
def train_agent(agent, env, episodes):
    """Train an agent on an environment."""
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest -n 8

# Run specific test file
pytest tests/test_agents/test_reinforce.py

# Run with coverage
pytest -n 8 --cov=viberl --cov-report=html
```

### Writing Tests

We use pytest for testing. Follow these guidelines:

#### Test Structure

```python
# tests/test_agents/test_reinforce.py
import pytest
from viberl.agents.reinforce import REINFORCEAgent

class TestREINFORCESpecific:
    """Test REINFORCE-specific functionality."""

    @pytest.fixture
    def agent(self) -> REINFORCEAgent:
        """Create REINFORCE agent for testing."""
        return REINFORCEAgent(state_size=4, action_size=3, learning_rate=0.01)

    def test_policy_network_output_shape(self, agent: REINFORCEAgent) -> None:
        """Test policy network outputs correct shape."""
        # Test implementation
        ...
```

#### Use Test Fixtures

Use fixtures for setup and teardown:

```python
@pytest.fixture
def mock_env() -> MockEnv:
    """Create mock environment for testing."""
    return MockEnv(state_size=4, action_size=2)

@pytest.fixture
def reinforce_agent() -> REINFORCEAgent:
    """Create REINFORCE agent for testing."""
    return REINFORCEAgent(state_size=4, action_size=2, learning_rate=0.01)
```

#### Parametrized Tests

Use parametrized tests for multiple scenarios:

```python
@pytest.mark.parametrize('episodes', [1, 10, 100])
def test_training_different_lengths(episodes: int) -> None:
    """Test training with different episode counts."""
    ...
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def train_agent(agent: Agent, env: gym.Env, episodes: int, **kwargs) -> dict[str, float]:
    """Train an agent on an environment.

    Args:
        agent: The agent to train.
        env: The environment to train on.
        episodes: Number of episodes to train for.
        **kwargs: Additional training parameters.

    Returns:
        Dictionary containing training metrics.

    Raises:
        ValueError: If episodes is negative.
    """
    ...
```

### API Documentation

We use mkdocs with mkdocstrings for API documentation. To build docs:

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following our style guidelines
- Add tests for new functionality
- Update documentation

### 3. Test Changes

```bash
# Run all checks
pre-commit run --all-files

# Run tests
pytest -n 8

# Check documentation builds
mkdocs build
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Commit Message Guidelines

Use conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add PPO agent implementation
fix: resolve memory leak in DQN replay buffer
docs: update API documentation for agents
```

## Project Structure

```
viberl/
├── agents/          # Agent implementations
├── envs/           # Environment implementations
├── networks/       # Neural network architectures
├── utils/          # Utility functions
└── typing.py       # Type definitions

tests/
├── test_agents/    # Agent tests
├── test_envs/      # Environment tests
├── test_networks/  # Network tests
├── test_utils/     # Utility tests
└── test_typing.py  # Type system tests

docs/
├── api/            # Auto-generated API docs
├── examples.md     # Usage examples
├── index.md        # Main documentation
└── quickstart.md   # Quick start guide
```

## Adding New Algorithms

When adding a new algorithm:

1. **Create Agent Class**
   - Inherit from `Agent` base class
   - Implement required methods: `act`, `learn`, `save`, `load`
   - Add type annotations

2. **Add Tests**
   - Create test file in `tests/test_agents/`
   - Test algorithm-specific functionality
   - Add to interface compliance tests

3. **Update Documentation**
   - Add to examples.md
   - Update quickstart.md if needed

4. **Update CLI**
   - Add to CLI commands in `cli.py`

## Adding New Environments

1. **Implement Environment**
   - Inherit from `gymnasium.Env`
   - Implement required methods
   - Add comprehensive tests

2. **Add Tests**
   - Create test file in `tests/test_envs/`
   - Test environment-specific functionality

## Reporting Issues

When reporting issues:

1. **Search existing issues** first
2. **Provide minimal reproduction** example
3. **Include environment details** (Python version, OS, etc.)
4. **Provide error messages** and stack traces

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and discussions
- **Documentation**: Check the docs first!

## Development Tips

### IDE Setup

#### VS Code
Install these extensions:
- Python
- Ruff
- MyPy
- GitLens

#### PyCharm
Enable:
- Ruff integration
- MyPy plugin
- Git integration

### Debugging

```python
# Use logging instead of print
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug logging to your code
logging.debug(f"Agent state: {agent.state_dict()}")
```

### Performance Profiling

```bash
# Profile training
python -m cProfile -o profile.prof examples/train_reinforce.py

# Analyze profile
python -m pstats profile.prof
```

Thank you for contributing to VibeRL! 🚀
