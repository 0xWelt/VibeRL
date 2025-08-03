# VibeRL Development Commands

## Build/Test Commands
```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run single test
uv run pytest tests/test_agents/test_dqn.py::test_dqn_agent_creation

# Format and lint
uv run ruff format
uv run ruff check --fix

# Pre-commit hooks
uv run pre-commit run --all-files
```

## Code Style Guidelines
- **Python**: 3.12+, 100-char lines
- **Imports**: Absolute imports only, no relative
- **Types**: Full type annotations required
- **Quotes**: Single quotes for strings, double for docstrings
- **Naming**: snake_case for functions/vars, PascalCase for classes
- **Errors**: Use loguru for logging, avoid bare except
- **Tests**: Descriptive names, test_*.py files
