# Contributing to VibeRL

Thank you for your interest in contributing to VibeRL! This guide will help you get started with contributing to our reinforcement learning framework.

## Getting Started

### Prerequisites
- Python 3.12 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/VibeRL.git
   cd VibeRL
   ```

2. **Create Development Environment**
   ```bash
   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   uv pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   uv run pytest -n 8

   # Run pre-commit checks
   uv run pre-commit run --all-files
   ```

## Development Workflow

### Branch Management
- **Main branch**: `main` (protected)
- **Feature branches**: `feature/description`
- **Bug fix branches**: `fix/description`
- **Hotfix branches**: `hotfix/description`

### Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/add-new-algorithm
   ```

2. **Make Your Changes**
   - Follow the existing code style and architecture
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run the test suite
   uv run pytest -n 8

   # Run linting
   uv run ruff check viberl/
   uv run ruff format viberl/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new PPO variant with improved stability"
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

Examples:
```
feat: add support for continuous action spaces
fix: resolve memory leak in replay buffer
docs: update installation guide for Windows users
test: add unit tests for DQN agent
```

## Code Style Guidelines

### Python Code
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters (per project's ruff config)
- Use meaningful variable and function names

### Documentation
- Use docstrings for all public functions and classes
- Follow [Google style](https://google.github.io/styleguide/pyguide.html) for docstrings
- Update README.md for significant changes
- Add examples for new features

### Testing
- Write unit tests for all new functionality
- Aim for high test coverage (minimum 80%)
- Use descriptive test names
- Group related tests in test classes

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Run pre-commit hooks
   - Update documentation if needed

2. **Create Pull Request**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link any related issues
   - Request review from maintainers

3. **Review Process**
   - Address review feedback promptly
   - Keep PRs focused and small when possible
   - Ensure CI checks pass

### PR Template
When creating a PR, please include:

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Adding New Features

### New Algorithms
To add a new reinforcement learning algorithm:

1. Create a new file in `viberl/agents/`
2. Inherit from the base `Agent` class
3. Implement required methods (`act`, `learn`)
4. Add tests in `tests/agents/`
5. Update CLI to support the new algorithm
6. Add documentation and examples

### New Environments
To add a new environment:

1. Create a new file in `viberl/envs/`
2. Implement the gymnasium.Env interface
3. Add comprehensive tests
4. Update CLI to support the new environment
5. Add documentation and examples

## Reporting Issues

### Bug Reports
Use the bug report template when creating issues:
- Provide clear reproduction steps
- Include environment details
- Add relevant logs and screenshots
- Check existing issues first

### Feature Requests
Use the feature request template:
- Describe the use case
- Explain the expected behavior
- Provide implementation ideas if possible
- Consider backward compatibility

## Getting Help

- **Documentation**: Check the [README.md](README.md) and inline documentation
- **Issues**: Create an issue for bugs or feature requests

## Recognition

Contributors will be mentioned in release notes for significant contributions.

Thank you for contributing to VibeRL! Your efforts help make reinforcement learning more accessible to everyone.
