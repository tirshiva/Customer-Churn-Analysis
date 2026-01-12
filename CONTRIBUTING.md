# Contributing to Customer Churn Analysis

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes: `make test`
6. Format code: `make format`
7. Commit: `git commit -m "Add feature: description"`
8. Push: `git push origin feature/your-feature-name`
9. Open a Pull Request

## Development Setup

```bash
# Install development dependencies
make install-dev

# Setup pre-commit hooks
make setup-pre-commit

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use Black for formatting (line length: 100)
- Use type hints where possible
- Write docstrings for all functions and classes

### Code Quality

- All code must pass linting: `make lint`
- All tests must pass: `make test`
- Code should be formatted: `make format`
- Pre-commit hooks will check these automatically

### Commit Messages

Use clear, descriptive commit messages:
- `Add feature: implement new model evaluation metric`
- `Fix bug: resolve memory leak in data preprocessing`
- `Update docs: improve API documentation`
- `Refactor: simplify model training pipeline`

## Testing

- Write tests for new features
- Ensure existing tests still pass
- Aim for good test coverage
- Use pytest fixtures for common setup

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update relevant documentation files in `docs/`
- Keep code comments clear and concise

## Pull Request Process

1. Update README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Request review from maintainers

## Questions?

Feel free to open an issue for questions or discussions!
