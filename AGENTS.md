# Project Instructions

The agent should update this file to reflect the project status.

## Project Structure

- `src/`: Source code directory.
- `notebooks/`: Exploratory code directory.
- `pyproject.toml`: Project configuration and dependencies (managed by uv/hatch).

## Development Guidelines

### Dependency Management

- Use `uv` for dependency management.
- Run `uv sync --all-extras` to install dependencies.

### Code Style

- Use modern Python practices (Type hints, etc.).
- Format and lint the code using Ruff: `uv run ruff check . --fix` and `uv run ruff format .`.
- Pre-commit hooks are configured (`.pre-commit-config.yaml`).
