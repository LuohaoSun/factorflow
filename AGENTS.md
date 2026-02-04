# Project Instructions

## Project Structure

- `src/`: Source code directory.
- `pyproject.toml`: Project configuration and dependencies (managed by uv/hatch).

## Development Guidelines

### Dependency Management

- Use `uv` for dependency management.
- Use `uv run ...` instead of `python ...` to run scripts.
- Use `uv sync --all-extras` to install dependencies.

### Coding Instructions

- Use modern Python practices (Type hints, etc.).
- Format and lint the code using Ruff: `uv run ruff check . --fix` and `uv run ruff format .`.
- Flat is better than nested. 严禁多层嵌套.
- Errors should never pass silently. 除非用户指定,任何防御性代码都不得自动回退, 必须显示抛出错误.
- In the face of ambiguity, refuse the temptation to guess. 如果用户需求不明确, 询问用户直到需求明确为止.
