# My Awesome Project

My awesome project description.

## Features

- **⚡️ Modern Tooling**: `uv` for fast dependency management.
- **🛡️ Code Quality**: `ruff` (lint/format), `pyright` (types), and `pre-commit` hooks.
- **🐳 Docker Optimized**: Multi-stage builds, cache mounts, and cross-platform support.
- **✨ Dev Experience**: Ready-to-use `dev` container with code-server.

## Using this Template

To initialize a new project from this template:

- **GitHub**: Click the **"Use this template"** button.
- **Git Clone**: Clone the repository and remove the `.git` directory (`rm -rf .git`) to start fresh.

Then follow these steps:

**Rename Project Metadata**:

- Edit `pyproject.toml`: Update `[project] name` and `description`.

**Rename Source Package**:

- Rename the directory: `mv src/my_lib src/your_package_name`.
- Update `pyproject.toml`: Change `packages = ["src/my_lib"]` to `packages = ["src/your_package_name"]`.

**Update Documentation**:

- Update the title and description in this `README.md`.
- Update the Docker image tags in the build commands below.

## Development

### Setup

Install dependencies and pre-commit hooks:

```bash
uv sync --all-extras
uv run pre-commit install
```

### Dev Image Build

Build a development docker image using code-server:

```bash
docker build -t my-awesome-project:dev -f dev.dockerfile .
```

### Docker Deployment

Deploy a minimal runtime docker image:

```bash
docker build -t my-awesome-project:runtime -f runtime.dockerfile .
```
