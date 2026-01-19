# 如果需要强制构建 AMD64 镜像 (例如部署到 x86 服务器):
# FROM --platform=linux/amd64 python:3.11-slim-bookworm AS builder
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 安装构建依赖 (需要git来生成动态版本号)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
# 使用缓存挂载加速构建
RUN --mount=type=cache,target=/root/.cache/uv uv build

# ---------------------------------------------------------

# 如果需要强制构建 AMD64 镜像 (例如部署到 x86 服务器):
# FROM --platform=linux/amd64 python:3.11-slim-bookworm AS runtime
FROM python:3.11-slim-bookworm AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=builder /app/dist /app/dist
# 使用缓存挂载加速安装
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv $VIRTUAL_ENV && \
    uv pip install dist/*.whl && \
    rm -rf dist

# 不使用 root 用户运行 (安全最佳实践)
RUN useradd -m appuser
USER appuser

# 默认启动命令 (请根据具体项目修改，例如 uvicorn main:app)
CMD ["python"]
