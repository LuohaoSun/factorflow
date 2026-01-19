# 如果需要强制构建 AMD64 镜像 (例如部署到 x86 服务器):
# FROM --platform=linux/amd64 codercom/code-server:latest
FROM codercom/code-server:latest
USER coder

# 1. 安装Python和uv
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    python3 python3-pip curl build-essential python3-dev && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/coder/.local/bin:${PATH}"

# 2. 安装vscode扩展
ENV EXTENSIONS_GALLERY="{\"serviceUrl\": \"https://marketplace.visualstudio.com/_apis/public/gallery\", \"itemUrl\": \"https://marketplace.visualstudio.com/items\"}"
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension ms-toolsai.jupyter && \
    code-server --install-extension ms-toolsai.datawrangler && \
    code-server --install-extension mhutchie.git-graph && \
    code-server --install-extension charliermarsh.ruff && \
    code-server --install-extension mechatroner.rainbow-csv && \
    code-server --install-extension esbenp.prettier-vscode && \
    code-server --install-extension yzhang.markdown-all-in-one && \
    code-server --install-extension DotJoshJohnson.xml && \
    code-server --install-extension redhat.vscode-yaml && \
    rm -rf ~/.local/share/code-server/CachedExtensionVSIXs/*

# [可选]2.1 安装Jedi语言服务器(pylance在code-server上可能不提供语义高亮和代码跳转)
RUN code-server --install-extension jedi.jedi && \
    mkdir -p /home/coder/.local/share/code-server/User && \
    echo '{ "python.languageServer": "Jedi" }' > /home/coder/.local/share/code-server/User/settings.json

# 3. 创建虚拟环境并安装依赖(使用pyproject.toml，不安装项目以利用层缓存)
WORKDIR /home/coder/project
COPY --chown=coder:coder pyproject.toml ./
RUN --mount=type=cache,target=/home/coder/.cache/uv \
    uv sync --no-install-project --all-extras

# 4. 拷贝和安装代码(需配合.dockerignore排除.venv, 否则会覆盖上一层)
COPY --chown=coder:coder . .
RUN --mount=type=cache,target=/home/coder/.cache/uv \
    uv sync --all-extras

# 5. 暴露端口和启动命令(需要--cert并配合firefox, 某些特性必须https才能正常工作)
EXPOSE 8080
CMD ["code-server", "--verbose", "--auth", "none", "--bind-addr", "0.0.0.0:8080", "--cert"]
