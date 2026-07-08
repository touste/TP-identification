# Binder / JupyterHub Dockerfile using pixi for dependency management
FROM ghcr.io/prefix-dev/pixi:latest

# Install required system libs + user management tools
USER root
RUN apt-get update && apt-get install -y libosmesa6 adduser && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create jovyan user for Binder/JupyterHub compatibility
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV NB_UID=${NB_UID}
ENV HOME=/home/${NB_USER}

RUN id -nu ${NB_UID} 2>/dev/null && userdel --force $(id -nu ${NB_UID}) || true
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Copy project files
WORKDIR ${HOME}
COPY --chown=${NB_UID} pixi.toml pixi.lock ${HOME}/

# Install as jovyan so .cache etc. are owned correctly
USER ${NB_USER}
RUN pixi install

# Back to root to copy remaining files
USER root
COPY --chown=${NB_UID} . ${HOME}

# Make pixi-managed binaries available in PATH
ENV PATH="${HOME}/.pixi/envs/default/bin:${PATH}"

USER ${NB_USER}

# Shell entrypoint that execs Binder's CMD (e.g. jupyter notebook ...)
ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]