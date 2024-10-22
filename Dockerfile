# Adapted from https://github.com/jorgensd/dolfinx-tutorial/blob/main/Dockerfile
FROM ghcr.io/jorgensd/dolfinx-tutorial:release

# Override from base image      
ENV PYVISTA_JUPYTER_BACKEND="html"

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

# 24.04 adds `ubuntu` as uid 1000;
# remove it if it already exists before creating our user
RUN id -nu ${NB_UID} && userdel --force $(id -nu ${NB_UID})
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER} 
ENTRYPOINT []