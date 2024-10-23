# Adapted from https://github.com/jorgensd/dolfinx-tutorial
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

# Required for pyvista
RUN apt-get update && apt-get install -y libosmesa6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install other dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Replace vtk with vtk-omesa
RUN pip uninstall -y vtk
RUN pip install --no-cache-dir --extra-index-url https://wheels.vtk.org vtk-osmesa

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV NB_UID=${NB_UID}
ENV HOME=/home/${NB_USER}

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