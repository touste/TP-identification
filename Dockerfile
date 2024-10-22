# From https://github.com/jorgensd/dolfinx-tutorial/blob/main/Dockerfile
FROM ghcr.io/jorgensd/dolfinx-tutorial:release

# Override from base image      
ENV PYVISTA_JUPYTER_BACKEND="html"

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1010
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER} 
ENTRYPOINT []