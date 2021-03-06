FROM jupyter/scipy-notebook

# Set the jupyter user's home directory as the working derectory and switch to the root user.
WORKDIR $HOME
USER root

# Install system dependencies
RUN conda install conda-build

# Copy configuration files
COPY config_files/overrides.json /opt/conda/share/jupyter/lab/settings/overrides.json
COPY config_files/pycodestyle ./.config/pycodestyle
COPY config_files/mypy_config ./.config/mypy/config

RUN pip install pyls-mypy
RUN conda install -c conda-forge nbresuse jupyter-lsp-python

RUN jupyter labextension install \
    @jupyterlab/toc \
    @lckr/jupyterlab_variableinspector \
    @aquirdturtle/collapsible_headings \
    @krassowski/jupyterlab-lsp \
    jupyterlab-topbar-extension \
    jupyterlab-system-monitor

# Copy over project
RUN mkdir ./topone
COPY ./external ./topone/external
COPY ./topone ./topone/topone
COPY ./setup.py ./topone/setup.py
COPY ./environment.yml ./topone/environment.yml

# Change ownership to the jupyter user
RUN chown -R $NB_UID:$NB_GID ./topone

# Switch back to the jupyter user
USER $NB_UID

# Install CW as developer.
RUN conda env update --name base -f ./topone/external/cw/environment.yml
RUN conda develop --name base ./topone/external/cw

# Install topone as developer
RUN conda env update --name base -f ./topone/environment.yml
RUN conda develop --name base ./topone
