# Master's project Jean Naftalski

This directory provides the principal codes that were developed for my Master's project. 
The codes of the numerical modeling part are based on the [FEniCSx tutorial](https://jorgensd.github.io/dolfinx-tutorial/index.html#) authored by Jørgen S. Dokken.
If needed, the scripts developed as part of my Master's project pre-study are also accessible at this [link](https://github.com/jeannafta/pre-study).

Below are some tips for using [FEniCS](https://fenicsproject.org/) with [Docker](https://www.docker.com/).

## Running FEniCS in Docker
**1.** Running **Python scripts** in `dolfinx/dolfinx` container: sequential and parallel programming
- To create a new container named `PDM` that will have acess to the local folder  `models`, and will have 512 megabytes of shared memory for the container:
  > docker run -ti --name PDM -v $(pwd)/models:/root --shm-size=512m dolfinx/dolfinx:stable
- To launch a terminal inside the container:
  - If the container has just been created, the terminal is launched automatically.
  - If the container has been restarted, the terminal needs to be explicitly initiated inside the container:
   > docker exec -it PDM bash
- To execute a Python script within the container terminal:
  > python3 script.py
- To execute a Python script in parallel within the container terminal using multiple processes (in this case, 2 processes):
  > mpirun -np 2 python3 script.py

**2.** Running **Jupyter Notebooks** in dolfinx/lab container
- To create a new container named `PDM` that will have acess to the local folder  `models`:
  > docker run --name PDM -p 8888:8888 -v $(pwd)/models:/root dolfinx/lab:stable
- Select the kernel `Python 3 (ipykernel)`

**3. Shared commands** for container management
- To stop the container:
  > docker stop PDM
- To restart the container:
  > docker restart PDM
- To remove the container (the container should be stopped first):
  > docker rm PDM
- To list all existing containers, including stopped containers:
  > docker ps -a


## Install other Python libraries inside a container
Installing an additional library in a Docker container is similar to installing a library in a Python virtual environment. 
If using the `dolfinx/dolfinx` container, the terminal should be launched as indicated in Point 1. On the other hand, with the dolfinx/lab container (JupyterLab interface), the terminal can be directly opened from the interface. Once the terminal is accessed, libraries, for example `pyvista`, can be installed using the package manager `pip3`:
> pip3 install pyvista
