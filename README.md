# Master's project Jean Naftalski

This directory provides all the codes that were developed for my Master's project.

Below are some tips for using [FEniCS](https://fenicsproject.org/) with [Docker](https://www.docker.com/).

## Running FEniCS in Docker
1. Running Python Scripts in Dolfinx/Dolfinx Container: Sequential and Parallel Programming
- To create a new container named `PDM` that will have acess to the local folder  `models`, and will have 512 megabytes of shared memory for the Docker container:
  > docker run -ti --name PDM -v $(pwd)/models:/root --shm-size=512m dolfinx/dolfinx:stable
- To launch a terminal inside the container:
-   If the container has just been created, the terminal is launched automatically.
-   If the container has been restarted, the terminal needs to be explicitly initiated inside the container:
   > docker exec -it PDM_python bash
- To execute a Python script:
  > python3 script.py
- To execute a Python script in parallel using multiple processes (in this case, 2 processes):
  > mpirun -np 2 python3 script.py
2. Running Jupyter Notebooks in Dolfinx/lab Container
- To create a new container named `PDM` that will have acess to the local folder  `models`:
  > docker run --name PDM -p 8888:8888 -v $(pwd)/models:/root dolfinx/lab:stable

3. General commands
- To stop the container:
  > docker stop PDM
- To restart the container:
  > docker restart PDM

## Operations on containers
- To list all containers (including stopped containers):
  > docker container ls -a
  Or
  > docker ps -a
- To remove a container:
  > docker container rm container_ID


## Install other Python libraries inside a container
Installing an additional library in a docker container is similar to installing a library in a Python virtual environment. For example one can use pip3 to install `pyvista` library:
> pip3 install pyvista
