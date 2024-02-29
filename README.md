# KinaseDocker²
*A PyMOL plugin with accompanying Docker image for kinase inhibitor binding and affinity prediction*

KinaseDocker² is a computational tool that implements fully automated docking and scoring. The tool allows for docking in either AutoDock VinaGPU or DiffDock and subsequent scoring by a Deep Neural Network that has been trained on kinase-inhibitor docking poses. This tool can both be installed as a PyMOL plugin and used through the CLI.

In the backend, it uses a Docker image to run the GPU-accelerated VinaGPU, DiffDock and PyTorch DNN implementation. The instructions below assume you have a working GPU-enable Docker installation on your system. Refer to guides such as https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html for detailed Docker installation instructions.

## How to install (PyMOL plugin):
### 1. Setup conda environment:
- Download the KinaseDocker.yml file
- Create environment
```bash
conda env create -f KinaseDocker.yml
```
- Activate environment
```bash
conda activate KinaseDocker
```

Or, if you really want to install it in an existing environment:
- Get PyMOL
```bash
conda install -c conda-forge PyMOL-open-source
```
- Install dependencies
```bash
conda install anaconda::h5py
pip install meeko==0.3.3 scipy docker pandas rdkit
```

### 2. Setup docker image:
- Download the docker image: kinasedocker_v1_0.tar # Will be replaced for `docker pull`
- Load the docker image
```bash
docker load -i kinasedocker_v1_0.tar
```

### 3. Install plugin into PyMOL
- Download kinasedocker_plugin.zip
- Run PyMOL
```bash
pymol
```
- Go to Plugin > Plugin manager > Install New Plugin
- Click Choose file... and select the kinasedocker_plugin.zip
- Press Ok a bunch of times

## How to install for only CLI-use:
### 1. Setup conda environment:
- Create environment
```bash
conda create -n MY_ENV python=3.9 
```
- Activate environment
```bash
conda activate MY_ENV
```
- Install dependecies
```bash
pip install meeko==0.3.3 scipy docker pandas rdkit
```

### 2. Setup docker image:
- Download the docker image: vina_diffdock_dnn.tar
- Load the image
```bash
docker load -i vina_diffdock_dnn.tar
```

### 3. Get the CLI:
- Download the files pipeline.py and kinase_data.csv
- Run the pipeline
```bash
python pipeline.py --help
```
