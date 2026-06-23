# Docker configuration
This configuration provides an environment with all the required dependencies. It runs on the **host side** and its goal is to compile a deep learning model using `dnnc`, producing a `.elf` binary file.
The container exposes a TTY interface so you can manually compile your model.

Tested configurations:
- host 1:
    - Ubuntu 24.04
    - RTX 3070
    - x86-64 arch, Intel i5-12400F
    - Docker 29.4.3 (Docker Engine)
- host 2:
    - Windows 11
    - No dedicated GPU
    - x86-64 arch, Intel Core i5-11320H
    - Docker 29.4.3 (Docker Desktop)

## Build container
- Clone the main repository if you have not done so already:
    ```bash
    git clone http://cas.polito.it/gitlab/special-project/pynqz2/facial_recognition.git
    cd facial_recognition/host/docker-dnndk
    ```
- Download DNNDK from [this link](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx_dnndk_v3.1_190809.tar.gz) (requires an AMD account). Then copy/move the `xilinx_dnndk_v3.1_190809` folder inside `docker-dnndk`.
- Copy `./scripts/<gpu | cpu>/install.sh` to `xilinx_dnndk_v3.1_190809/xilinx_dnndk_v3.1/host_x86/install.sh`.
- Modify `docker-compose.yml` following the commented sections if you want to enable GPU mode.
- Run the following commands:
    ```bash
    docker compose build --progress=plain
    docker compose up -d
    ```
    > **CAUTION**
    > Make sure you are inside the `docker-dnndk` folder before running these commands.

### Caffe
The DNNDK tools use the Caffe framework (when the model is Caffe-based, as is our ResNet50), so it must be installed in the container. The setup was tested with an RTX 3070, so be cautious if your GPU is different.
- See [here](https://caffe.berkeleyvision.org/installation.html#compilation) for more details about compilation from source.
- Information about `Makefile.config` can be found [here](https://github.com/BVLC/caffe/blob/master/Makefile.config.example). The `caffe-makefiles` folder contains two additional example configurations.



## Verify the build
- Open the container shell using:
    ```bash
    docker exec -it dnndk-container bash
    ```
    or run the `dnndk.sh` script. Then, inside the container, activate the conda environment:
    ```
    conda activate decent
    ```
- Check `cuda`:
    ```bash
    nvcc --version
    ```
    The output should look similar to:
    ```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2018 NVIDIA Corporation
    Built on Sat_Aug_25_21:08:01_CDT_2018
    Cuda compilation tools, release 10.0, V10.0.130
    ```
    > **IMPORTANT**
    > If the output does not match, restart the container.
- Check `caffe`:
    ```
    caffe --version
    ```
    Expected output:
    ```
    caffe version 1.0.0
    ```
- (OPTIONAL) Check `dnnc` and `decent`.


## Run

Run the following command to open the container shell:
```bash
docker exec -it dnndk-container bash
```
### `dnndk.sh`
This script simply calls the command above. If you add it to your `PATH`, it is faster than typing the full command every time.
When the graphics card driver appears to be unavailable, a container restart is needed. The `dnndk.sh` script provides a `--restart|-r` flag for this.

To add the script to your PATH on the host (not inside the container):
```
echo "export PATH=path/to/repository/docker-dnndk/dnndk.sh:\$PATH" >> ${HOME}/.bashrc
```

### `dnndk.ps1`
Windows equivalent of `dnndk.sh`.

## References
- [Docker Engine for Linux](https://docs.docker.com/engine/install/) — Docker without a frontend interface
- [Docker Desktop](https://docs.docker.com/desktop/)
- [Caffe](https://github.com/BVLC/caffe)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — required for running containers with NVIDIA driver access
- [DNNDK 1.3 user guide](https://www.scribd.com/document/669009811/ug1327-dnndk-user-guide)
