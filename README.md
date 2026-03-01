# Sandbox-MoGe

## Prerequisites

- Python 3.12 or newer (a virtual environment is recommended).
- Git to clone this repository.
- [uv](https://github.com/astral-sh/uv) (a Python package/dependency manager used in this project).
- A CUDA-capable GPU and the corresponding CUDA runtime (for GPU acceleration).  See the [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for instructions.
- Any system packages required by the dependencies (none specific at the moment).

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/odorumaharaja/Sandbox-MoGe.git
   cd Sandbox-MoGe
   ```
2. Use `uv` to sync the project and its dependencies:
   ```bash
   uv sync
   ```
   This command reads `pyproject.toml` and `uv.lock` to install pinned versions.

3. Run the application to verify startup (using uv to invoke the script):
   ```bash
   uv run python app.py [--share] [--pretrained MODEL] [--version v2] [--fp16]
   ```
   This will launch the web demo and should open a local browser window.

## Notes

- The setup and test steps were carried out following the guidelines on the [MoGe-2 project page](https://github.com/microsoft/MoGe).
  - MoGe code is released under the MIT license, except for DINOv2 code in `moge/model/dinov2` which is released by Meta AI under the Apache 2.0 license. See [LICENSE](https://github.com/microsoft/MoGe/blob/main/LICENSE) for more details.
- The repository is a sandbox implementation of the MoGe depth estimation application with an inference module and web UI.
