# Sandbox-MoGe

This repository is a sandbox implementation of the MoGe depth estimation application with an inference module and web UI. It's my sandbox using AI agent (GitHub Copilot and Gemmini Code Assist).

- The setup and test steps were carried out following the guidelines on the [MoGe-2 project page](https://github.com/microsoft/MoGe).
- MoGe code is released under the MIT license, except for DINOv2 code in `moge/model/dinov2` which is released by Meta AI under the Apache 2.0 license. See [LICENSE](https://github.com/microsoft/MoGe/blob/main/LICENSE) for more details.



## Prerequisites (Local Machine)

- Python 3.12 or newer (a virtual environment is recommended).
- Git to clone this repository.
- [uv](https://github.com/astral-sh/uv) (a Python package/dependency manager used in this project).
- A CUDA-capable GPU and the corresponding CUDA runtime (for GPU acceleration).  See the [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for instructions.

## Setup Instructions (Local Machine)

1. Clone the repository:
   ```bash
   git clone https://github.com/odorumaharaja/Sandbox-MoGe.git
   cd Sandbox-MoGe
   ```
2. Use `uv` to sync the project and its dependencies:
   ```bash
   uv sync
   ```
3. Run the application to verify startup (using uv to invoke the script):
   ```bash
   uv run python app.py [--share] [--pretrained MODEL] [--version v2] [--fp16]
   ```

## Prerequisites (Docker)

- Docker Engine or Docker Desktop.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (required for GPU acceleration).

## Setup Instructions (Docker)

1. Clone the repository:
   ```bash
   git clone https://github.com/odorumaharaja/Sandbox-MoGe.git
   cd Sandbox-MoGe
   ```

2. Create a `.env` file and set your Hugging Face token:
   ```bash
   echo "HF_TOKEN=your_hugging_face_token" > .env
   ```

3. Build and run the container:
   ```bash
   docker compose up --build
   ```
   The application will be accessible at `http://localhost:7860`.

## Notes
