# Sandbox-MoGe

This repository is a sandbox implementation of the MoGe 3D reconstruction application, refactored into a modular FastAPI backend and a Gradio frontend.

## Architecture

- **Backend (FastAPI)**: Handles 3D reconstruction inference using MoGe models.
- **Frontend (Gradio)**: Provides a web interface that communicates with the API.
- **Organization**: Refactored into `backend/` and `frontend/` directories for better maintainability.

## Prerequisites (Local Machine)

- Python 3.12 or newer.
- [uv](https://github.com/astral-sh/uv) (Python package/dependency manager).
- CUDA-capable GPU and corresponding runtime.

## Setup & Execution (Local Machine)

1. Clone and navigate:
   ```bash
   git clone https://github.com/odorumaharaja/Sandbox-MoGe.git
   cd Sandbox-MoGe
   ```
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Run both Backend and Frontend:
   ```bash
   ./run_all.sh
   ```
   - API: `http://localhost:8000`
   - UI: `http://localhost:7860`

## Docker Setup

1. Create `.env` and set your Hugging Face token:
   ```bash
   echo "HF_TOKEN=your_token_here" > .env
   ```
2. Build and run with Docker Compose:
   ```bash
   docker compose up --build
   ```
   - Access the UI at `http://localhost:7860`.

## Documentation
- [API Specification](doc/api_spec.md)
- [Inference Details](doc/inference.md)

## License
MoGe code is released under the MIT license, except for DINOv2 code (`moge/model/dinov2`) which is under Apache 2.0.
