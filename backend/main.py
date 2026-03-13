import os
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .inference import MoGeInference
from . import routes

app = FastAPI(
    title="MoGe API",
    description="Refactored FastAPI implementation for MoGe 3D reconstruction",
    version="1.1.0"
)

# Constants
ASSETS_DIR = Path(tempfile.gettempdir()) / "moge_assets"
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

# App state for MoGe inference
app.state.moge_inference: Optional[MoGeInference] = None

@app.on_event("startup")
async def startup_event():
    model_version = os.getenv("MOGE_VERSION", "v2")
    model_path = os.getenv("MOGE_MODEL_PATH", None)
    use_fp16 = os.getenv("MOGE_FP16", "true").lower() == "true"
    
    print(f"Initializing MoGe {model_version} (fp16={use_fp16})...")
    app.state.moge_inference = MoGeInference(
        pretrained_model_name_or_path=model_path,
        model_version=model_version,
        use_fp16=use_fp16
    )

# Include API routes
app.include_router(routes.router)

# Serve static files
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
