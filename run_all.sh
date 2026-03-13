#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Shutting down MoGe services..."
    # Kill background jobs
    kill $(jobs -p) 2>/dev/null
    # Proactively kill port usage to be sure
    fuser -k 8000/tcp 7860/tcp 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Proactively clear ports at the start
echo "Cleaning up existing processes on ports 8000 and 7860..."
fuser -k 8000/tcp 7860/tcp 2>/dev/null
sleep 1

echo "Starting MoGe Backend in background..."
export MOGE_VERSION=${MOGE_VERSION:-v2}
export MOGE_FP16=${MOGE_FP16:-true}
uv run python -m backend.main &

# Wait for backend to start (check health)
echo "Waiting for backend to be healthy..."
until curl -s http://localhost:8000/health | grep -q "healthy"; do
  sleep 2
done

echo "Backend is ready. Starting MoGe Frontend..."
export BACKEND_URL=${BACKEND_URL:-http://localhost:8000}
uv run python frontend/app.py
