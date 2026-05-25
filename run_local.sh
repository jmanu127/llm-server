#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================================="
echo "        Gemma 4 LLM Server: Local Execution Setup (with MPS)"
echo "======================================================================="

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Setup HF_HOME to share the same cache folder as the Docker setup
export HF_HOME="$SCRIPT_DIR/hf_cache"
mkdir -p "$HF_HOME"
echo "Shared Hugging Face cache set to: $HF_HOME"

# 2. Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed. Please install Python 3.10+ and try again."
    exit 1
fi

# 3. Create python virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv'..."
    python3 -m venv .venv
else
    echo "Virtual environment '.venv' already exists."
fi

# 4. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 5. Install / Upgrade dependencies
echo "Installing/updating dependencies (this may take a minute)..."
pip install --upgrade pip
pip install -r requirements.txt

# 6. Check for .env file configuration
if [ -f ".env" ]; then
    # Simple check if user forgot to add token
    if grep -q "HF_TOKEN=ADD_TOKEN_HERE" .env; then
        echo "======================================================================="
        echo "WARNING: Your '.env' file still has the placeholder token."
        echo "Gemma 4 requires Hugging Face authentication."
        echo "To download the model, edit '.env' and set your real HF_TOKEN first."
        echo "======================================================================="
        echo ""
    fi
else
    echo "WARNING: '.env' file not found. Creating a default '.env'..."
    cat > .env << EOL
HF_TOKEN=ADD_TOKEN_HERE
MODEL_ID=google/gemma-4-E4B-it
EOL
fi

# 7. Start the server
echo "Starting FastAPI server..."
echo "Press Ctrl+C to stop."
echo "-----------------------------------------------------------------------"
python server.py
