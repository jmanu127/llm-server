# Gemma 4 Streaming LLM Server

A high-performance, custom LLM server built from scratch with **FastAPI**, **PyTorch**, and **Hugging Face Transformers**. 

This server is designed to host **Google DeepMind's Gemma 4 E4B (4.5B Effective Parameters)** model (`google/gemma-4-E4B-it`), featuring dynamic device acceleration, robust token-by-token streaming, and an OpenAI-compatible interface.

---

## Features

- **Tri-Mode Execution**:
  - **Local Mode (with MPS)**: Run directly on Apple Silicon macOS leveraging the GPU via Metal Performance Shaders (MPS) for high-speed generation.
  - **Docker CPU Mode**: Packaged as a clean, portable, lightweight container running on CPU (optimized using PyTorch CPU-only wheels to save ~2GB).
  - **Docker CUDA GPU Mode**: Accelerated container utilizing NVIDIA GPUs with CUDA 12.1 for blazing-fast inference in cloud/workstation environments.
- **Quantization Support**: Native 8-bit and 4-bit loading toggle via environment variables to run the model on standard consumer hardware.
- **Asynchronous Token Streaming**: Utilizes a threaded `TextIteratorStreamer` generator to push tokens in real-time without locking the FastAPI event loop.
- **OpenAI-Compatible Endpoint**: Exposes a `/v1/chat/completions` API that implements the Server-Sent Events (SSE) spec, serving as a drop-in replacement for standard OpenAI SDKs and frontends.
- **Persistent Shared Model Cache**: Configured to mount host cache volume to the Docker container, guaranteeing that you download the ~9GB model weights only once.

---

## Architecture

```
                       ┌────────────────────────┐
                       │  Client Request        │
                       │  (/generate or chat)   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  FastAPI Web Server    │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   Hardware Detection   │
                       └─────┬──────┬───────┬───┘
                             │      │       │
                    (macOS)  │      │       │ (Nvidia GPU)
                             ▼      │       ▼
                       ┌─────────┐  │  ┌─────────┐
                       │   MPS   │  │  │  CUDA   │
                       │ (GPU)   │  │  │ (GPU)   │
                       └────┬────┘  │  └────┬────┘
                            │       │       │
                            │ (CPU) ▼       │
                            │  ┌─────────┐  │
                            │  │   CPU   │  │
                            │  └────┬────┘  │
                            │       │       │
                            └───────┼───────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │   Gemma 4 E4B Model    │
                        └───────────┬────────────┘
                                    │
                                    ▼
                        ┌────────────────────────┐
                        │  Threaded Generator    │
                        │  (TextIteratorStreamer)│
                        └────────────────────────┘
```

---

## Setup & Execution

### 1. Configure Hugging Face Access

Gemma 4 is a gated model. Before starting the server:
1. Log in to [Hugging Face](https://huggingface.co) and accept the license terms on the [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) model repository.
2. Generate a User Access Token (read permission) in your account settings under **Settings > Access Tokens**.
3. Open the `.env` file in the root of this project and paste your token:
   ```env
   HF_TOKEN=your_real_token_here
   ```

### 2. Configure Quantization (Optional)

You can toggle model precision directly inside the `.env` file to reduce RAM usage:
```env
# Load in 8-bit precision (Requires bitsandbytes - Optimized for CUDA/CPU)
LOAD_IN_8BIT=False

# Load in 4-bit precision (Requires bitsandbytes - Optimized for CUDA/CPU)
LOAD_IN_4BIT=False
```
*(Note: Standard float16 runs best natively on Mac MPS without quantization. If running in CPU/Docker mode, 4-bit or 8-bit is highly recommended to conserve memory).*

---

### Option A: Run Locally (Mac GPU Accelerated - Recommended)

To leverage your Apple Silicon GPU (M1/M2/M3/M4) for fast, hardware-accelerated generation:

1. Open your terminal and run the local execution script:
   ```bash
   ./run_local.sh
   ```
2. The script will automatically:
   * Create a Python virtual environment (`.venv`) if one does not exist.
   * Install all requirements, including PyTorch with native MPS support.
   * Set up your Hugging Face cache in a local directory (`./hf_cache`) to share storage with Docker.
   * Launch the FastAPI server.
3. Observe the server startup logs to verify the correct device assignment:
   ```
   HARDWARE DETECTION: Host using 'MPS' with precision 'torch.float16'
   ```

---

### Option B: Run in Docker (CPU Mode)

To run the application inside a fully isolated Docker container using your CPU:

1. Launch the CPU server container:
   ```bash
   docker compose up llm-server-cpu -d
   ```
2. Monitor startup and weights downloading progress:
   ```bash
   docker compose logs -f llm-server-cpu
   ```
   *Note: The container will securely download the ~9GB model weights on first boot and cache them locally inside your host `./hf_cache` folder. Subsequent startups are near-instantaneous.*

---

### Option C: Run in Docker (NVIDIA GPU / CUDA Mode)

To run the application inside a container utilizing your host's NVIDIA GPU for fast, hardware-accelerated inference:

#### Prerequisites:
- Ensure the **NVIDIA Container Toolkit** is installed on your host OS ([Setup Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

#### Run:
1. Launch the CUDA GPU server container:
   ```bash
   docker compose up llm-server-cuda -d
   ```
2. Monitor startup progress:
   ```bash
   docker compose logs -f llm-server-cuda
   ```
3. Verify the startup logs to ensure CUDA acceleration:
   ```
   HARDWARE DETECTION: Host using 'CUDA' with precision 'torch.float16'
   ```

---

## Verification and Testing

When the server starts successfully, it will expose the API on `http://localhost:8000`.

### 1. Check Server Health
Verify that the model is loaded and check what hardware device is in use:
```bash
curl http://localhost:8000/health
```

### 2. Test Stream via Python Client
We have provided an automated test client that connects to the server and streams responses into your terminal.

1. If running locally, activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Run the test client:
   ```bash
   python3 test_client.py
   ```
   This script will test two separate streaming endpoints:
   - **`/generate`**: Direct prompt-to-text raw streaming.
   - **`/v1/chat/completions`**: OpenAI-compatible chat message delta streaming.

### 3. Test Raw API manually (cURL)
To test streaming manually via a `curl` call:

```bash
curl -N http://localhost:8000/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain gravity in one simple sentence.", "max_tokens": 150}'
```

---

## File Structure

```
├── Dockerfile            # Optimized slim CPU-only container configuration
├── Dockerfile.cuda       # Accelerated CUDA 12.1 GPU container configuration
├── README.md             # Project documentation (this file)
├── docker-compose.yml    # Docker services orchestration (CPU and CUDA services)
├── download_model.py     # Utility script to pre-fetch model weights securely
├── run_local.sh          # Local macOS environment setup & executor
├── requirements.txt      # Python dependencies pinned versions
├── server.py             # Main FastAPI server and generation engine
└── test_client.py        # Streaming client API test utility
```