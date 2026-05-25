# Gemma 4 Streaming LLM Server

A high-performance, custom LLM server built from scratch with **FastAPI**, **PyTorch**, and **Hugging Face Transformers**. 

This server is designed to host **Google DeepMind's Gemma 4 E4B (4.5B Effective Parameters)** model (`google/gemma-4-E4B-it`), featuring dynamic device acceleration, robust token-by-token streaming, and an OpenAI-compatible interface.

---

## ✨ Features

- **Dual-Mode Execution**:
  - **Local Mode (with MPS)**: Run directly on Apple Silicon macOS leveraging the GPU via Metal Performance Shaders (MPS) for high-speed generation.
  - **Docker Mode**: Packaged as a clean, portable container running on CPU (optimized using PyTorch CPU-only wheels).
- **Asynchronous Token Streaming**: Utilizes a threaded `TextIteratorStreamer` generator to push tokens in real-time without locking the FastAPI event loop.
- **OpenAI-Compatible Endpoint**: Exposes a `/v1/chat/completions` API that implements the Server-Sent Events (SSE) spec, serving as a drop-in replacement for standard OpenAI SDKs and frontends.
- **Persistent Shared Model Cache**: Configured to mount host cache volume to the Docker container, guaranteeing that you download the ~9GB model weights only once.

---

## 🏗️ Architecture

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
                       └─────┬────────────┬─────┘
                             │            │
                    (macOS)  │            │  (Docker)
                             ▼            ▼
                       ┌─────────┐    ┌─────────┐
                       │   MPS   │    │   CPU   │
                       │ (GPU)   │    │  (Only) │
                       └────┬────┘    └────┬────┘
                            │              │
                            └──────┬───────┘
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

## 🚀 Setup & Execution

### 1. Configure Hugging Face Access

Gemma 4 is a gated model. Before starting the server:
1. Log in to [Hugging Face](https://huggingface.co) and accept the license terms on the [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) model repository.
2. Generate a User Access Token (read permission) in your account settings under **Settings > Access Tokens**.
3. Open the `.env` file in the root of this project and replace the placeholder with your token:
   ```env
   HF_TOKEN=your_real_token_here
   ```

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
   * Launch the FastAPI server using your Mac's GPU.
3. Observe the server startup logs to verify the correct device assignment:
   ```
   HARDWARE DETECTION: Host using 'MPS' with precision 'torch.float16'
   ```

---

### Option B: Run in Docker (Containerized CPU Mode)

To run the application inside a fully isolated Docker container:

1. Launch the server in the background:
   ```bash
   docker compose up -d
   ```
2. Monitor startup and weights downloading progress:
   ```bash
   docker compose logs -f
   ```
   *Note: On the first boot, the container will securely download and download the ~9GB model weights, caching them locally inside your host `./hf_cache` folder. Subsequent startups are near-instantaneous.*
3. Verify the startup logs to ensure CPU operation:
   ```
   HARDWARE DETECTION: Host using 'CPU' with precision 'torch.bfloat16' (or float32)
   ```

---

## 🧪 Verification and Testing

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
  -d '{"prompt": "Tell me a joke about programming.", "max_tokens": 150}'
```

---

## 📁 File Structure

```
├── Dockerfile            # Optimized slim container configuration
├── README.md             # Project documentation (this file)
├── docker-compose.yml    # Docker services orchestration with volume caches
├── download_model.py     # Utility script to pre-fetch model weights securely
├── run_local.sh          # Local macOS environment setup & executor
├── requirements.txt      # Python dependencies pinned versions
├── server.py             # Main FastAPI server and generation engine
└── test_client.py        # Streaming client API test utility
```