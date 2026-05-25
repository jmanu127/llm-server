import os
import time
import json
import uuid
import torch
import asyncio
from typing import List, Dict, Any, Optional
from threading import Thread
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "False").lower() in ("true", "1", "yes")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "False").lower() in ("true", "1", "yes")

# Shared references to model and tokenizer
model = None
tokenizer = None
device = "cpu"
dtype = torch.float32

# Configure dynamic hardware detection & model loading
def detect_hardware():
    global device, dtype
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16  # float16 is optimized for Apple Silicon MPS
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # float16/bfloat16 for Nvidia GPUs
    else:
        device = "cpu"
        # Check if CPU supports bfloat16 (highly recommended for modern CPUs running LLMs)
        if torch.cuda.is_bf16_supported() if hasattr(torch, "cuda") else False:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    
    print("=" * 60)
    print(f"HARDWARE DETECTION: Host using '{device.upper()}' with precision '{dtype}'")
    print("=" * 60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic: Load model & tokenizer
    global model, tokenizer
    detect_hardware()
    
    if not HF_TOKEN or HF_TOKEN.strip() == "" or HF_TOKEN == "ADD_TOKEN_HERE":
        print("WARNING: HF_TOKEN is not configured. Server may fail to load Gemma 4.")
        print("Please ensure your token is set in the '.env' file before proceeding.")
        
    print(f"Loading tokenizer & model: {MODEL_ID}...")
    start_time = time.time()
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # Load weights on correct device with correct dtype
        if LOAD_IN_4BIT:
            print("Loading model in 4-bit precision (requires bitsandbytes)...")
            if device == "mps":
                print("WARNING: Native bitsandbytes 4-bit quantization is not fully supported on Apple Silicon (MPS) by official bitsandbytes.")
                print("The server might fail to load or run on CPU. For Mac, standard float16 is recommended.")
            
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if device in ("cuda", "mps") else torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quant_config,
                device_map="auto",
                token=HF_TOKEN,
                trust_remote_code=True
            )
        elif LOAD_IN_8BIT:
            print("Loading model in 8-bit precision (requires bitsandbytes)...")
            if device == "mps":
                print("WARNING: Native bitsandbytes 8-bit quantization is not fully supported on Apple Silicon (MPS) by official bitsandbytes.")
                print("The server might fail to load or run on CPU. For Mac, standard float16 is recommended.")
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                load_in_8bit=True,
                device_map="auto",
                token=HF_TOKEN,
                trust_remote_code=True
            )
        elif device == "mps":
            # For Apple Silicon, we load model on CPU first, then transfer to MPS to avoid OOM
            # or load directly. Transferring to MPS is extremely standard and safe.
            print("Loading weights (this may take a minute)...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                token=HF_TOKEN,
                trust_remote_code=True
            ).to(device)
        elif device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                token=HF_TOKEN,
                trust_remote_code=True
            )
        else:
            print("Loading weights on CPU (this requires substantial RAM)...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                token=HF_TOKEN,
                trust_remote_code=True
            ).to(device)
            
        print(f"SUCCESS: Loaded {MODEL_ID} in {time.time() - start_time:.2f} seconds!")
    except Exception as e:
        print("=" * 60)
        print(f"CRITICAL ERROR loading model: {e}")
        print("Please check your HF_TOKEN, model license agreements, and system memory limit.")
        print("=" * 60)
        # We don't exit the process so the web server can still display health status and descriptive errors
        
    yield
    # Shutdown logic
    print("Shutting down LLM server...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize FastAPI with lifespan handler
app = FastAPI(
    title="Gemma 4 Streaming LLM Server",
    description="A high-performance custom LLM server built from scratch with FastAPI and PyTorch.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for easy web app/client connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas for Requests
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to feed into the LLM")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author (system, user, assistant)")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of messages representing the chat history")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(True, description="Whether to stream response chunks back in real-time")

# Direct Raw Token Streaming Function
def token_generator(prompt: str, max_tokens: int, temperature: float):
    from transformers import TextIteratorStreamer
    
    if model is None or tokenizer is None:
        yield "Error: LLM model is not loaded. Please verify your token and server configuration."
        return

    # Prepare inputs and push to device
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    # Initialize the streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Configure generation parameters
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Start generation in a background thread to allow yielding on the main thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they arrive
    for new_text in streamer:
        yield new_text
        
    thread.join()

@app.get("/health")
async def health():
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Model failed to load. Check server logs."}
        )
    return {"status": "healthy", "device": device, "model": MODEL_ID}

@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Direct prompt-to-text raw streaming endpoint.
    Streams raw generated text chunks.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    async def stream_wrapper():
        # Using an async loop to yield token slices and avoid locking the event loop
        for chunk in token_generator(req.prompt, req.max_tokens, req.temperature):
            yield chunk
            # Yield control back to event loop for high concurrency
            await asyncio.sleep(0.001)
            
    return StreamingResponse(stream_wrapper(), media_type="text/plain")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-Compatible Chat Completion API supporting both streaming and standard formats.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # Convert standard API messages to dictionary format for transformers
    chat_history = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    
    try:
        # Use Gemma 4's built-in chat template
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to apply chat template: {e}")
        
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # Mode 1: Non-streaming Response
    if not req.stream:
        # Run standard non-streaming generation
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        # Offload CPU execution block to background executor to keep API responsive
        def sync_generate():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id
            )
            return outputs
            
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, sync_generate)
        
        # Slice out the generated tokens
        input_len = inputs.input_ids.shape[1]
        completion_tokens = outputs[0][input_len:]
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(completion_tokens),
                "total_tokens": input_len + len(completion_tokens)
            }
        }

    # Mode 2: OpenAI SSE Streaming Response
    async def sse_wrapper():
        try:
            # Yield initial delta opening the assistant role
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            
            # Yield tokens as they are generated
            for chunk in token_generator(prompt, req.max_tokens, req.temperature):
                payload = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": chunk
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.001)
                
            # Yield final packet closing the chunk stream
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            
            # OpenAI standard SSE closure sequence
            yield "data: [DONE]\n\n"
        except Exception as e:
            # Send error payload down the stream
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_wrapper(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 for Docker routing compatibility
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
