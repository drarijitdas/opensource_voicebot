# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A low-latency voice bot built with Modal (serverless cloud platform) and Pipecat (real-time voice AI framework). Features real-time speech-to-speech interaction with RAG capabilities for Modal documentation.

**Key Technologies:**
- **Modal**: Serverless GPU deployment platform
- **Pipecat v0.0.92**: Real-time voice AI pipeline framework
- **WebRTC**: Low-latency audio streaming (SmallWebRTC)
- **Python 3.12**: Backend runtime
- **React 19 + Vite**: Frontend

**AI Stack:**
- LLM: Qwen3-4B-Instruct-2507 (vLLM or SGLang)
- STT: NVIDIA Parakeet TDT 0.6B v3
- TTS: Kokoro TTS
- VAD: Silero VAD
- Embeddings: all-minilm-l6-v2 (OpenVINO)
- Vector DB: ChromaDB

## Development Commands

### Setup

```bash
# Install Python dependencies (requires uv)
uv sync
source .venv/bin/activate

# Configure Modal authentication
modal setup

# Install and build frontend
cd client
npm i
npm run build
cd ..
```

### Deployment

```bash
# Deploy individual services (from project root)

# LLM Service (choose one)
modal deploy -m server.llm.vllm_server      # Optimized TTFT
modal deploy -m server.llm.sglang_server    # Faster cold starts with GPU snapshots

# STT Service
modal deploy -m server.stt.parakeet_stt

# TTS Service
modal deploy -m server.tts.kokoro_tts

# Main bot + frontend
modal deploy -m app
```

### GPU Snapshot Warmup

Speed up cold starts by warming GPU snapshots. Run after deployment:

```bash
python -m server.stt.parakeet_stt
python -m server.llm.sglang_server
python -m app
```

### Development

```bash
# Frontend development
cd client
npm run dev          # Dev server with hot reload
npm run build        # Production build
npm run lint         # ESLint
cd ..

# Run bot locally (executes on Modal cloud)
modal run app.py
```

### Testing

```bash
# Run streaming JSON parser tests
python -m server.llm.tests.test_streaming_json_parser
```

Note: This project does not have a comprehensive test suite. The main test file validates the streaming JSON parser used for LLM response parsing.

## Architecture

### High-Level Structure

```
app.py                          # Modal app entry point + frontend server
├── ModalVoiceAssistant (class) # Bot container with WebRTC handler
└── serve_frontend (function)   # ASGI app serving React build

server/
├── bot/                        # Pipecat pipeline orchestration
│   ├── moe_and_dal_bot.py     # Main bot logic (run_bot function)
│   ├── services/              # Pipecat service wrappers for Modal
│   └── processors/            # Custom Pipecat processors (RAG, JSON parser)
├── llm/                        # LLM inference servers
├── stt/                        # Speech-to-text service
├── tts/                        # Text-to-speech service
└── common/                     # Shared constants

client/                         # React frontend
└── src/components/app.tsx     # Main voice UI component
```

### Modal Architecture Patterns

#### Service Communication via Tunnels
Services expose WebSocket endpoints using `modal.forward(port)`. The `ModalTunnelManager` base class (`server/bot/services/modal_services.py`) handles:
- Spawning Modal functions in the background
- Creating forwarded tunnels to service ports
- Passing tunnel URLs via ephemeral `modal.Dict`

#### GPU Snapshot Optimization
Services use `enable_memory_snapshot=True` and `experimental_options={"enable_gpu_snapshot": True}` with `@modal.enter(snap=True)` for heavy initialization (model loading). Post-snapshot setup (tunnel creation) uses `@modal.enter(snap=False)`.

#### Container Lifecycle
- `@modal.enter(snap=True)`: Model loading, pre-snapshot warmup
- `@modal.enter(snap=False)`: Tunnel creation, post-snapshot setup
- `@modal.exit()`: Cleanup, resource release

### Pipecat Pipeline Architecture

**Frame-Based Processing**: All data flows through typed frames:
```
AudioFrame → TranscriptionFrame → LLMFrame → TTSFrame → AudioFrame
```

**Main Pipeline** (in `server/bot/moe_and_dal_bot.py:run_bot`):
```python
Pipeline([
    transport.input(),           # WebRTC audio input
    rtvi,                        # RTVI event protocol handler
    stt,                         # Parakeet STT
    modal_rag,                   # RAG processor (ChromaDB vector search)
    context_aggregator.user(),   # Context management
    llm,                         # OpenAI-compatible LLM
    tts,                         # Kokoro TTS
    transport.output(),          # WebRTC audio output
    context_aggregator.assistant()
])
```

### RAG Pipeline Flow

1. User speaks → Parakeet STT transcribes
2. Transcription triggers ChromaDB vector search (`server/bot/processors/modal_rag.py`)
3. Retrieved docs appended to LLM context
4. LLM generates structured JSON:
   ```json
   {
     "spoke_response": "Natural language answer",
     "code_blocks": ["code snippet 1", "..."],
     "links": ["https://modal.com/docs/..."]
   }
   ```
5. Custom streaming JSON parser extracts fields in real-time
6. `spoke_response` → TTS → audio output
7. `code_blocks` and `links` → UI via RTVI messages

### Frontend Architecture

- **Pipecat Voice UI Kit**: Pre-built React components (`@pipecat-ai/voice-ui-kit`)
- **SmallWebRTC Transport**: Low-latency WebRTC (`@pipecat-ai/small-webrtc-transport`)
- **RTVI Protocol**: Handles custom bot messages (code blocks, links)
- **Entry**: `client/src/main.tsx`
- **Main Component**: `client/src/components/app.tsx`

## Configuration

### GPU Regions
Edit `server/common/const.py`:
```python
SERVICE_REGIONS = ['us-east']  # Or ['us-west'], ['us-east-1', ...], or None
```
This affects all Modal services that import from `server`.

### Model Selection
- **LLM**: Change `MODEL_NAME` in `server/llm/vllm_server.py` or `server/llm/sglang_server.py`
- **STT**: Modify `server/stt/parakeet_stt.py` model path
- **TTS**: Adjust `server/tts/kokoro_tts.py` model configuration

### RAG Documents
Update `server/bot/assets/modal_docs_short.md` to change RAG knowledge base.

### Audio Parameters
Edit `server/bot/moe_and_dal_bot.py`:
- Sample rates (16000 Hz default)
- Frame sizes
- VAD thresholds

## Key Implementation Details

### Service Discovery Pattern
Services pass WebSocket URLs via ephemeral `modal.Dict`. Example:
```python
# In service
connection_urls_dict = modal.Dict.ephemeral()
await connection_urls_dict.put.aio("url", f"ws://{tunnel.url}")

# In consumer
url = await connection_urls_dict.get.aio("url")
```

### Streaming JSON Parser
Custom parser in `server/bot/processors/parser.py` handles character-by-character JSON streaming from LLM. Critical for low-latency response streaming while extracting structured fields.

### RTVI Protocol
Standard protocol for voice UI communication. Custom messages sent via:
```python
await rtvi.send_message_frame(
    RTVIMessageFrame(
        label="code_blocks",
        data={"blocks": code_blocks}
    )
)
```

### Turn Detection
Uses `LocalSmartTurnAnalyzerV3` with VAD for detecting when user stops speaking. Critical for natural conversation flow.

### Metrics
Pipeline has metrics enabled for latency tracking. Access via pipeline task when running.

## Important Files

- **app.py**: Main Modal application, WebRTC handler, frontend server
- **server/bot/moe_and_dal_bot.py**: Core bot orchestration and pipeline setup
- **server/bot/services/modal_services.py**: Base class for Modal tunnel management
- **server/bot/processors/modal_rag.py**: RAG processor with ChromaDB
- **server/bot/processors/parser.py**: Streaming JSON parser
- **server/llm/sglang_server.py**: SGLang inference server (recommended)
- **server/llm/vllm_server.py**: vLLM inference server (alternative)
- **server/stt/parakeet_stt.py**: Parakeet STT service
- **server/tts/kokoro_tts.py**: Kokoro TTS service
- **client/src/components/app.tsx**: React voice UI component

## Debugging

- Each service has `if __name__ == "__main__"` block for standalone testing
- Check Modal logs: `modal app logs modal-voice-assistant`
- Frontend dev server proxies `/offer` to deployed Modal app
- Enable verbose logging in Pipecat pipeline for frame-level debugging
