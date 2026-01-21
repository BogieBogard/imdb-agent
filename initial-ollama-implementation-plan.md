# Local Model Implementation Plan
(FYI this is possibly outdated)
## Objective
Replace OpenAI dependencies (`gpt-4o`, `OpenAIEmbeddings`) with local Ollama models (`gemma3:12b`, `qwen3-embedding:8b`).

## Status: :warning: In Progress
- Code changes applied.
- Waiting for `ollama pull` to complete.
- Verification pending.

## Changes Applied

### [agent.py](file:///Users/matthewbogard/Documents/GitHub/imdb-agent/agent.py)
- **LLM**: Switched to `ChatOllama(model="gemma3:12b")`.
- **Embeddings**: Switched to `OllamaEmbeddings(model="qwen3-embedding:8b")`.
- **Vector Store**: Pointed to new index directory `faiss_index_qwen`.
- **Imports**: Updated to `langchain_community`.

## Verification Steps
1. **Model Availability**: Ensure `qwen3-embedding:8b` is installed (Running now).
2. **Integration Test**: 
   - Run `python verify_ollama.py`.
   - Checks: Agent init, Pandas Agent query, Vector Search (Embedding generation).
3. **Manual Check**: User to confirm app responsiveness.

## Notes
- If [verify_ollama.py](cci:7://file:///Users/matthewbogard/Documents/GitHub/imdb-agent/verify_ollama.py:0:0-0:0) fails with "404 not found", verify that the downloaded model tag matches `qwen3-embedding:8b`.