# SKYSAFE - UAV Safety Assessment System

A RAG-powered (Retrieval-Augmented Generation) intelligent assistant for UAV safety assessment, certification, and regulatory compliance. SKYSAFE provides context-aware answers and initial safety indicators based on SORA (Specific Operations Risk Assessment) regulatory documents.

## Architecture

### System Overview

SKYSAFE is built as a three-tier microservices architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                        Web GUI (Port 3000)                  │
│  Express.js + Static HTML/CSS/JS + Server-Sent Events       │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP Proxy (/api/*)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend (Port 8080)                │
│  REST API + RAG Pipeline + PostgreSQL Chat History          │
└────────────────────┬────────────────────────────────────────┘
                     │ OpenAI-compatible API
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                 LLM Stack (Port 11434)                      │
│  Ollama (gpt-oss:20b) or OpenAI-compatible endpoint         │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Backend API (`server.py`)

**Technology**: Flask + Gunicorn  
**Port**: 8080  
**Database**: PostgreSQL (SQLAlchemy ORM)

**Key Endpoints**:
- `POST /api/v1/chat` - Conversational Q&A with RAG retrieval
- `POST /api/v1/chat/stream` - Streaming chat with Server-Sent Events
- `POST /api/v1/classification` - Initial operation indicators assessment
- `GET /api/v1/history/{user_id}` - Retrieve user chat history
- `DELETE /api/v1/history/{user_id}` - Clear user chat history
- `GET /health` - Health check endpoint

**Features**:
- Lazy-loaded singleton instances for LLM components (memory efficiency)
- Strict request validation with detailed error responses
- Multi-query preprocessing for complex questions
- Per-user conversation persistence

#### 2. RAG Pipeline

**Retrieval Strategy**: Hybrid (FAISS + BM25 with RRF fusion)

```
User Query
    │
    ├─→ Dense Retrieval (FAISS HNSW)
    │   └─→ MMR Diversification
    │
    ├─→ Lexical Retrieval (BM25)
    │
    └─→ RRF Fusion
         │
         ├─→ Field-Aware Scoring
         │   ├─ chunk_summary semantic score
         │   └─ chunk_keywords boost
         │
         └─→ ColBERT/Cross-Encoder Reranking
              │
              └─→ Top-K Results
```

**Chunk Structure** (SORA regulatory documents):
- `chunk_title`: Section title (weighted 2x in retrieval)
- `chunk_text`: Main content
- `chunk_summary`: Post-retrieval semantic signal
- `chunk_keywords`: Lexical boost terms
- `_retrieval_doc`: Pre-computed title+text for indexing

**Index**: FAISS HNSW (M=32, efConstruction=200, Inner Product metric)  
**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2  
**Reranker**: ColBERT v2.0 or Cross-Encoder MiniLM

#### 3. LLM Integration

**Chatbot Mode** (`LLM/LLM_openAI_Chatbot.py`):
- **Context-only responses**: Strict adherence to provided RAG context
- **Multi-turn conversations**: Chat history included in context window
- **Query preprocessing**: Optional decomposition into sub-queries via `QueryGenerator`
- **Reasoning effort**: Configurable (none/low/medium/high)
- **System rules**: Defined in `LLM/system_rules_chatbot.txt`

**Classification Mode** (`LLM/LLM_openAI_Classification.py`):
- **Single-indicator design**: One indicator per request
- **Minimal context**: Only required operation fields included
- **Indicators supported**:
  - `likely_regulatory_pathway` (Open/Specific PDRA/Specific SORA)
  - `initial_ground_risk_orientation` (very_low/low/medium/high)
  - `initial_air_risk_orientation` (very_low/low/medium/high)
  - `expected_assessment_depth` (simple_declaration/structured_assessment/full_sora)
- **Structured output**: JSON with name, value, explanation
- **Query optimization**: Domain-specific terms from `query_structure.py`

#### 4. Web GUI

**Technology**: Express.js + Vanilla JavaScript  
**Port**: 3000

**Features**:
- Real-time streaming chat with markdown rendering (marked.js + DOMPurify)
- User identity management (cookies + localStorage)
- Per-user chat history load/clear
- Classification form with operation parameters
- Sources sidebar with regulatory references
- Responsive design (mobile/tablet/desktop)

**Client Architecture**:
- `chatbot.js`: Chat interface, SSE streaming, history management
- `classification.js`: Indicators form, batch requests, result cards
- `server.js`: Express proxy, static file serving, route handling

## Setup & Installation

### Prerequisites

- **Python**: 3.10+ (Conda recommended)
- **Node.js**: 14.0.0+
- **Ollama**: Running locally on port 11434 (or OpenAI API key)
- **PostgreSQL**: 16+ (for Docker deployment)

### Local Development

#### 1. Backend Setup

```bash
# Clone repository
cd Skysafe_cluster

# Install Python dependencies
pip install flask openai faiss-cpu sentence-transformers rank-bm25 \
    transformers sqlalchemy psycopg2-binary pyyaml

# Build FAISS index (one-time setup)
# Open chatbot.ipynb or classificationTask.ipynb and run the first cells
# This generates PreProcessing/ProcessedFiles/index/faiss.index and docs.json

# Start Ollama (separate terminal)
ollama serve

# Pull model (if not already available)
ollama pull gpt-oss:20b

# Start Flask backend
python server.py
# Backend runs on http://localhost:8080
```

#### 2. Web GUI Setup

```bash
cd Web-GUI

# Install Node dependencies
npm install

# Start development server
npm start
# GUI runs on http://localhost:3000
```

#### 3. Environment Variables

Create `.env` in project root (optional):

```env
# Backend
DATABASE_URL=sqlite:///chat.db
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Web GUI
PORT=3000
BACKEND_URL=http://localhost:8080
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Services:
# - app: Flask backend (8080)
# - db: PostgreSQL database
# - web-gui: Express GUI (3000)

# Stop services
docker-compose down
```

**Note**: Docker setup uses `host.docker.internal` to connect to Ollama running on host machine.

## Configuration

Edit `config.yml` to customize RAG and LLM behavior:

```yaml
# RAG Configuration
rag_mode: "hybrid"           # dense, bm25, or hybrid
reranker_mode: "colbert"     # colbert, ce, or none
top_k: 50                    # Initial retrieval count
ce_keep_k: 10                # Post-reranking results

# LLM Parameters
temperature: 0.2
top_p: 0.9
max_new_tokens: 56000
reasoning_effort: "low"      # none, low, medium, high

# Streaming
stream: true                 # Enable SSE streaming
```

## API Reference

### POST /api/v1/chat

**Request**:
```json
{
  "user_id": "user_123",
  "user_name": "John Doe",
  "preprocess_query": false,
  "reasoning_effort": "medium",
  "chat_history": [
    {"role": "user", "content": "What is SORA?"},
    {"role": "assistant", "content": "SORA is..."}
  ]
}
```

**Response**:
```json
{
  "user_id": "user_123",
  "answer": "Based on the SORA documentation...",
  "sources": [
    "[1] SORA_2.5.pdf: Introduction, page 3"
  ],
  "reasoning": "..."
}
```

### POST /api/v1/classification

**Request**:
```json
{
  "maximum_takeoff_mass_category": "lt_25kg",
  "vlos_or_bvlos": "VLOS",
  "ground_environment": "sparsely_populated",
  "airspace_type": "uncontrolled",
  "maximum_altitude_category": "gt_50m_le_120m",
  "indicators": [
    "likely_regulatory_pathway",
    "initial_ground_risk_orientation"
  ]
}
```

**Response**:
```json
{
  "operation": { /* echoed input */ },
  "indicators": {
    "likely_regulatory_pathway": {
      "name": "likely_regulatory_pathway",
      "value": "Open Category",
      "explanation": "Based on mass < 25kg and VLOS..."
    }
  },
  "sources": {
    "likely_regulatory_pathway": [
      "[1] AMC1 Article 6: Open category, page 12"
    ]
  }
}
```

## Database Schema

**User Table**:
- `id`: Primary key
- `user_id`: External user identifier (unique)
- `username`: Display name
- `created_at`: Registration timestamp

**Message Table**:
- `id`: Primary key
- `user_id`: Foreign key → User
- `role`: "user" | "assistant" | "system"
- `content`: Message text
- `created_at`: Message timestamp

## Development Workflow

### Adding New Indicators

1. Define indicator in `LLM/LLM_openAI_Classification.py`:
   ```python
   _indicator_specs["new_indicator"] = {
       "required_fields": ["field1", "field2"],
       "prompt": "Requested indicator: new_indicator\n..."
   }
   ```

2. Add query terms in `LLM/query_structure.py`:
   ```python
   BASE_QUERIES["new_indicator"] = ["term1", "term2"]
   ```

3. Update allowed indicators in `server.py`:
   ```python
   _ALLOWED_INDICATORS = (..., "new_indicator")
   ```

### Rebuilding FAISS Index

After updating SORA documents in `Documents/SORA_chunks_cleaned_manual.json`:

```python
from pathlib import Path
from PreProcessing.embeddingToolsFAISSv2 import EmbeddingToolFAISS
import json

# Load chunks
with open("Documents/SORA_chunks_cleaned_manual.json") as f:
    chunks = json.load(f)

# Build index
embedder = EmbeddingToolFAISS(output_dir=Path("PreProcessing/ProcessedFiles"))
embedder.build_index(chunks)
```

### Testing

```bash
# Test backend health
curl http://localhost:8080/health

# Test chat endpoint
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "user_name": "Test",
    "preprocess_query": false,
    "chat_history": [
      {"role": "user", "content": "What is VLOS?"}
    ]
  }'
```

## System Prompts

### Chatbot Rules (`LLM/system_rules_chatbot.txt`)

- **Context-only responses**: No external knowledge or speculation
- **Strict inference**: Only use information explicitly in RAG context
- **Missing info handling**: State clearly what information is unavailable
- **Regulatory compliance**: Follow SORA terminology and structure exactly
- **Documentation support**: Assist with drafting compliance documents

### Classification Rules (`LLM/system_rules_classificationTask.txt`)

- **Single indicator focus**: Return only the requested indicator
- **Minimal context**: Use only required operation fields
- **Structured output**: Always return valid JSON
- **Indicative results**: All outputs are preliminary, not final determinations
- **No extrapolation**: Don't compute unasked indicators

## Troubleshooting

### FAISS Index Not Found
```
FileNotFoundError: FAISS index not found at PreProcessing/ProcessedFiles/index/faiss.index
```
**Solution**: Run notebook cells to build index first (see Setup step 1)

### Ollama Connection Failed
```
Failed to initialize LLM: Connection refused
```
**Solution**: Start Ollama service (`ollama serve`) before backend

### Docker Container Can't Reach Ollama
```
backend_unavailable: http://host.docker.internal:11434/v1
```
**Solution**: Ensure Ollama is running on host, or update `LLM_BASE_URL` to container network

### Empty Indicator Response
```
No indicators returned. The server response was empty.
```
**Solution**: Check that all required operation fields are provided and valid

## Performance Considerations

- **Index Loading**: FAISS index loads once at first request (~5-10s for 10k chunks)
- **RAG Retrieval**: Hybrid search ~200-500ms for 50 candidates
- **Reranking**: ColBERT adds ~100-300ms for 10 final results
- **LLM Generation**: 20B model generates ~20-30 tokens/sec on GPU
- **Streaming**: SSE reduces perceived latency for long responses

## Security Notes

- **Input Validation**: All API endpoints validate and sanitize inputs
- **XSS Protection**: Client uses DOMPurify for markdown rendering
- **SQL Injection**: SQLAlchemy ORM with parameterized queries
- **API Keys**: Never commit real API keys; use environment variables
- **User Isolation**: Chat history isolated per `user_id`

## Contributors

Andrea Vaiuso, Gabriele Immordino

## References

- SORA Documentation: EASA Easy Access Rules for Unmanned Aircraft Systems
- FAISS: Facebook AI Similarity Search
- ColBERT: Contextualized Late Interaction over BERT
- Ollama: Local LLM runtime
