```
       __                    __          __
  ____/ /___  _____    _____/ /_  ____ _/ /_
 / __  / __ \/ ___/   / ___/ __ \/ __ `/ __/
/ /_/ / /_/ / /__    / /__/ / / / /_/ / /_
\__,_/\____/\___/____\___/_/ /_/\__,_/\__/
               /_____/
```

**Your local RAG assistant** — Ingest documents, ask questions, get answers with source citations. All from the terminal.

---

## Overview

doc_chat is a local-first RAG (Retrieval-Augmented Generation) CLI tool that lets you:

1. **Ingest** PDF, DOCX, and PPTX documents — parsed into semantic chunks with optional image/diagram descriptions
2. **Query** your documents using natural language — with streaming LLM responses and source citations
3. **Interact** via a Claude Code-style terminal UI with slash commands

Everything runs locally (embeddings, vector DB) except the LLM calls which go through LiteLLM to your chosen provider.

---

## Key Technologies

| Component | Package | Purpose |
|---|---|---|
| Document Parsing | [Docling](https://github.com/DS4SD/docling) | Parse PDF, DOCX, PPTX into structured chunks with headings, tables, and images |
| Vector Database | [ChromaDB](https://www.trychroma.com/) | Local persistent vector store (stored at `.chroma/`) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) | `all-mpnet-base-v2` — 768-dim general-purpose embeddings |
| Image Classification | [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) | `openai/clip-vit-base-patch32` — zero-shot classification to detect graphs/diagrams |
| Image Description | [LiteLLM](https://docs.litellm.ai/) | Vision-capable LLM for describing charts, diagrams, and slides |
| RAG Agent | [LangChain](https://python.langchain.com/) + [LiteLLM](https://docs.litellm.ai/) | Query chain with session memory and streaming |
| Terminal UI | [Rich](https://rich.readthedocs.io/) + [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/) + [pyfiglet](https://github.com/pwaller/pyfiglet) | Styled output, spinners, input history, ASCII art banner |

### Models Used

| Model | Role |
|---|---|
| `sentence-transformers/all-mpnet-base-v2` | Embedding model for chunking and vector search |
| `openai/clip-vit-base-patch32` | Image classification (graph/diagram vs decorative) |
| Configurable via `LITELLM_MODEL` | LLM for RAG query responses |
| Configurable via `LITELLM_VLM_MODEL` | Vision LLM for image/slide descriptions during ingestion |

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- An LLM API key (OpenAI, Azure, Anthropic, or local Ollama)

### Installation

```bash
# Clone the repository
git clone https://github.com/JashwanthSA/doc_chat.git

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Create your .env file from the template
cp .env.example .env
```

### Configure `.env`

Edit `.env` with your LLM provider credentials:

```env
# LLM for querying (any LiteLLM-supported model)
LITELLM_MODEL=gpt-4o

# Vision LLM for image descriptions during ingestion
LITELLM_VLM_MODEL=gpt-4o

# Provider key (set whichever applies)
OPENAI_API_KEY=sk-...
```

See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for all supported providers (OpenAI, Azure, Anthropic, Ollama, etc.).

---

## Usage

### Start the TUI

```bash
python main.py
```

This launches the interactive terminal interface:

```
       __                    __          __
  ____/ /___  _____    _____/ /_  ____ _/ /_
 / __  / __ \/ ___/   / ___/ __ \/ __ `/ __/
/ /_/ / /_/ / /__    / /__/ / / / /_/ / /_
\__,_/\____/\___/____\___/_/ /_/\__,_/\__/
               /_____/

  Your local RAG assistant  v0.1.0
  Collection: doc_chat (0 documents)  |  Embedding: all-mpnet-base-v2

doc_chat ❯
```

### Commands

| Command | Description |
|---|---|
| `/ingest <filepath>` | Ingest a single document (PDF, DOCX, PPTX) end-to-end |
| `/ingest-all <folder>` | Ingest all supported documents in a folder |
| `/info` | Show collection stats (document count, sources) |
| `/clear` | Clear chat history (reset session memory) |
| `/help` | Show available commands |
| `/quit` | Exit (or `Ctrl+C`) |

Any other input is treated as a natural language query.

### Ingest Options

Both `/ingest` and `/ingest-all` support:

```
--output-path <dir>        Directory to save generated JSONL chunks (default: chunks)
--describe-images yes|no   Whether to describe images/diagrams using VLM (default: yes)
```

#### Examples

```
doc_chat ❯ /ingest raw/company-policy.pdf
doc_chat ❯ /ingest raw/presentation.pptx --describe-images no
doc_chat ❯ /ingest-all raw/ --output-path chunks --describe-images yes
```

### Querying

Simply type your question:

```
doc_chat ❯ What is the employee recognition policy?

The employee recognition policy states that...

Sources:
  • Employee Recognition and Gift Policy (p. 3)
  • Employee Recognition and Gift Policy (p. 5)
```

Responses stream token-by-token with source citations shown after completion.

### Standalone Ingestion (without TUI)

You can also ingest pre-generated JSONL chunk files directly:

```bash
python ingest.py chunks/my-document.jsonl
python ingest.py chunks/                    # all JSONL files in folder
python ingest.py --info                     # show collection stats
```

---

## Project Structure

```
doc_chat/
├── main.py              # TUI entry point — interactive terminal interface
├── ingest.py            # ChromaDB ingestion from JSONL chunk files
├── query.py             # RAG query agent (LangChain + LiteLLM + streaming)
├── generate_chunks.py   # Document parsing (Docling) → JSONL chunks
├── clip.py              # CLIP image classification + VLM descriptions
├── pyproject.toml       # Dependencies
├── .env.example         # Environment variable template
├── .gitignore
└── README.md
```

---

## How It Works

1. **Ingestion**: Document → Docling parses into structured chunks → CLIP classifies images → VLM describes diagrams → Chunks saved as JSONL → Embedded with `all-mpnet-base-v2` and stored in ChromaDB
2. **Querying**: User question → Embedded and searched against ChromaDB → Top-k chunks retrieved → LLM generates answer with context → Streamed to terminal with source citations