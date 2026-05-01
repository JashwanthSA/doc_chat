import os
import uuid
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableGenerator
from langchain_litellm import ChatLiteLLM
from ingest import get_chromadb_collection

load_dotenv()

# In-memory session store
_sessions: dict[str, list] = {}

SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions based on the provided context from the user's documents.

Rules:
- Answer ONLY based on the provided context. If the context does not contain the answer, say so clearly.
- Cite the source filename and page number when referencing information.
- Be concise and direct.
- If the user asks something conversational (greetings, etc.), respond naturally without requiring context.

Context from documents:
{context}"""


def get_llm(model: str | None = None, **kwargs):
    """Create a LiteLLM chat model. Model can be any LiteLLM-supported model string."""
    model = model or os.getenv("LITELLM_MODEL", "gpt-4o")
    return ChatLiteLLM(model=model, streaming=True, **kwargs)


def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """Query ChromaDB and return top-k relevant chunks with metadata."""
    collection = get_chromadb_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
    docs = []
    for i in range(len(results['ids'][0])):
        docs.append({
            'id': results['ids'][0][i],
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i] if results.get('distances') else None
        })
    return docs


def format_context(docs: list[dict]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc['metadata']
        source = meta.get('source_filename', 'unknown')
        page = meta.get('page_number', '?')
        parts.append(f"[Source: {source}, Page {page}]\n{doc['document']}")
    return "\n\n---\n\n".join(parts)


def get_session_history(session_id: str) -> list:
    """Get or create message history for a session."""
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


def clear_session(session_id: str):
    """Clear the message history for a session."""
    _sessions.pop(session_id, None)


def new_session_id() -> str:
    return str(uuid.uuid4())


async def query_stream(question: str, session_id: str, model: str | None = None, n_results: int = 5):
    """
    Async generator that yields (event_type, data) tuples.
    event_type: 'sources' | 'token' | 'done' | 'error'
    """
    try:
        # Retrieve relevant documents
        docs = retrieve(question, n_results=n_results)
        source_info = []
        for doc in docs:
            meta = doc['metadata']
            source_info.append({
                'source_filename': meta.get('source_filename', 'unknown'),
                'page_number': meta.get('page_number', '?'),
                'distance': doc.get('distance')
            })
        yield ('sources', source_info)

        # Build context
        context = format_context(docs)

        # Get session history
        history = get_session_history(session_id)

        # Build messages
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
        messages.extend(history)
        messages.append(HumanMessage(content=question))

        # Stream response
        llm = get_llm(model=model)
        full_response = ""
        async for chunk in llm.astream(messages):
            token = chunk.content
            if token:
                full_response += token
                yield ('token', token)

        # Update session history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=full_response))

        # Keep history manageable (last 20 messages = 10 turns)
        if len(history) > 20:
            _sessions[session_id] = history[-20:]

        yield ('done', full_response)

    except Exception as e:
        yield ('error', str(e))
