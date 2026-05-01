import json
import os
import argparse
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = "doc_chat"
EMBEDDING_MODEL = "all-mpnet-base-v2"

def get_chromadb_collection():
    client = chromadb.PersistentClient(path=".chroma")
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def combine_for_embedding(entry):
    """
    Combine content with optional picture descriptions for richer embeddings.
    This text is stored as the ChromaDB 'document' and auto-embedded.
    """
    parts = []
    content = entry.get('content', '')
    if content and content.strip():
        parts.append(content.strip())
    picture_desc = entry.get('picture_description')
    if picture_desc:
        if isinstance(picture_desc, list):
            for desc in picture_desc:
                if desc:
                    parts.append(f"Image/Diagram: {desc}")
        elif isinstance(picture_desc, str) and picture_desc.strip():
            parts.append(f"Image/Diagram: {picture_desc}")
    return '\n\n'.join(parts) if parts else ''

def ingest_chunks(jsonl_path: str) -> int:
    """
    Read a JSONL chunks file and upsert into the ChromaDB doc_chat collection.
    Returns the count of ingested documents.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    if not entries:
        print(f"⚠️  No entries found in {jsonl_path}")
        return 0

    collection = get_chromadb_collection()
    batch_size = 100
    ingested_count = 0

    print(f"🔄 Ingesting {len(entries)} chunks from {jsonl_path}...")
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        ids = []
        documents = []
        metadatas = []

        for record in batch:
            doc_text = combine_for_embedding(record)
            if not doc_text:
                continue

            pic_desc = record.get('picture_description', [])
            if isinstance(pic_desc, list):
                pic_desc_str = json.dumps(pic_desc, ensure_ascii=False)
            else:
                pic_desc_str = str(pic_desc) if pic_desc else "[]"

            ids.append(record['id'])
            documents.append(doc_text)
            metadatas.append({
                "source_filename": record.get('source_filename', ''),
                "page_number": record.get('page_number', 0),
                "picture_description": pic_desc_str
            })

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            ingested_count += len(ids)
            processed = min(i + batch_size, len(entries))
            print(f"  ✅ Ingested {processed}/{len(entries)} chunks...")

    print(f"✅ Total {ingested_count} chunks ingested from {jsonl_path}")
    return ingested_count

def ingest_path(target_path: str) -> int:
    """
    Ingest a single JSONL file or all JSONL files in a folder.
    Returns total count of ingested documents.
    """
    p = Path(target_path)
    total = 0
    if p.is_file() and p.suffix == '.jsonl':
        total += ingest_chunks(str(p))
    elif p.is_dir():
        jsonl_files = sorted(p.glob('*.jsonl'))
        if not jsonl_files:
            print(f"⚠️  No .jsonl files found in {target_path}")
            return 0
        for jsonl_file in jsonl_files:
            total += ingest_chunks(str(jsonl_file))
    else:
        print(f"❌ Invalid path: {target_path} (must be a .jsonl file or a directory)")
        return 0
    return total

def show_collection_info():
    """Print collection stats."""
    collection = get_chromadb_collection()
    count = collection.count()
    print(f"📊 Collection: {COLLECTION_NAME}")
    print(f"   Documents:  {count}")
    print(f"   Embedding:  {EMBEDDING_MODEL}")
    if count > 0:
        sample = collection.peek(limit=5)
        sources = set()
        for meta in sample.get('metadatas', []):
            if meta and meta.get('source_filename'):
                sources.add(meta['source_filename'])
        if sources:
            print(f"   Sources (sample): {', '.join(sorted(sources))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest JSONL chunks into ChromaDB",
        usage="python ingest.py <path> [options]\n       python ingest.py --info"
    )
    parser.add_argument('path', nargs='?', help='Path to a .jsonl file or folder containing .jsonl files')
    parser.add_argument('--info', action='store_true', help='Show collection info (doc count, sources)')
    args = parser.parse_args()

    if args.info:
        show_collection_info()
    elif args.path:
        ingest_path(args.path)
    else:
        parser.print_help()