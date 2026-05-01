import asyncio
import sys
import os
import pyfiglet
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML

from ingest import ingest_path, show_collection_info, get_chromadb_collection, COLLECTION_NAME, EMBEDDING_MODEL
from generate_chunks import get_converter, process_document_to_chunks
from query import query_stream, new_session_id, clear_session
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

console = Console()
VERSION = "0.1.0"

# Load tokenizer at bootup (used for chunking during ingestion)
tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2"))


def print_banner():
    logo = pyfiglet.figlet_format("doc_chat", font="slant")
    console.print(Text(logo, style="bold cyan"), end="")
    collection = get_chromadb_collection()
    count = collection.count()
    console.print(f"  Your local RAG assistant  [dim]v{VERSION}[/dim]")
    console.print(f"  Collection: [bold]{COLLECTION_NAME}[/bold] ({count} chunks)  |  Embedding: [dim]{EMBEDDING_MODEL}[/dim]")
    console.print()


def print_help():
    help_text = """[bold]Available commands:[/bold]

  [cyan]/ingest <filepath>[/cyan]   Ingest a document (PDF, DOCX, PPTX) end-to-end
                         Options: --output-path <dir>  (default: chunks)
                                  --describe-images yes|no  (default: yes)
  [cyan]/ingest-all <folder>[/cyan]  Ingest all documents in a folder end-to-end
                         Options: --output-path <dir>  (default: chunks)
                                  --describe-images yes|no  (default: yes)
  [cyan]/info[/cyan]                Show collection stats
  [cyan]/clear[/cyan]               Clear chat history
  [cyan]/help[/cyan]                Show this help message
  [cyan]/quit[/cyan]                Exit (or Ctrl+C)

  Anything else is treated as a query to the RAG agent."""
    console.print(Panel(help_text, title="Help", border_style="dim"))


def handle_ingest(args_str: str):
    """Handle /ingest <filepath> [--output-path <dir>] [--describe-images yes|no]"""
    import shlex
    try:
        parts = shlex.split(args_str)
    except ValueError:
        parts = args_str.split()

    if not parts:
        console.print("[red]Usage: /ingest <filepath> [--output-path <dir>] [--describe-images yes|no][/red]")
        return

    input_path = parts[0]
    output_path = "chunks"
    describe_images = "yes"

    i = 1
    while i < len(parts):
        if parts[i] == "--output-path" and i + 1 < len(parts):
            output_path = parts[i + 1]
            i += 2
        elif parts[i] == "--describe-images" and i + 1 < len(parts):
            describe_images = parts[i + 1].lower()
            i += 2
        else:
            i += 1

    if not os.path.exists(input_path):
        console.print(f"[red]File not found: {input_path}[/red]")
        return

    # Step 1: Generate chunks
    console.print(f"[cyan]Step 1/2:[/cyan] Generating chunks from [bold]{input_path}[/bold]...")
    try:
        converter = get_converter(describe_images=(describe_images == 'yes'))
        if describe_images == 'yes':
            from clip import init_clip_model, init_vlm_model
            clip_model, clip_processor, device = init_clip_model()
            vlm_model = init_vlm_model()
            jsonl_path = process_document_to_chunks(
                input_path, output_path, converter, tokenizer,
                clip_model, clip_processor, device, vlm_model=vlm_model, describe_images=True
            )
        else:
            jsonl_path = process_document_to_chunks(
                input_path, output_path, converter, tokenizer,
                None, None, None, describe_images=False
            )
        console.print(f"  ✅ Chunks generated: [bold]{jsonl_path}[/bold]")
    except Exception as e:
        console.print(f"[red]Error generating chunks: {e}[/red]")
        return

    # Step 2: Ingest into ChromaDB
    console.print(f"[cyan]Step 2/2:[/cyan] Ingesting into ChromaDB...")
    try:
        count = ingest_path(jsonl_path)
        console.print(f"  ✅ {count} chunks ingested into collection [bold]{COLLECTION_NAME}[/bold]")
    except Exception as e:
        console.print(f"[red]Error ingesting chunks: {e}[/red]")


def handle_ingest_all(args_str: str):
    """Handle /ingest-all <folder> [--output-path <dir>] [--describe-images yes|no]"""
    import shlex
    from pathlib import Path
    try:
        parts = shlex.split(args_str)
    except ValueError:
        parts = args_str.split()

    if not parts:
        console.print("[red]Usage: /ingest-all <folder> [--output-path <dir>] [--describe-images yes|no][/red]")
        return

    folder_path = parts[0]
    output_path = "chunks"
    describe_images = "yes"

    i = 1
    while i < len(parts):
        if parts[i] == "--output-path" and i + 1 < len(parts):
            output_path = parts[i + 1]
            i += 2
        elif parts[i] == "--describe-images" and i + 1 < len(parts):
            describe_images = parts[i + 1].lower()
            i += 2
        else:
            i += 1

    if not os.path.isdir(folder_path):
        console.print(f"[red]Not a valid folder: {folder_path}[/red]")
        return

    supported_extensions = {'.pdf', '.docx', '.pptx'}
    files = sorted(
        p for p in Path(folder_path).iterdir()
        if p.is_file() and p.suffix.lower() in supported_extensions
    )

    if not files:
        console.print(f"[yellow]No supported documents (PDF, DOCX, PPTX) found in {folder_path}[/yellow]")
        return

    console.print(f"Found [bold]{len(files)}[/bold] document(s) in [bold]{folder_path}[/bold]")
    console.print()

    # Initialize models once for all documents
    converter = get_converter(describe_images=(describe_images == 'yes'))
    clip_model = clip_processor = device = vlm_model = None
    if describe_images == 'yes':
        from clip import init_clip_model, init_vlm_model
        clip_model, clip_processor, device = init_clip_model()
        vlm_model = init_vlm_model()

    total_ingested = 0
    for idx, filepath in enumerate(files, 1):
        console.print(f"[bold]\n[{idx}/{len(files)}][/bold] Processing [cyan]{filepath.name}[/cyan]...")
        try:
            if describe_images == 'yes':
                jsonl_path = process_document_to_chunks(
                    str(filepath), output_path, converter, tokenizer,
                    clip_model, clip_processor, device, vlm_model=vlm_model, describe_images=True
                )
            else:
                jsonl_path = process_document_to_chunks(
                    str(filepath), output_path, converter, tokenizer,
                    None, None, None, describe_images=False
                )
            count = ingest_path(jsonl_path)
            total_ingested += count
            console.print(f"  ✅ {count} chunks ingested")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    console.print(f"\n✅ Done! Total [bold]{total_ingested}[/bold] chunks ingested from {len(files)} document(s).")


async def handle_query(question: str, session_id: str):
    """Stream a RAG query response to the console."""
    full_response = ""
    sources = []

    # Show retrieval spinner
    with console.status("[bold cyan]Searching documents...", spinner="dots"):
        # We need to get the first event (sources) before streaming
        stream = query_stream(question, session_id)
        first_event = await stream.__anext__()
        if first_event[0] == 'sources':
            sources = first_event[1]
        elif first_event[0] == 'error':
            console.print(f"\n[red]Error: {first_event[1]}[/red]")
            return

    # Stream tokens
    console.print()
    try:
        async for event_type, data in stream:
            if event_type == 'token':
                console.print(data, end="", highlight=False)
                full_response += data
            elif event_type == 'error':
                console.print(f"\n[red]Error: {data}[/red]")
                return
            elif event_type == 'done':
                pass
    except Exception as e:
        console.print(f"\n[red]Error during streaming: {e}[/red]")
        return

    console.print()  # newline after streamed response

    # Show sources
    if sources:
        console.print()
        seen = set()
        source_parts = []
        for s in sources:
            key = (s['source_filename'], s['page_number'])
            if key not in seen:
                seen.add(key)
                source_parts.append(f"  • {s['source_filename']} (p. {s['page_number']})")
        if source_parts:
            console.print("[dim]Sources:[/dim]")
            for part in source_parts:
                console.print(f"[dim]{part}[/dim]")
    console.print()


async def main_loop():
    print_banner()
    session_id = new_session_id()
    prompt_session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: prompt_session.prompt(
                    HTML('<style fg="ansibrightcyan" bg="" bold="true">doc_chat</style> <style fg="ansibrightcyan">❯</style> ')
                )
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash commands
        if user_input.startswith('/'):
            cmd_parts = user_input.split(None, 1)
            cmd = cmd_parts[0].lower()
            cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ('/quit', '/exit'):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == '/help':
                print_help()
            elif cmd == '/info':
                show_collection_info()
            elif cmd == '/clear':
                clear_session(session_id)
                session_id = new_session_id()
                console.print("[dim]Chat history cleared.[/dim]")
            elif cmd == '/ingest':
                handle_ingest(cmd_args)
            elif cmd == '/ingest-all':
                handle_ingest_all(cmd_args)
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]. Type [cyan]/help[/cyan] for available commands.")
        else:
            # Treat as a query
            await handle_query(user_input, session_id)


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")