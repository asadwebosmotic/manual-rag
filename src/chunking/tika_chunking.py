import os
import re
import tika
from tika import parser as tika_parser

# 🔧 Init Java VM for Tika
tika.TikaClientOnly = True
tika.initVM()

# 📂 Configs
MIN_WORDS = 5
MAX_CHARS = 2500  # Adjusted for typical embedding model token limits (~512 tokens)
TABLE_MARKERS = [
    r"(Particulars\s+.*?((Rs\.|%)|Absolute|Percentage))",  # Detects table headers
    r"^\s*\w+\s*\d{4}-\d{2}\s+\d{4}-\d{2}",  # Financial years in tables
    r"^\s*[a-zA-Z]\)\s+",  # Sub-items like a), b)
]
SECTION_MARKERS = [
    r"^(Illustration \d+|Test your Understanding|Notes to Accounts|Comparative [a-zA-Z\s]+:)",
    r"^(Common Size|Balance Sheet|Profit and Loss)",
]
EXERCISE_MARKERS = [
    r"^(Prepare|From the following|Choose the right answer|State whether)",
]

def parse_file(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    parsed = tika_parser.from_file(filepath)
    content = parsed.get("content", "")
    metadata = parsed.get("metadata", {})
    return content.strip(), metadata

def estimate_pages(text):
    return text.split('\f')

def is_table_content(line):
    """Check if a line is part of a table based on patterns."""
    return any(re.search(marker, line, re.IGNORECASE | re.MULTILINE) for marker in TABLE_MARKERS)

def is_section_start(line):
    """Check if a line starts a new section or exercise."""
    return any(re.search(marker, line, re.IGNORECASE | re.MULTILINE) for marker in SECTION_MARKERS + EXERCISE_MARKERS)

def content_aware_chunk(page_text):
    """Split page text into semantically meaningful chunks."""
    lines = page_text.split('\n')
    chunks = []
    current_chunk = []
    current_type = "text"  # Default type: text, table, section, exercise
    chunk_size = 0

    for line in lines:
        line = line.strip()
        if not line or len(line.split()) < MIN_WORDS:
            continue

        # Determine if line starts a new section or exercise
        if is_section_start(line):
            if current_chunk:
                chunks.append({
                    "content": "\n".join(current_chunk).strip(),
                    "type": current_type
                })
                current_chunk = []
                chunk_size = 0
            current_type = "exercise" if any(re.search(m, line, re.IGNORECASE) for m in EXERCISE_MARKERS) else "section"
            current_chunk.append(line)
            chunk_size += len(line)
            continue

        # Check if line is part of a table
        if is_table_content(line):
            if current_type != "table" and current_chunk:
                chunks.append({
                    "content": "\n".join(current_chunk).strip(),
                    "type": current_type
                })
                current_chunk = []
                chunk_size = 0
            current_type = "table"
        else:
            # If non-table content and current chunk is a table, close table chunk
            if current_type == "table" and current_chunk:
                chunks.append({
                    "content": "\n".join(current_chunk).strip(),
                    "type": current_type
                })
                current_chunk = []
                chunk_size = 0
            current_type = "text"

        # Add line to current chunk
        if chunk_size + len(line) <= MAX_CHARS:
            current_chunk.append(line)
            chunk_size += len(line)
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append({
                    "content": "\n".join(current_chunk).strip(),
                    "type": current_type
                })
            current_chunk = [line]
            chunk_size = len(line)
            current_type = "table" if is_table_content(line) else "text"

    # Append the last chunk if it exists
    if current_chunk:
        chunks.append({
            "content": "\n".join(current_chunk).strip(),
            "type": current_type
        })

    return [chunk for chunk in chunks if len(chunk["content"].split()) >= MIN_WORDS]

def build_chunks(text, filepath):
    all_chunks = []
    pages = estimate_pages(text)
    for page_number, page_text in enumerate(pages, start=1):
        raw_chunks = content_aware_chunk(page_text)
        for chunk in raw_chunks:
            all_chunks.append({
                "page_content": chunk["content"],
                "metadata": {
                    "page_number": page_number,
                    "source": filepath,
                    "content_type": chunk["type"]
                }
            })
    return all_chunks

# 🔮 Main runner
if __name__ == "__main__":
    filename = input("Enter the filename to parse: ").strip()

    try:
        print("\ Perrin file with Tika...")
        parsed_text, tika_meta = parse_file(filename)

        print("\n🧠 Performing content-aware chunking for RAG pipeline...")
        chunks = build_chunks(parsed_text, filepath=filename)

        print(f"\n📦 Total Chunks Created: {len(chunks)}\n")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print("📄 Page:", chunk['metadata']['page_number'])
            print("📂 Source:", chunk['metadata']['source'])
            print("📋 Type:", chunk['metadata']['content_type'])
            print("🔤 Content:\n", chunk['page_content'])

    except Exception as e:
        print(f"\n❌ Error: {e}")