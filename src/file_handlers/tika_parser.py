import os
import tika
from tika import parser as tika_parser

# 🔧 Init Java VM for Tika (required only once)
tika.TikaClientOnly = True
tika.initVM()

def parse_file(filepath: str):
    """
    Universal file parser using Apache Tika.
    Returns a list with a single dict containing page_content and metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    parsed = tika_parser.from_file(filepath)
    content = parsed.get("content", "")
    metadata = {"source": filepath}

    return [{"page_content": content.strip(), "metadata": metadata}]

# 🧪 Quick test
if __name__ == "__main__":
    filename = input("Enter the filename to parse: ").strip()

    try:
        docs = parse_file(filename)
        print("\n📝 Parsed Content:\n")
        print(docs)
    except Exception as e:
        print(f"\n❌ Error parsing file: {e}")
        