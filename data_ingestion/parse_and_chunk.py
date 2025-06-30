from src.chunking.plumber_chunking import parse_file, chunk_pdfplumber_parsed_data

def parse_and_chunk(pdf_path):
    pages = parse_file(pdf_path)
    chunks = chunk_pdfplumber_parsed_data(pages)
    return chunks