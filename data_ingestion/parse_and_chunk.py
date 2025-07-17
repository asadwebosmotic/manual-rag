import os
import re
from typing import List, Dict
import pdfplumber
from hashlib import md5
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_file(filepath: str) -> List[Dict]:
    if os.path.splitext(filepath)[1].lower() != ".pdf":
        raise ValueError("Unsupported file type")

    pages_data = []
    
    # Use pdfplumber directly as it's more reliable for complex layouts and OCR
    with pdfplumber.open(filepath) as pdf:
        logger.info(f"Total pages in PDF: {len(pdf.pages)}")
        
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            
            # 1. Attempt standard text extraction
            text = page.extract_text()

            # 2. If standard extraction fails, use OCR as a fallback
            if not text or len(text.strip()) < 50: # The threshold helps catch partial failures
                logger.warning(f"Standard text extraction yielded little or no text for page {page_num}. Attempting OCR.")
                try:
                    # pdfplumber.to_image().ocr() requires pytesseract and its dependencies
                    text = page.to_image(resolution=300).ocr()
                    logger.info(f"Successfully extracted text with OCR from page {page_num}. Text length: {len(text)}")
                except Exception as ocr_error:
                    logger.error(f"OCR failed for page {page_num}: {ocr_error}")
                    text = "" # Ensure text is empty if OCR fails

            # 3. Extract tables
            tables = page.extract_tables() or []

            # Clean up text for better processing
            cleaned_text = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())

            pages_data.append({
                "text": cleaned_text,
                "tables": tables,
                "metadata": {
                    "page_number": page_num,
                    "source": os.path.basename(filepath)
                }
            })
            
    return pages_data

def create_table_chunk(table: Dict, source: str) -> List[Dict]:
    header = table["header"]
    rows = table["rows"]
    header = [str(cell) if cell is not None else '' for cell in header]
    rows = [[str(cell) if cell is not None else '' for cell in row] for row in rows]

    # Skip incomplete tables
    if len(header) <= 1 or not any(row for row in rows if any(cell.strip() for cell in row)):
        logger.warning(f"Skipping incomplete table on pages {table['pages']}: {header}")
        return []

    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "|" + "-|" * len(header) + "\n"
    for row in rows:
        md_table += "| " + " | ".join(row) + " |\n"
    content = md_table.strip()

    # Ensure valid page numbers
    valid_pages = [p for p in table["pages"] if isinstance(p, int) and p > 0]
    if not valid_pages:
        valid_pages = [1]
    page_num = min(valid_pages)

    # Split large tables
    MAX_CHUNK_SIZE = 10000
    if len(content) > MAX_CHUNK_SIZE:
        split_chunks = []
        mid_point = len(rows) // 2
        for part_rows in [rows[:mid_point], rows[mid_point:]]:
            part_table = "| " + " | ".join(header) + " |\n"
            part_table += "|" + "-|" * len(header) + "\n"
            for row in part_rows:
                part_table += "| " + " | ".join(row) + " |\n"
            split_chunks.append(part_table.strip())
        return [{
            "page_content": chunk,
            "metadata": {
                "page_number": page_num,
                "source": source or "unknown",
                "type": "table"
            }
        } for chunk in split_chunks]
    else:
        return [{
            "page_content": content,
            "metadata": {
                "page_number": page_num,
                "source": source or "unknown",
                "type": "table"
            }
        }]

def normalize_content(content: str) -> str:
    """Normalize content for deduplication by removing extra whitespace and converting to lowercase."""
    return re.sub(r'\s+', ' ', content.strip().lower())

def chunk_pdfplumber_parsed_data(pages: List[Dict]) -> List[Dict]:
    chunks = []
    seen_content = set()
    current_table = None
    text_limit = 1500
    flex_limit = 1600

    for page in pages:
        page_num = page["metadata"]["page_number"]
        source = page["metadata"]["source"]

        # Process tables
        for table in page["tables"]:
            if table and table[0]:
                header = tuple(str(cell) if cell is not None else '' for cell in table[0])
                if current_table and current_table["header"] == header and current_table["pages"][-1] + 1 == page_num:
                    current_table["rows"].extend(table[1:])
                    current_table["pages"].append(page_num)
                else:
                    if current_table:
                        table_chunks = create_table_chunk(current_table, source)
                        for chunk in table_chunks:
                            content_hash = md5(normalize_content(chunk["page_content"]).encode()).hexdigest()
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                chunks.append(chunk)
                                logger.info(f"Created table chunk: {chunk['page_content'][:50]}... with metadata: {chunk['metadata']}")
                            else:
                                logger.warning(f"Skipped duplicate table chunk: {chunk['page_content'][:50]}...")
                    current_table = {
                        "header": header,
                        "rows": table[1:],
                        "pages": [page_num]
                    }

        # Process text
        text = page["text"]
        if text:
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para:
                    lower_para = para.lower()
                    section_type = (
                        "illustration" if "illustration" in lower_para else
                        "exhibit" if "exhibit" in lower_para else
                        "question" if any(q in lower_para for q in ["question", "questions for practice"]) else
                        "text"
                    )

                    if len(para) <= flex_limit:
                        content_hash = md5(normalize_content(para).encode()).hexdigest()
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            chunks.append({
                                "page_content": para,
                                "metadata": {
                                    "page_number": page_num,
                                    "source": source or "unknown",
                                    "type": section_type
                                }
                            })
                            logger.info(f"Created text chunk: {para[:50]}... with metadata: page={page_num}, type={section_type}")
                        else:
                            logger.warning(f"Skipped duplicate text chunk: {para[:50]}...")
                    else:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        temp_chunk = ""
                        for sentence in sentences:
                            if len(temp_chunk) + len(sentence) <= text_limit:
                                temp_chunk += sentence + " "
                            else:
                                if temp_chunk:
                                    content_hash = md5(normalize_content(temp_chunk.strip()).encode()).hexdigest()
                                    if content_hash not in seen_content:
                                        seen_content.add(content_hash)
                                        chunks.append({
                                            "page_content": temp_chunk.strip(),
                                            "metadata": {
                                                "page_number": page_num,
                                                "source": source or "unknown",
                                                "type": section_type
                                            }
                                        })
                                        logger.info(f"Created text chunk: {temp_chunk[:50]}... with metadata: page={page_num}, type={section_type}")
                                    else:
                                        logger.warning(f"Skipped duplicate text chunk: {temp_chunk[:50]}...")
                                    temp_chunk = sentence + " "
                        if temp_chunk:
                            content_hash = md5(normalize_content(temp_chunk.strip()).encode()).hexdigest()
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                chunks.append({
                                    "page_content": temp_chunk.strip(),
                                    "metadata": {
                                        "page_number": page_num,
                                        "source": source or "unknown",
                                        "type": section_type
                                    }
                                })
                                logger.info(f"Created text chunk: {temp_chunk[:50]}... with metadata: page={page_num}, type={section_type}")
                            else:
                                logger.warning(f"Skipped duplicate text chunk: {temp_chunk[:50]}...")

    # Finalize any remaining table
    if current_table:
        table_chunks = create_table_chunk(current_table, pages[0]["metadata"]["source"])
        for chunk in table_chunks:
            content_hash = md5(normalize_content(chunk["page_content"]).encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                chunks.append(chunk)
                logger.info(f"Created table chunk: {chunk['page_content'][:50]}... with metadata: {chunk['metadata']}")
            else:
                logger.warning(f"Skipped duplicate table chunk: {chunk['page_content'][:50]}...")

    return chunks