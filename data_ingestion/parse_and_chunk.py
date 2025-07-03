'''take the pdf, parses the data with pdfplumber, meke chunks according to the text-table logic, remove duplicated chunks and return deduplicated chunks'''
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

    pdf = pdfplumber.open(filepath)
    pages = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        tables = page.extract_tables() or []
        pages.append({
            "text": text,
            "tables": tables,
            "metadata": {
                "page_number": page.page_number,
                "source": os.path.basename(filepath)
            }
        })
    pdf.close()
    return pages

def create_table_chunk(table: Dict, source: str) -> List[Dict]:
    header = table["header"]
    rows = table["rows"]
    header = [str(cell) if cell is not None else '' for cell in header]
    rows = [[str(cell) if cell is not None else '' for cell in row] for row in rows]

    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "|" + "-|" * len(header) + "\n"
    for row in rows:
        md_table += "| " + " | ".join(row) + " |\n"
    content = md_table.strip()

    # Ensure valid page numbers
    valid_pages = [p for p in table["pages"] if isinstance(p, int) and p > 0]
    if not valid_pages:
        valid_pages = [1]  # Fallback to page 1
    page_num = min(valid_pages)

    if len(content) > 10000:
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

def chunk_pdfplumber_parsed_data(pages: List[Dict]) -> List[Dict]:
    chunks = []
    seen_content = set()  # For deduplicating chunks
    current_table = None
    text_limit = 1500
    flex_limit = 1600

    for page in pages:
        page_num = page["metadata"]["page_number"]
        source = page["metadata"]["source"]

        # Tables
        for table in page["tables"]:
            if table and table[0]:
                header = tuple(table[0])
                # Only merge tables from consecutive pages with matching headers
                if current_table and current_table["header"] == header and current_table["pages"][-1] + 1 == page_num:
                    current_table["rows"].extend(table[1:])
                    current_table["pages"].append(page_num)
                else:
                    if current_table:
                        chunks.extend(create_table_chunk(current_table, source))
                    current_table = {
                        "header": header,
                        "rows": table[1:],
                        "pages": [page_num]
                    }

        # Text with headers detection
        text = page["text"]
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para:
                # Enhanced section type detection
                lower_para = para.lower()
                section_type = (
                    "illustration" if "illustration" in lower_para else
                    "exhibit" if "exhibit" in lower_para else
                    "question" if any(q in lower_para for q in ["question", "questions for practice"]) else
                    "text"
                )

                if len(para) <= flex_limit:
                    chunks.append({
                        "page_content": para,
                        "metadata": {
                            "page_number": page_num,
                            "source": source or "unknown",
                            "type": section_type
                        }
                    })
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= text_limit:
                            temp_chunk += sentence + " "
                        else:
                            if temp_chunk:
                                chunks.append({
                                    "page_content": temp_chunk.strip(),
                                    "metadata": {
                                        "page_number": page_num,
                                        "source": source or "unknown",
                                        "type": section_type
                                    }
                                })
                                temp_chunk = sentence + " "
                    if temp_chunk:
                        chunks.append({
                            "page_content": temp_chunk.strip(),
                            "metadata": {
                                "page_number": page_num,
                                "source": source or "unknown",
                                "type": section_type
                            }
                        })

    if current_table:
        chunks.extend(create_table_chunk(current_table, pages[0]["metadata"]["source"]))

    # Deduplicate chunks
    deduped_chunks = []
    for chunk in chunks:
        content_hash = md5(chunk["page_content"].encode()).hexdigest()
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            deduped_chunks.append(chunk)
            logger.info(f"Created chunk: {chunk['page_content'][:50]}... with metadata: {chunk['metadata']}")

    return deduped_chunks
