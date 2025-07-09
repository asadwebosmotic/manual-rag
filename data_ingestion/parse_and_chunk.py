import os
import re
from typing import List, Dict
import pdfplumber
from hashlib import md5
import logging
from langchain_community.document_loaders import PDFPlumberLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_file(filepath: str) -> List[Dict]:
    if os.path.splitext(filepath)[1].lower() != ".pdf":
        raise ValueError("Unsupported file type")

    pages = []
    total_pages = 0
    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
    logger.info(f"Total pages in PDF: {total_pages}")

    try:
        # Try using PDFPlumberLoader with OCR for scanned documents
        loader = PDFPlumberLoader(filepath, extract_images=True)
        documents = loader.load()
        logger.info(f"PDFPlumberLoader extracted {len(documents)} pages")

        # Create a map of page numbers to content
        page_content_map = {doc.metadata.get("page", 1): doc.page_content or "" for doc in documents}

        # Process each page, ensuring all pages are included
        for page_num in range(1, total_pages + 1):
            text = page_content_map.get(page_num, "")
            logger.info(f"Processing page {page_num} with PDFPlumberLoader text length: {len(text)}")

            # Clean up excessive newlines and spaces
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

            # Use pdfplumber for table extraction
            with pdfplumber.open(filepath) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]  # pdfplumber is 0-indexed
                    tables = page.extract_tables() or []
                    table_texts = []
                    table_bboxes = []
                    found_tables = page.find_tables()
                    for table, table_obj in zip(tables, found_tables):
                        table_text = "\n".join(" ".join(str(cell) if cell is not None else "" for cell in row) for row in table)
                        table_texts.append(table_text)
                        table_bboxes.append(table_obj.bbox if table_obj else None)

                    # Filter out text within table regions
                    if text and table_bboxes:
                        def filter_non_table(obj):
                            for bbox in table_bboxes:
                                if bbox and obj.get('x0', float('inf')) >= bbox[0] and obj.get('x1', 0) <= bbox[2] and \
                                   obj.get('top', float('inf')) >= bbox[1] and obj.get('bottom', 0) <= bbox[3]:
                                    return False
                            return True
                        filtered_page = page.filter(filter_non_table)
                        filtered_text = filtered_page.extract_text() or ""
                        for table_text in table_texts:
                            filtered_text = filtered_text.replace(table_text, "")
                        text = "\n".join(line.strip() for line in filtered_text.splitlines() if line.strip())

                    pages.append({
                        "text": text,
                        "tables": tables,
                        "metadata": {
                            "page_number": page_num,
                            "source": os.path.basename(filepath)
                        }
                    })
                else:
                    logger.warning(f"Page {page_num} out of range for pdfplumber")

        # Check if all pages were processed
        if len(pages) < total_pages:
            logger.warning(f"PDFPlumberLoader missed {total_pages - len(pages)} pages, falling back to pdfplumber for missing pages")

    except Exception as e:
        logger.warning(f"PDFPlumberLoader failed: {str(e)}. Falling back to pdfplumber for all pages.")

    # Fallback to pdfplumber for missing pages or if PDFPlumberLoader failed
    if len(pages) < total_pages:
        with pdfplumber.open(filepath) as pdf:
            for page_num in range(1, total_pages + 1):
                if page_num not in [p["metadata"]["page_number"] for p in pages]:
                    page = pdf.pages[page_num - 1]
                    tables = page.extract_tables() or []
                    table_texts = []
                    table_bboxes = []
                    found_tables = page.find_tables()
                    for table, table_obj in zip(tables, found_tables):
                        table_text = "\n".join(" ".join(str(cell) if cell is not None else "" for cell in row) for row in table)
                        table_texts.append(table_text)
                        table_bboxes.append(table_obj.bbox if table_obj else None)

                    text = page.extract_text()
                    if not text:  # Try OCR with pdfplumber for scanned pages
                        try:
                            images = page.images
                            if images:
                                # Convert first image to text using pdfplumber's OCR (requires pytesseract)
                                text = page.to_image().ocr().text or ""
                                logger.info(f"Extracted OCR text for page {page_num} using pdfplumber")
                        except Exception as ocr_e:
                            logger.warning(f"pdfplumber OCR failed for page {page_num}: {str(ocr_e)}")
                            text = ""

                    if text and table_bboxes:
                        def filter_non_table(obj):
                            for bbox in table_bboxes:
                                if bbox and obj.get('x0', float('inf')) >= bbox[0] and obj.get('x1', 0) <= bbox[2] and \
                                   obj.get('top', float('inf')) >= bbox[1] and obj.get('bottom', 0) <= bbox[3]:
                                    return False
                            return True
                        filtered_page = page.filter(filter_non_table)
                        filtered_text = filtered_page.extract_text() or text
                        for table_text in table_texts:
                            filtered_text = filtered_text.replace(table_text, "")
                        text = "\n".join(line.strip() for line in filtered_text.splitlines() if line.strip())

                    pages.append({
                        "text": text,
                        "tables": tables,
                        "metadata": {
                            "page_number": page_num,
                            "source": os.path.basename(filepath)
                        }
                    })
                    logger.info(f"Fallback: Processed page {page_num} with pdfplumber")

    return pages

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