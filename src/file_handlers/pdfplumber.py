import os
from langchain_community.document_loaders import PDFPlumberLoader

def parse_file(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return PDFPlumberLoader(filepath).load()

# 🧪 Test it locally
if __name__ == "__main__":
    test_file = "apple_statement.pdf"
    docs = parse_file(test_file)
    print(docs)
