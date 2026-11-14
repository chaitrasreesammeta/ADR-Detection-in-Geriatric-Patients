from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config import CLEANED_MARKDOWN_FILE
from langchain_core.documents import Document
import uuid


HEADERS = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]

markdown_file = CLEANED_MARKDOWN_FILE.read_text()


def chunked_documents():
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    documents = splitter.split_text(text=markdown_file)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=60)
    split_documents = recursive_splitter.split_documents(documents=documents)

    for doc in split_documents:
        doc.metadata["source"] = CLEANED_MARKDOWN_FILE.name
        doc.metadata["chunk_id"] = f"{CLEANED_MARKDOWN_FILE.stem}_{uuid.uuid4().hex}"

    return split_documents


if __name__ == "__main__":
    docs = chunked_documents()
    for _doc in docs:
        print(_doc.metadata)
        print("----" * 10)
        print(_doc.page_content)
        print("----" * 40)