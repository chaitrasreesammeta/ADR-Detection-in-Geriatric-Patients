import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import CLEANED_MARKDOWN_FILE

CHUNK_SIZE = 800
CHUNK_OVERLAP = 60

HEADERS = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]


def split_by_headers(markdown_text: str) -> list[Document]:
    """
    Split the markdown at each header.
    Each piece remembers its drug name (header_1) and section (header_2) as metadata.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )

    return splitter.split_text(markdown_text)


def split_by_size(documents: list[Document]) -> list[Document]:
    """
    Cut long sections into chunks of about 800 characters.
    The 60-character overlap keeps sentences at the edges from being lost.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def add_chunk_metadata(
    documents: list[Document],
    markdown_path: Path,
) -> list[Document]:
    """
    Add the file name and a unique chunk ID to every chunk (Chroma needs an ID for each one).
    """
    for document in documents:
        random_id = uuid.uuid4().hex

        document.metadata["source"] = markdown_path.name
        document.metadata["chunk_id"] = f"{markdown_path.stem}_{random_id}"

    return documents


def chunked_documents(markdown_path: Path) -> list[Document]:
    """
    Read the markdown file and return it as chunks with metadata.
    """
    markdown_text = markdown_path.read_text(encoding="utf-8")

    header_documents = split_by_headers(markdown_text=markdown_text)
    chunks = split_by_size(documents=header_documents)

    return add_chunk_metadata(
        documents=chunks,
        markdown_path=markdown_path,
    )


def main() -> None:
    chunks = chunked_documents(markdown_path=CLEANED_MARKDOWN_FILE)

    print(f"{len(chunks)} chunks created from {CLEANED_MARKDOWN_FILE.name}")


if __name__ == "__main__":
    main()