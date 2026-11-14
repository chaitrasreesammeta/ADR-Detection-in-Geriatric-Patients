from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import load_config, CLEANED_MARKDOWN_FILE, VECTOR_DIR
from splitter import chunked_documents

load_config()

COLLECTION_NAME = "drugs_only"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

CHAT_TEMPERATURE = 0


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Get the OpenAI model that turns text into number vectors (embeddings).
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_chat_llm_model() -> ChatOpenAI:
    """
    Get the OpenAI chat model that writes the answers.
    """
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
    )


def get_vector_store() -> Chroma:
    """
    Open the saved Chroma collection from the chroma folder.
    """
    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DIR),
    )


def load_vector() -> None:
    """
    Rebuild the Chroma collection from DRUGS.MD.
    Run this every time DRUGS.MD changes.
    """
    vector_store = get_vector_store()
    vector_store.delete_collection()

    drug_chunks = chunked_documents(markdown_path=CLEANED_MARKDOWN_FILE)

    chunk_ids = []

    for chunk in drug_chunks:
        chunk_ids.append(chunk.metadata["chunk_id"])

    vector_store = get_vector_store()

    vector_store.add_documents(
        documents=drug_chunks,
        ids=chunk_ids,
    )

    print(f"{len(drug_chunks)} chunks from {CLEANED_MARKDOWN_FILE.name} added to '{COLLECTION_NAME}'")


if __name__ == "__main__":
    load_vector()