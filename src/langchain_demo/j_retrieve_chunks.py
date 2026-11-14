from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from config import load_config


load_config()
vector_store_dir = Path(__file__).parent / "chroma_db"

def main():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(vector_store_dir),
        embedding_function=embedding,
        collection_name="sample_chunks"
    )

    question = input("Ask a question from the document: ")
    retrieved_docs = vector_store.similarity_search(query=question, k=3)


    for doc in retrieved_docs:
        print("---")
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content)


if __name__ == '__main__':
    main()