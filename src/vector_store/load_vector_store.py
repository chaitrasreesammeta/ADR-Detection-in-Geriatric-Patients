from document_processing.splitter import chunked_documents
from config import load_config,VECTOR_DIR
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma

load_config()

collection_name = "adverse_drug_reaction"
def get_emdedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_chat_llm_model():
    return ChatOpenAI(model="gpt-4o-mini")

def get_vector_store():
    vector_store = Chroma(
        collection_name="collection_name",
        embedding_function=get_emdedding_model(),
        persist_directory= str(VECTOR_DIR),
    )
    return vector_store

def load_vector():
    vector_store = get_vector_store()
    docs = chunked_documents()
    id_list = [doc.metadata["chunk_id"] for doc in docs]
    print(id_list)
    vector_store.add_documents(documents=docs, ids=id_list)
    print(f"Document added to {collection_name }")


if __name__ == "__main__":
    load_vector()