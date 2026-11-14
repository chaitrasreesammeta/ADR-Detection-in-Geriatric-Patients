from config import load_config
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path


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
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    model = ChatOpenAI(model="gpt-4o-mini")
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based on the context below. If the question cannot be answered with the information provided, just say IDK"),
            ("human", "Context: {context}"),
            ("human", "Question: {question}"),
        ]
    )
    chain = template | model

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print(response.content)


if __name__ == '__main__':
    main()