from vector_store.load_vector_store import get_vector_store
from config import load_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_config()


def get_chat_llm_model():
    return ChatOpenAI(model="gpt-4o-mini")


def chat():
    search_query = "Explain the Epidemiology of Adverse Drug Reactions"
    vector_store = get_vector_store()
    metadata_filter = {
        "$and": [
            {"header_1": {"$eq": "Adverse drug reaction"}},
            {"header_2": {"$eq": "Epidemiology"}},
        ]
    }
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": metadata_filter
        }
    )
    documents = retriever.invoke(search_query)
    if not documents:
        print("No documents found")
        return

    template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Answer the question based on the context below. If the question cannot be answered with the information provided, just say IDK"),
            ("human", "Context: {context}"),
            ("human", "Question: {question}"),
        ]
    )

    context_parts = []
    for doc in documents:
        header_1 = doc.metadata.get("header_1")
        header_2 = doc.metadata.get("header_2")
        content = doc.page_content
        part = f"""
{header_1}
{header_2}
{content}
"""
        context_parts.append(part.strip())

    prompt = template.invoke(
        {
            "context": context_parts,
            "question": search_query
        }
    )
    response = chat_model.invoke(input=prompt)
    print(response.content)


if __name__ == '__main__':
    chat()