from load_vector_store import get_vector_store, get_chat_llm_model
from config import load_config
from langchain_core.prompts import ChatPromptTemplate

load_config()


def chat():
    search_query = "What are the adverse drug reaction risks of diphenhydramine in older adults?"
    vector_store = get_vector_store()
    metadata_filter = {"header_2": {"$eq": "Diphenhydramine"}}
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
             """
Answer the user's question using only the retrieved context.

Rules:
- Do not use outside knowledge.
- If the retrieved context does not contain enough information to answer,
  say that the available documents do not provide enough information.
- Do not make a diagnosis.
- Be concise but complete.
""".strip()),
            ("human", "Context: {context}"),
            ("human", "Question: {question}"),
        ]
    )

    context_parts = []
    for doc in documents:
        header_1 = doc.metadata.get("header_1")
        header_2 = doc.metadata.get("header_2")
        header_3 = doc.metadata.get("header_3")
        content = doc.page_content
        part = f"""
{header_1}
{header_2}
{header_3}
{content}
"""
        context_parts.append(part.strip())

    prompt = template.invoke(
        {
            "context": context_parts,
            "question": search_query
        }
    )

    chat_model = get_chat_llm_model()
    response = chat_model.invoke(input=prompt)
    print(response.content)


if __name__ == '__main__':
    chat()