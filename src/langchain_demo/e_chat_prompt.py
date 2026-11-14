from config import load_config
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_config()


def main():
    model = ChatOpenAI(model="gpt-4o-mini")

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based on the context below. If the question cannot be answered with the information provided, just say IDK"),
            ("human", "Context: {context}"),
            ("human", "Question: {question}"),
        ]
    )

    prompt = template.invoke({
        "context": "Baahubali: The Beginning along with its successor, is widely regarded as one of the most influential films of Indian cinema. It became the first Indian film to be nominated for Saturn Awards, receiving five nominations at the 42nd ceremony, including Best Fantasy Film and Best Supporting Actress. It received several accolades such as the National Film Award for Best Feature Film, and the National Award for Best Special Effects.",
        "question": "Explain about bahubali movie?"
    })

    response = model.invoke(prompt)
    print(response.content)


if __name__ == '__main__':
    main()