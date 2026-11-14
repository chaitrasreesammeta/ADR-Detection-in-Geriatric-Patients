from config import load_config


SYSTEM_PROMPT = """
You are a clinical decision support assistant helping a physician assess
adverse drug reaction (ADR) risk in geriatric patients.

Answer the physician's question using only the retrieved guideline
context provided with each turn.

Rules:
- Do not use outside knowledge.
- If the retrieved context does not contain enough information to answer,
  say that the available documents do not provide enough information.
- Do not make a diagnosis.
- Answer in terms of specific adverse drug reactions (named symptoms or
  conditions), not abstract mechanism descriptions.
- Be concise but complete.
""".strip()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_config()



def main()
    model = ChatOpenAI(model="gpt-40-mini")
    template = PromptTemplate.from_template(
        """
        Answer the question based on the context below. If the question answer cannot be answere with information provided, just say IDK
        
        Context: {context}
        
        Question: {question}
        """
    )

prompt = template.invoke({
    "Context": "Baahubali: The Beginning along with its successor, is widely regarded as one of the most influential films of Indian cinema. It became the first Indian film to be nominated for Saturn Awards, receiving five nominations at the 42nd ceremony, including Best Fantasy Film and Best Supporting Actress. It received several accolades such as the National Film Award for Best Feature Film, and the National Award for Best Special Effects. ",
    "question": "Explain about bahubali movie?"
})
response = model.invoke({input=prompt})
print(response.content)

if __name__ == '__main__':
    main()
