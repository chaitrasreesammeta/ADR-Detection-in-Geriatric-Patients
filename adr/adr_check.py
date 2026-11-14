from load_vector_store import get_vector_store, get_chat_llm_model
from config import load_config
from langchain_core.prompts import ChatPromptTemplate
from fhir_client import get_medications_for_patient
from rxnorm_client import get_ingredient_rxcui, normalize_drug_name

load_config()


def get_all_medication_names(patient_id):
    medications = get_medications_for_patient(patient_id)
    if not medications:
        return []

    names = []
    for entry in medications:
        resource = entry["resource"]
        contained = resource.get("contained", [])
        medication_ref = resource.get("medicationReference", {}).get("reference", "")
        contained_id = medication_ref.lstrip("#")
        matched = next((c for c in contained if c.get("id") == contained_id), None)

        if matched:
            coding = matched.get("code", {}).get("coding", [{}])[0]
            rxcui = coding.get("code")
            display = coding.get("display", "")
            if rxcui:
                _, ingredient_name = get_ingredient_rxcui(rxcui)
                names.append((ingredient_name or display).capitalize())
            else:
                names.append(display.capitalize())
        else:
            text = resource.get("medicationCodeableConcept", {}).get("text")
            if text:
                _, ingredient_name = normalize_drug_name(text)
                names.append((ingredient_name or text).capitalize())

    return names


def check():
    patient_id = "1004"

    drug_names = get_all_medication_names(patient_id)
    if not drug_names:
        print("No medications found for this patient")
        return

    print("Medications for this patient:", ", ".join(drug_names))

    if len(drug_names) < 2:
        print("Only one medication found -- no combination to check.")
        return

    vector_store = get_vector_store()

    all_documents = []
    for drug_name in drug_names:
        metadata_filter = {"header_1": {"$eq": drug_name}}
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": metadata_filter
            }
        )
        search_query = f"What are the mechanism, adverse drug reaction profile, and interactions of {drug_name} in older adults?"
        documents = retriever.invoke(search_query)
        all_documents.extend(documents)

    if not all_documents:
        print("No guideline information found for these medications")
        return

    print(f"Retrieved {len(all_documents)} documents across {len(drug_names)} drugs:")
    for i, doc in enumerate(all_documents, start=1):
        print(f"{i}. header_1={doc.metadata.get('header_1')} | header_2={doc.metadata.get('header_2')}")
    print()

    combination_question = (
        f"This patient is taking {' and '.join(drug_names)} together. "
        f"List the specific adverse drug reactions (ADRs) this combination "
        f"can cause in an older adult -- name the actual symptoms and "
        f"conditions (e.g., confusion, dry mouth, urinary retention, falls), "
        f"not just the underlying mechanism. State clearly if any of these "
        f"ADRs are worsened or made more likely because both drugs act "
        f"through the same or related mechanism."
    )

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
- Answer in terms of specific adverse drug reactions (named symptoms or
  conditions), not abstract descriptions of mechanism overlap.
- If multiple medications are involved, list the ADRs each drug can cause
  and clearly flag which ADRs are increased by taking both drugs together.
- Be concise but complete.
""".strip()),
            ("human", "Context: {context}"),
            ("human", "Question: {question}"),
        ]
    )

    context_parts = []
    for doc in all_documents:
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
            "question": combination_question
        }
    )

    chat_model = get_chat_llm_model()
    response = chat_model.invoke(input=prompt)
    print(response.content)


if __name__ == '__main__':
    check()