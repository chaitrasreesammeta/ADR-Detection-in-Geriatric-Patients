import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from chatbot_cli import (
    SYSTEM_PROMPT,
    build_patient_record,
    format_context,
    get_all_medication_names,
    retrieve_for_patient,
)
from config import DATA_DIR, load_config
from input_helpers import find_patient_id
from load_vector_store import EMBEDDING_MODEL, get_chat_llm_model
from recommendation_tool import recommend_with_dose

load_config()

TEST_ID = "TP01"
QUESTION = "Check patient 1101 for ADR risks"
JUDGE_MODEL = "gpt-4o"
JUDGE_MAX_TOKENS = 8192
TEST_PATIENTS_FILE = DATA_DIR / "test_patients.json"


def load_reference_answer(test_id: str) -> str:
    """
    Read the expected_findings (answer key) for one test patient from test_patients.json.
    Returns the findings joined into one text, one finding per line.
    Returns an empty string if the test patient is not in the file.
    """
    with open(TEST_PATIENTS_FILE) as file:
        test_data = json.load(file)

    patients = test_data["patients"]

    for patient in patients:
        if patient["test_id"] != test_id:
            continue

        expected_findings = patient["expected_findings"]

        return "\n".join(expected_findings)

    return ""


def build_retrieved_contexts(
    patient_record: str,
    documents: list,
) -> list[str]:
    """
    Turn the Patient Record and each retrieved chunk into a list of texts for RAGAS.
    The Patient Record goes first because the chatbot also uses it to answer.
    Returns a list of strings, one per context.
    """
    retrieved_contexts = [patient_record]

    for document in documents:
        chunk_text = format_context(documents=[document])
        retrieved_contexts.append(chunk_text)

    return retrieved_contexts


def get_chatbot_answer(
    patient_record: str,
    documents: list,
    question: str,
) -> str:
    """
    Ask gpt-4o-mini the question the same way chatbot_node does, with no earlier messages.
    Returns the chatbot's answer text.
    """
    context = format_context(documents=documents)

    question_with_context = f"""
{patient_record}

Guideline Context:
{context}

Question: {question}
""".strip()

    messages_for_llm = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question_with_context),
    ]

    model = get_chat_llm_model()
    model_with_tool = model.bind_tools([recommend_with_dose])

    response = model_with_tool.invoke(messages_for_llm)

    if not response.tool_calls:
        return str(response.content)

    tool_calls = response.tool_calls
    first_tool_call = tool_calls[0]
    tool_arguments = first_tool_call["args"]

    tool_result = recommend_with_dose.invoke(tool_arguments)

    return str(tool_result)


async def score_with_ragas(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    reference: str,
) -> dict:
    """
    Score one question and answer with four RAGAS metrics, using gpt-4o as the judge.
    Returns a dict with the metric names and their scores (0 to 1).
    """
    client = AsyncOpenAI()

    judge_llm = llm_factory(
        model=JUDGE_MODEL,
        client=client,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    judge_embeddings = embedding_factory(
        provider="openai",
        model=EMBEDDING_MODEL,
        client=client,
    )

    faithfulness_metric = Faithfulness(llm=judge_llm)
    context_precision_metric = ContextPrecision(llm=judge_llm)
    context_recall_metric = ContextRecall(llm=judge_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    faithfulness_result = await faithfulness_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts,
    )

    context_precision_result = await context_precision_metric.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )

    context_recall_result = await context_recall_metric.ascore(
        user_input=question,
        retrieved_contexts=retrieved_contexts,
        reference=reference,
    )

    answer_relevancy_result = await answer_relevancy_metric.ascore(
        user_input=question,
        response=answer,
    )

    scores = {
        "faithfulness": faithfulness_result.value,
        "context_precision": context_precision_result.value,
        "context_recall": context_recall_result.value,
        "response_relevancy": answer_relevancy_result.value,
    }

    return scores


def print_scores(scores: dict) -> None:
    """
    Print each metric name and its score, rounded to 2 decimals.
    """
    print("\nRAGAS scores:")

    for metric_name in scores:
        score = scores[metric_name]

        if score is None:
            print(f"  {metric_name}: no score")
            continue

        print(f"  {metric_name}: {round(score, 2)}")


def run_single_example() -> None:
    """
    Run the chatbot on one test patient and one question, then score the answer with RAGAS.
    """
    patient_id = find_patient_id(user_input=QUESTION)

    if not patient_id:
        print(f"No patient ID found in the question: {QUESTION}")
        return

    reference = load_reference_answer(test_id=TEST_ID)

    if not reference:
        print(f"No expected_findings found for {TEST_ID} in {TEST_PATIENTS_FILE.name}")
        return

    drug_names = get_all_medication_names(patient_id=patient_id)

    if not drug_names:
        print(f"No medications found for patient {patient_id}. Is Docker running?")
        return

    patient_record = build_patient_record(
        patient_id=patient_id,
        drug_names=drug_names,
    )

    documents = retrieve_for_patient(
        drug_names=drug_names,
        user_question=QUESTION,
    )

    if not documents:
        print("No guideline chunks were retrieved, so there is nothing to score.")
        return

    retrieved_contexts = build_retrieved_contexts(
        patient_record=patient_record,
        documents=documents,
    )

    answer = get_chatbot_answer(
        patient_record=patient_record,
        documents=documents,
        question=QUESTION,
    )

    print(f"Question: {QUESTION}")
    print(f"Retrieved contexts: {len(retrieved_contexts)} (Patient Record + {len(documents)} chunks)")
    print(f"\nChatbot answer:\n{answer}")
    print(f"\nReference (answer key):\n{reference}")

    scores = asyncio.run(
        score_with_ragas(
            question=QUESTION,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
    )

    print_scores(scores=scores)


if __name__ == "__main__":
    run_single_example()
