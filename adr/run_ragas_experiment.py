import asyncio
from datetime import datetime

from ragas import Dataset, experiment

from chatbot_cli import (
    are_all_drugs_out_of_scope,
    build_out_of_scope_reply,
    build_patient_record,
    get_all_medication_names,
    get_section_names,
    retrieve_for_patient,
)
from config import PROJECT_ROOT, load_config
from load_vector_store import CHAT_MODEL
from ragas_single_example import (
    build_retrieved_contexts,
    get_chatbot_answer,
    score_with_ragas,
)

load_config()

DATASET_NAME = "adr_test_dataset"
DATASET_BACKEND = "local/csv"
EVALS_DIR = PROJECT_ROOT / "evaluation"
MAX_PATIENTS_AT_ONCE = 2
MAX_CHATBOT_RUNS_AT_ONCE = 1

patient_limit = asyncio.Semaphore(MAX_PATIENTS_AT_ONCE)
chatbot_limit = asyncio.Semaphore(MAX_CHATBOT_RUNS_AT_ONCE)

def run_chatbot_for_patient(
    patient_id: str,
    question: str,
) -> dict:
    """
    Run the chatbot steps for one patient: medications, Patient Record, retrieval, answer.
    Returns a dict with the answer, the retrieved contexts, the retrieved section names
    and "out_of_scope" (True when all drugs are outside the knowledge base, so nothing is retrieved).
    Returns a dict with an "error" message if a step finds nothing.
    """
    drug_names = get_all_medication_names(patient_id=patient_id)

    if not drug_names:
        return {
            "error": f"No medications found for patient {patient_id}. Is Docker running?",
        }

    patient_record = build_patient_record(
        patient_id=patient_id,
        drug_names=drug_names,
    )

    if are_all_drugs_out_of_scope(drug_names=drug_names):
        out_of_scope_reply = build_out_of_scope_reply(
            patient_record=patient_record,
            drug_names=drug_names,
        )

        return {
            "answer": out_of_scope_reply,
            "retrieved_contexts": [],
            "section_names": [],
            "out_of_scope": True,
        }

    documents = retrieve_for_patient(
        drug_names=drug_names,
        user_question=question,
    )

    if not documents:
        return {
            "error": "No guideline chunks were retrieved.",
        }

    retrieved_contexts = build_retrieved_contexts(
        patient_record=patient_record,
        documents=documents,
    )

    answer = get_chatbot_answer(
        patient_record=patient_record,
        documents=documents,
        question=question,
    )

    section_names = get_section_names(documents=documents)

    return {
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "section_names": section_names,
        "out_of_scope": False,
    }


def build_empty_ragas_scores() -> dict:
    """
    Build empty RAGAS scores for a patient where nothing was retrieved, so the context metrics cannot be scored.
    Returns a dict with the four RAGAS metric names and empty values.
    """
    return {
        "faithfulness": "",
        "context_precision": "",
        "context_recall": "",
        "response_relevancy": "",
    }


def build_error_row(
    row: dict,
    error_message: str,
) -> dict:
    """
    Build a result row for a patient whose chatbot run failed, so it still shows up in the results.
    Returns a dict with empty scores and the error message.
    """
    return {
        "test_id": row["test_id"],
        "patient_id": row["patient_id"],
        "user_input": row["user_input"],
        "reference": row["reference"],
        "response": "",
        "retrieved_sections": "",
        "faithfulness": "",
        "context_precision": "",
        "context_recall": "",
        "response_relevancy": "",
        "error": error_message,
    }


@experiment()
async def adr_experiment(row: dict) -> dict:
    """
    Run the chatbot on one dataset row and score its answer with the four RAGAS metrics.
    Returns one result row with the answer, the retrieved sections and the scores.
    """
    test_id = row["test_id"]
    patient_id = row["patient_id"]
    user_input = row["user_input"]
    reference = row["reference"]

    async with patient_limit:
        print(f"Running {test_id} (patient {patient_id})")

        async with chatbot_limit:
            chatbot_result = await asyncio.to_thread(
                run_chatbot_for_patient,
                patient_id=patient_id,
                question=user_input,
            )

        if "error" in chatbot_result:
            return build_error_row(
                row=row,
                error_message=chatbot_result["error"],
            )

        answer = chatbot_result["answer"]
        retrieved_contexts = chatbot_result["retrieved_contexts"]
        section_names = chatbot_result["section_names"]
        out_of_scope = chatbot_result["out_of_scope"]

        if out_of_scope:
            scores = build_empty_ragas_scores()
        else:
            scores = await score_with_ragas(
                question=user_input,
                answer=answer,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )

    return {
        "test_id": test_id,
        "patient_id": patient_id,
        "user_input": user_input,
        "reference": reference,
        "response": answer,
        "retrieved_sections": "\n".join(section_names),
        "faithfulness": scores["faithfulness"],
        "context_precision": scores["context_precision"],
        "context_recall": scores["context_recall"],
        "response_relevancy": scores["response_relevancy"],
        "error": "",
    }


def make_experiment_name() -> str:
    """
    Make a name for this run from the chatbot model and the current date and time,
    like "ragas-gpt-4o-mini-2026-11-13-21-30".
    So every run is saved to a new CSV, older runs are kept, and the model is visible.
    """
    now = datetime.now()

    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    return f"ragas-{CHAT_MODEL}-{year}-{month}-{day}-{hour}-{minute}"


def run_experiment() -> None:
    """
    Load the RAGAS test dataset, run the experiment on every row,
    and save the results as a CSV in evaluation/experiments/.
    """
    dataset = Dataset.load(
        name=DATASET_NAME,
        backend=DATASET_BACKEND,
        root_dir=str(EVALS_DIR),
    )

    print(f"Loaded {len(dataset)} test cases from {DATASET_NAME}")

    experiment_name = make_experiment_name()

    asyncio.run(
        adr_experiment.arun(
            dataset=dataset,
            name=experiment_name,
        )
    )

    results_file = EVALS_DIR / "experiments" / f"{experiment_name}.csv"

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    run_experiment()
