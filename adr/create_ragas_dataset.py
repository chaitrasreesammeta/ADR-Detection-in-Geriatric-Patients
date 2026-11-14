import json

from ragas import Dataset

from config import DATA_DIR, PROJECT_ROOT, load_config

load_config()

DATASET_NAME = "adr_test_dataset"
DATASET_BACKEND = "local/csv"
EVALS_DIR = PROJECT_ROOT / "evaluation"
TEST_PATIENTS_FILE = DATA_DIR / "test_patients.json"
TEST_PATIENT_IDS_FILE = DATA_DIR / "test_patient_ids.json"
QUESTION_TEMPLATE = "Check patient {patient_id} for ADR risks"


def load_test_patients() -> list[dict]:
    """
    Read the test patients (with their expected_findings) from test_patients.json.
    Returns the list of patient dicts.
    """
    with open(TEST_PATIENTS_FILE) as file:
        test_data = json.load(file)

    return test_data["patients"]


def load_test_patient_ids() -> dict:
    """
    Read the FHIR IDs of the test patients from test_patient_ids.json.
    Returns a dict like {"TP01": "1101", "TP02": "1113"}.
    """
    with open(TEST_PATIENT_IDS_FILE) as file:
        return json.load(file)


def build_dataset_row(
    test_id: str,
    patient_id: str,
    expected_findings: list[str],
) -> dict:
    """
    Build one dataset row: the question for the chatbot and the answer key.
    Returns a dict with test_id, patient_id, user_input and reference.
    """
    user_input = QUESTION_TEMPLATE.format(patient_id=patient_id)
    reference = "\n".join(expected_findings)

    return {
        "test_id": test_id,
        "patient_id": patient_id,
        "user_input": user_input,
        "reference": reference,
    }


def create_dataset() -> None:
    """
    Create the RAGAS test dataset (one row per test patient) and save it as a CSV
    in evaluation/datasets/.
    """
    test_patients = load_test_patients()
    test_patient_ids = load_test_patient_ids()

    dataset = Dataset(
        name=DATASET_NAME,
        backend=DATASET_BACKEND,
        root_dir=str(EVALS_DIR),
    )

    for patient in test_patients:
        test_id = patient["test_id"]

        if test_id not in test_patient_ids:
            print(f"Skipping {test_id}: no FHIR ID in {TEST_PATIENT_IDS_FILE.name}. Run create_test_patients.py first.")
            continue

        patient_id = test_patient_ids[test_id]
        expected_findings = patient["expected_findings"]

        row = build_dataset_row(
            test_id=test_id,
            patient_id=patient_id,
            expected_findings=expected_findings,
        )

        dataset.append(row)

    dataset.save()

    dataset_file = EVALS_DIR / "datasets" / f"{DATASET_NAME}.csv"

    print(f"Saved {len(dataset)} rows to {dataset_file}")


if __name__ == "__main__":
    create_dataset()
