import logging
from datetime import date

from config import load_config, FHIR_BASE_URL, get_today
from fhir_requests import get_fhir_json

load_config()

logger = logging.getLogger(__name__)

MAX_PATIENTS = 100

WEIGHT_LOINC_CODE = "29463-7"
CREATININE_LOINC_CODE = "2160-0"


def get_patient(patient_id: str) -> dict:
    """
    Get the full Patient resource from FHIR.
    """
    url = f"{FHIR_BASE_URL}/Patient/{patient_id}"

    return get_fhir_json(
        url=url,
        params={},
    )


def get_all_patients() -> list[dict]:
    """
    Get every Patient on the FHIR server (up to MAX_PATIENTS).
    """
    url = f"{FHIR_BASE_URL}/Patient"

    params = {
        "_count": MAX_PATIENTS,
    }

    data = get_fhir_json(
        url=url,
        params=params,
    )

    return data.get("entry", [])


def get_patient_name(patient_id: str) -> str:
    """
    Build the patient's full name, like "Lakshmi Reddy".
    Returns "Unknown" if the patient has no name.
    """
    patient = get_patient(patient_id=patient_id)
    name_list = patient.get("name", [])

    if not name_list:
        return "Unknown"

    first_name = name_list[0]
    given_names = first_name.get("given", [])
    given = " ".join(given_names)
    family = first_name.get("family", "")

    full_name = f"{given} {family}".strip()

    if not full_name:
        return "Unknown"

    return full_name


def get_patient_demographics(patient_id: str) -> tuple[int | None, bool | None]:
    """
    Get the patient's age and whether they are female.
    Returns (None, None) if the patient or their birth date can't be found.
    """
    patient = get_patient(patient_id=patient_id)
    birth_date_text = patient.get("birthDate", "")

    if not birth_date_text:
        return None, None

    birth_date = date.fromisoformat(birth_date_text)
    today = get_today()

    age = today.year - birth_date.year

    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    gender = patient.get("gender", "")

    if gender.lower() == "female":
        is_female = True
    else:
        is_female = False

    return age, is_female


def get_medications_for_patient(patient_id: str) -> list[dict]:
    """
    Get only the ACTIVE MedicationRequests for the patient.
    Stopped medications are not included.
    """
    url = f"{FHIR_BASE_URL}/MedicationRequest"

    params = {
        "subject": f"Patient/{patient_id}",
        "status": "active",
    }

    data = get_fhir_json(
        url=url,
        params=params,
    )
    entries = data.get("entry", [])

    logger.info(f"Found {len(entries)} active medications for patient {patient_id}")

    return entries


def get_conditions_for_patient(patient_id: str) -> list[str]:
    """
    Get the names of all the patient's conditions, like ["Glaucoma"].
    """
    url = f"{FHIR_BASE_URL}/Condition"

    params = {
        "subject": f"Patient/{patient_id}",
    }

    data = get_fhir_json(
        url=url,
        params=params,
    )
    entries = data.get("entry", [])

    condition_names = []

    for entry in entries:
        condition = entry["resource"]
        code = condition.get("code", {})
        codings = code.get("coding", [])

        if codings:
            first_coding = codings[0]
            condition_name = first_coding.get("display", "Unknown condition")
            condition_names.append(condition_name)
        else:
            condition_names.append("Unknown condition")

    return condition_names


def get_observation(
    patient_id: str,
    loinc_code: str,
) -> float | None:
    """
    Get one lab value for the patient by its LOINC code.
    Returns None if there is no result.
    """
    url = f"{FHIR_BASE_URL}/Observation"

    params = {
        "subject": f"Patient/{patient_id}",
        "code": loinc_code,
    }

    data = get_fhir_json(
        url=url,
        params=params,
    )
    entries = data.get("entry", [])

    if not entries:
        return None

    first_entry = entries[0]
    observation = first_entry["resource"]
    value_quantity = observation.get("valueQuantity", {})

    return value_quantity.get("value")


def get_weight_and_creatinine(patient_id: str) -> tuple[float | None, float | None]:
    """
    Get the patient's weight (kg) and serum creatinine (mg/dL).
    """
    weight = get_observation(
        patient_id=patient_id,
        loinc_code=WEIGHT_LOINC_CODE,
    )

    creatinine = get_observation(
        patient_id=patient_id,
        loinc_code=CREATININE_LOINC_CODE,
    )

    return weight, creatinine


def main() -> None:
    """
    Print the name and active medications for every patient on the server.
    """
    patients = get_all_patients()

    print(f"{len(patients)} patients on the FHIR server\n")

    for patient_entry in patients:
        patient = patient_entry["resource"]
        patient_id = patient["id"]
        patient_name = get_patient_name(patient_id=patient_id)

        medications = get_medications_for_patient(patient_id=patient_id)

        print(f"Patient {patient_id} - {patient_name}:")

        for medication_entry in medications:
            medication_request = medication_entry["resource"]
            medication = medication_request.get("medicationCodeableConcept", {})
            medication_text = medication.get("text", "(coded, no display)")
            print(f"  {medication_text}")

        print()


if __name__ == "__main__":
    main()