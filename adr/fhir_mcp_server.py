from datetime import date

import requests
from fastmcp import FastMCP

from config import load_config, FHIR_BASE_URL, get_today
from rxnorm_client import normalize_drug_name

load_config()

mcp = FastMCP("FHIR Access")

HEADERS = {
    "Content-Type": "application/fhir+json",
    "Prefer": "return=representation",
}

# stops HAPI from giving old cached results
NO_CACHE = {
    "Cache-Control": "no-cache",
}

RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
WEIGHT_LOINC_CODE = "29463-7"
CREATININE_LOINC_CODE = "2160-0"
UNKNOWN_CONDITION = "Unknown condition"


def is_success(status_code: int) -> bool:
    """
    Check if FHIR answered 200 (OK) or 201 (Created).
    """
    if status_code == 200 or status_code == 201:
        return True

    return False


def fetch_patient(patient_id: str) -> dict:
    """
    Get the full Patient resource from FHIR.
    """
    url = f"{FHIR_BASE_URL}/Patient/{patient_id}"

    response = requests.get(url=url)

    return response.json()


def get_patient_name(patient: dict) -> str:
    """
    Build the full name, like "Lakshmi Reddy".
    Returns "Unknown" if the patient has no name.
    """
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


def calculate_age(birth_date_text: str) -> int:
    """
    Work out the age from the birth date.
    Takes off one year if the birthday has not come yet this year.
    """
    birth_date = date.fromisoformat(birth_date_text)
    today = get_today()

    age = today.year - birth_date.year

    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


def is_female_patient(patient: dict) -> bool:
    """
    Check if the patient's gender is female.
    """
    gender = patient.get("gender", "")

    if gender.lower() == "female":
        return True

    return False


def get_observation_value(
    patient_id: str,
    loinc_code: str,
) -> float | None:
    """
    Get one lab value (weight or creatinine) for the patient.
    Returns None if there is no result.
    """
    url = f"{FHIR_BASE_URL}/Observation"

    params = {
        "subject": f"Patient/{patient_id}",
        "code": loinc_code,
    }

    response = requests.get(
        url=url,
        params=params,
        headers=NO_CACHE,
    )

    data = response.json()
    entries = data.get("entry", [])

    if not entries:
        return None

    first_entry = entries[0]
    observation = first_entry["resource"]
    value_quantity = observation["valueQuantity"]

    return value_quantity["value"]


def get_condition_name(condition: dict) -> str:
    """
    Get the name of one condition, like "Glaucoma".
    Returns "Unknown condition" if it has no name.
    """
    code = condition.get("code", {})
    codings = code.get("coding", [])

    if not codings:
        return UNKNOWN_CONDITION

    first_coding = codings[0]

    return first_coding.get("display", UNKNOWN_CONDITION)


def get_condition_names(patient_id: str) -> list[str]:
    """
    Get the names of all the patient's conditions, like ["Glaucoma"].
    """
    url = f"{FHIR_BASE_URL}/Condition"

    params = {
        "subject": f"Patient/{patient_id}",
    }

    response = requests.get(
        url=url,
        params=params,
        headers=NO_CACHE,
    )

    data = response.json()
    entries = data.get("entry", [])

    condition_names = []

    for entry in entries:
        condition = entry["resource"]
        condition_name = get_condition_name(condition=condition)
        condition_names.append(condition_name)

    return condition_names


def fetch_active_medications(patient_id: str) -> list[dict]:
    """
    Get only the ACTIVE MedicationRequests for the patient.
    Stopped medications are not included.
    """
    url = f"{FHIR_BASE_URL}/MedicationRequest"

    params = {
        "subject": f"Patient/{patient_id}",
        "status": "active",
    }

    response = requests.get(
        url=url,
        params=params,
        headers=NO_CACHE,
    )

    data = response.json()

    return data.get("entry", [])


def get_medication_text(medication_request: dict) -> str:
    """
    Get the drug name text from a MedicationRequest, like "Diphenhydramine 25mg".
    """
    medication = medication_request.get("medicationCodeableConcept", {})

    return medication.get("text", "")


def find_medication_by_name(
    entries: list[dict],
    drug_name: str,
) -> dict | None:
    """
    Find the first medication whose text contains drug_name.
    Returns None if nothing matches.
    """
    drug_name_lower = drug_name.lower()

    for entry in entries:
        medication_request = entry["resource"]
        medication_text = get_medication_text(medication_request=medication_request)

        if drug_name_lower in medication_text.lower():
            return medication_request

    return None


@mcp.tool
def get_patient(patient_id: str) -> dict:
    """
    Return the full Patient resource.
    """
    return fetch_patient(patient_id=patient_id)


@mcp.tool
def get_patient_summary(patient_id: str) -> dict:
    """
    Return name, age, sex, weight, creatinine and conditions in one dictionary.
    """
    patient = fetch_patient(patient_id=patient_id)

    name = get_patient_name(patient=patient)
    age = calculate_age(birth_date_text=patient["birthDate"])
    is_female = is_female_patient(patient=patient)
    conditions = get_condition_names(patient_id=patient_id)

    weight_kg = get_observation_value(
        patient_id=patient_id,
        loinc_code=WEIGHT_LOINC_CODE,
    )

    serum_creatinine = get_observation_value(
        patient_id=patient_id,
        loinc_code=CREATININE_LOINC_CODE,
    )

    return {
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "is_female": is_female,
        "weight_kg": weight_kg,
        "serum_creatinine_mg_dl": serum_creatinine,
        "conditions": conditions,
    }


@mcp.tool
def get_medications_for_patient(patient_id: str) -> list[dict]:
    """
    Return the patient's active medications.
    """
    return fetch_active_medications(patient_id=patient_id)


@mcp.tool
def post_medication_request(
    patient_id: str,
    drug_name: str,
    dose_mg: float,
    note: str,
) -> dict:
    """
    Add a new active medication to FHIR.
    Adds the RxNorm code too, if RxNorm finds the drug.
    """
    rxcui, ingredient_name = normalize_drug_name(drug_name=drug_name)

    medication: dict = {
        "text": f"{drug_name} {dose_mg}mg",
    }

    if rxcui:
        medication["coding"] = [
            {
                "system": RXNORM_SYSTEM,
                "code": rxcui,
                "display": ingredient_name,
            }
        ]

    today_text = date.today().isoformat()

    body = {
        "resourceType": "MedicationRequest",
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": medication,
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "authoredOn": today_text,
        "note": [
            {
                "text": note,
            }
        ],
    }

    url = f"{FHIR_BASE_URL}/MedicationRequest"

    response = requests.post(
        url=url,
        json=body,
        headers=HEADERS,
    )

    status_code = response.status_code
    success = is_success(status_code=status_code)

    return {
        "success": success,
        "status_code": status_code,
    }


@mcp.tool
def discontinue_medication(
    patient_id: str,
    drug_name: str,
) -> dict:
    """
    Find the patient's active medication by name and set its status to "stopped".
    """
    entries = fetch_active_medications(patient_id=patient_id)

    medication_request = find_medication_by_name(
        entries=entries,
        drug_name=drug_name,
    )

    if not medication_request:
        return {
            "success": False,
            "reason": f"No active medication matching '{drug_name}' found for patient {patient_id}.",
        }

    medication_request["status"] = "stopped"
    medication_request_id = medication_request["id"]

    url = f"{FHIR_BASE_URL}/MedicationRequest/{medication_request_id}"

    response = requests.put(
        url=url,
        json=medication_request,
        headers=HEADERS,
    )

    status_code = response.status_code
    success = is_success(status_code=status_code)

    return {
        "success": success,
        "status_code": status_code,
    }


if __name__ == "__main__":
    mcp.run(
        show_banner=False,
        log_level="WARNING",
    )