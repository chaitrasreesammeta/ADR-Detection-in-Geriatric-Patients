import json
from datetime import date
from pathlib import Path

import requests

from config import load_config, FHIR_BASE_URL, DATA_DIR
from rxnorm_client import normalize_drug_name

load_config()

HEADERS = {
    "Content-Type": "application/fhir+json",
    "Prefer": "return=representation",
}

# stops HAPI from giving old cached results
NO_CACHE = {
    "Cache-Control": "no-cache",
}

TEST_PATIENT_SYSTEM = "urn:adr-capstone:test-patient"
SNOMED_SYSTEM = "http://snomed.info/sct"
LOINC_SYSTEM = "http://loinc.org"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"
UCUM_SYSTEM = "http://unitsofmeasure.org"
CONDITION_CLINICAL_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-clinical"
DOSE_RATE_TYPE_SYSTEM = "http://terminology.hl7.org/CodeSystem/dose-rate-type"
MEDICATION_REQUEST_SYSTEM = "urn:adr-capstone:medication-request"

ORAL_ROUTE_CODE = "26643006"
ORAL_ROUTE_DISPLAY = "Oral route (qualifier value)"
PRESCRIBER_DISPLAY = "ADR Capstone Test Prescriber"

WEIGHT_LOINC_CODE = "29463-7"
WEIGHT_DISPLAY = "Body weight"
CREATININE_LOINC_CODE = "2160-0"
CREATININE_DISPLAY = "Creatinine [Mass/volume] in Serum or Plasma"

INPUT_FILE = DATA_DIR / "test_patients.json"
OUTPUT_FILE = DATA_DIR / "test_patient_ids.json"


def load_test_patients(input_file: Path) -> dict:
    """
    Read the test patient definitions (conditions and patients) from the JSON file.
    """
    file_text = input_file.read_text(encoding="utf-8")

    return json.loads(file_text)


def find_existing_patient(test_id: str) -> str | None:
    """
    Check if this test patient is already in FHIR (so running the script twice doesn't make duplicates).
    Returns the FHIR patient ID, or None if the patient isn't there yet.
    """
    url = f"{FHIR_BASE_URL}/Patient"

    params = {
        "identifier": f"{TEST_PATIENT_SYSTEM}|{test_id}",
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
    patient = first_entry["resource"]

    return patient["id"]


def post_resource(
    resource_type: str,
    resource: dict,
) -> str | None:
    """
    Create one FHIR resource (Patient, Condition, Observation or MedicationRequest).
    Returns the new resource ID, or None if FHIR did not create it.
    """
    url = f"{FHIR_BASE_URL}/{resource_type}"

    response = requests.post(
        url=url,
        json=resource,
        headers=HEADERS,
    )

    if response.status_code != 201:
        print(f"  Could not create {resource_type}: HTTP {response.status_code}")
        return None

    created_resource = response.json()

    return created_resource["id"]


def build_patient(test_patient: dict) -> dict:
    """
    Build the FHIR Patient resource for one test patient.
    """
    return {
        "resourceType": "Patient",
        "identifier": [
            {
                "system": TEST_PATIENT_SYSTEM,
                "value": test_patient["test_id"],
            }
        ],
        "name": [
            {
                "family": test_patient["family"],
                "given": [test_patient["given"]],
            }
        ],
        "gender": test_patient["gender"],
        "birthDate": test_patient["birth_date"],
    }


def build_condition(
    patient_id: str,
    condition: dict,
) -> dict:
    """
    Build an active FHIR Condition with its SNOMED CT code.
    """
    return {
        "resourceType": "Condition",
        "clinicalStatus": {
            "coding": [
                {
                    "system": CONDITION_CLINICAL_SYSTEM,
                    "code": "active",
                }
            ]
        },
        "code": {
            "coding": [
                {
                    "system": SNOMED_SYSTEM,
                    "code": condition["code"],
                    "display": condition["display"],
                }
            ],
            "text": condition["display"],
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
    }


def build_observation(
    patient_id: str,
    loinc_code: str,
    display: str,
    value: float,
    unit: str,
) -> dict:
    """
    Build a FHIR Observation (lab value or vital sign) with its LOINC code.
    """
    today_text = date.today().isoformat()

    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": LOINC_SYSTEM,
                    "code": loinc_code,
                    "display": display,
                }
            ],
            "text": display,
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "effectiveDateTime": today_text,
        "valueQuantity": {
            "value": value,
            "unit": unit,
            "system": UCUM_SYSTEM,
            "code": unit,
        },
    }


def get_unit_display(dose_unit: str) -> str:
    """
    Turn a UCUM unit code into the text a person reads, like "ug" into "mcg".
    """
    if dose_unit == "ug":
        return "mcg"

    if dose_unit == "{tablet}":
        return "tablet"

    return dose_unit


def build_reason_code(condition: dict) -> list:
    """
    Build the R4 reasonCode (the disease the drug is for) with its SNOMED CT code.
    """
    return [
        {
            "coding": [
                {
                    "system": SNOMED_SYSTEM,
                    "code": condition["code"],
                    "display": condition["display"],
                }
            ],
            "text": condition["display"],
        }
    ]


def build_dosage_instruction(test_medication: dict) -> dict:
    """
    Build one dosageInstruction with the text, how often (timing), route and dose amount.
    """
    dose_unit = test_medication["dose_unit"]
    unit_display = get_unit_display(dose_unit=dose_unit)

    dosage_instruction = {
        "sequence": 1,
        "text": test_medication["dosage"],
        "timing": {
            "repeat": {
                "frequency": test_medication["frequency"],
                "period": 1,
                "periodUnit": test_medication["period_unit"],
            }
        },
        "asNeededBoolean": test_medication["as_needed"],
        "route": {
            "coding": [
                {
                    "system": SNOMED_SYSTEM,
                    "code": ORAL_ROUTE_CODE,
                    "display": ORAL_ROUTE_DISPLAY,
                }
            ]
        },
        "doseAndRate": [
            {
                "type": {
                    "coding": [
                        {
                            "system": DOSE_RATE_TYPE_SYSTEM,
                            "code": "ordered",
                            "display": "Ordered",
                        }
                    ]
                },
                "doseQuantity": {
                    "value": test_medication["dose_value"],
                    "unit": unit_display,
                    "system": UCUM_SYSTEM,
                    "code": dose_unit,
                },
            }
        ],
    }

    return dosage_instruction


def build_medication_request(
    patient_id: str,
    test_id: str,
    medication_number: int,
    test_medication: dict,
    condition_lookup: dict,
) -> dict:
    """
    Build an active FHIR R4 MedicationRequest with the drug and its RxNorm code,
    the reason (disease) with its SNOMED CT code, and a structured dose (timing, route, amount).
    test_medication looks like {"text": "Gabapentin 600 mg", "dosage": "1 capsule by mouth three times daily",
    "reason": "postherpetic_neuralgia", "dose_value": 600, "dose_unit": "mg", "frequency": 3,
    "period_unit": "d", "as_needed": false}.
    """
    medication_text = test_medication["text"]

    rxcui, ingredient_name = normalize_drug_name(drug_name=medication_text)

    medication: dict = {
        "text": medication_text,
    }

    if rxcui:
        medication["coding"] = [
            {
                "system": RXNORM_SYSTEM,
                "code": rxcui,
                "display": ingredient_name,
            }
        ]
    else:
        print(f"  Warning: RxNorm could not confirm '{medication_text}', saved as text only")

    reason_key = test_medication["reason"]
    reason_condition = condition_lookup[reason_key]
    reason_code = build_reason_code(condition=reason_condition)

    dosage_instruction = build_dosage_instruction(test_medication=test_medication)

    today_text = date.today().isoformat()

    return {
        "resourceType": "MedicationRequest",
        "identifier": [
            {
                "use": "official",
                "system": MEDICATION_REQUEST_SYSTEM,
                "value": f"{test_id}-{medication_number}",
            }
        ],
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": medication,
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "authoredOn": today_text,
        "requester": {
            "display": PRESCRIBER_DISPLAY,
        },
        "reasonCode": reason_code,
        "dosageInstruction": [
            dosage_instruction,
        ],
    }


def create_test_patient(
    test_patient: dict,
    condition_lookup: dict,
) -> str | None:
    """
    Create one test patient in FHIR with their conditions, weight, creatinine and medications.
    Returns the new FHIR patient ID, or None if the patient could not be created.
    """
    patient = build_patient(test_patient=test_patient)

    patient_id = post_resource(
        resource_type="Patient",
        resource=patient,
    )

    if not patient_id:
        return None

    for condition_key in test_patient["conditions"]:
        condition = build_condition(
            patient_id=patient_id,
            condition=condition_lookup[condition_key],
        )

        post_resource(
            resource_type="Condition",
            resource=condition,
        )

    weight_kg = test_patient["weight_kg"]

    if weight_kg is not None:
        weight = build_observation(
            patient_id=patient_id,
            loinc_code=WEIGHT_LOINC_CODE,
            display=WEIGHT_DISPLAY,
            value=weight_kg,
            unit="kg",
        )

        post_resource(
            resource_type="Observation",
            resource=weight,
        )

    creatinine_mg_dl = test_patient["creatinine_mg_dl"]

    if creatinine_mg_dl is not None:
        creatinine = build_observation(
            patient_id=patient_id,
            loinc_code=CREATININE_LOINC_CODE,
            display=CREATININE_DISPLAY,
            value=creatinine_mg_dl,
            unit="mg/dL",
        )

        post_resource(
            resource_type="Observation",
            resource=creatinine,
        )

    test_id = test_patient["test_id"]
    medication_number = 0

    for test_medication in test_patient["medications"]:
        medication_number = medication_number + 1

        medication_request = build_medication_request(
            patient_id=patient_id,
            test_id=test_id,
            medication_number=medication_number,
            test_medication=test_medication,
            condition_lookup=condition_lookup,
        )

        post_resource(
            resource_type="MedicationRequest",
            resource=medication_request,
        )

    return patient_id


def save_patient_ids(patient_ids: dict) -> None:
    """
    Save which FHIR patient ID belongs to each test patient (TP01, TP02, ...) for the evaluation.
    """
    file_text = json.dumps(
        patient_ids,
        indent=2,
    )

    OUTPUT_FILE.write_text(
        data=file_text,
        encoding="utf-8",
    )


def main() -> None:
    """
    Create every test patient in FHIR (skipping ones that already exist) and save their IDs.
    """
    test_data = load_test_patients(input_file=INPUT_FILE)
    condition_lookup = test_data["conditions"]
    test_patients = test_data["patients"]

    patient_ids = {}

    for test_patient in test_patients:
        test_id = test_patient["test_id"]
        full_name = f"{test_patient['given']} {test_patient['family']}"

        existing_id = find_existing_patient(test_id=test_id)

        if existing_id:
            print(f"{test_id} {full_name}: already in FHIR as Patient/{existing_id}, skipped")
            patient_ids[test_id] = existing_id
            continue

        print(f"{test_id} {full_name}: creating...")

        patient_id = create_test_patient(
            test_patient=test_patient,
            condition_lookup=condition_lookup,
        )

        if not patient_id:
            print(f"{test_id} {full_name}: FAILED")
            continue

        print(f"{test_id} {full_name}: created as Patient/{patient_id}")
        patient_ids[test_id] = patient_id

    save_patient_ids(patient_ids=patient_ids)

    print(f"\n{len(patient_ids)} test patients saved to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
