import logging

from langchain_core.tools import tool

from fhir_mcp_client import get_patient_summary
from dosage_client import estimate_renal_dose_adjustment
from input_helpers import ask_for_number

logger = logging.getLogger(__name__)


@tool
def recommend_with_dose(
    patient_id: str,
    new_drug_name: str,
    dose_text: str,
    reason: str,
) -> str:
    """Recommend a safer drug and dose for the patient, checks kidney function too."""
    patient_summary = get_patient_summary(patient_id=patient_id)

    age = patient_summary["age"]
    is_female = patient_summary["is_female"]
    weight_kg = patient_summary.get("weight_kg")
    serum_creatinine = patient_summary.get("serum_creatinine_mg_dl")
    conditions = patient_summary.get("conditions", [])

    if weight_kg is None:
        weight_kg = ask_for_number(
            prompt=f"Weight (kg) not found for patient {patient_id}, enter manually: "
        )

    if serum_creatinine is None:
        serum_creatinine = ask_for_number(
            prompt=f"Serum creatinine (mg/dL) not found for patient {patient_id}, enter manually: "
        )

    dose_check = estimate_renal_dose_adjustment(
        age=age,
        weight_kg=weight_kg,
        serum_creatinine_mg_dl=serum_creatinine,
        is_female=is_female,
    )

    estimated_crcl = dose_check["estimated_crcl_ml_min"]
    renal_guidance = dose_check["guidance"]

    logger.info(f"Patient {patient_id}: CrCl {estimated_crcl} mL/min, recommending {new_drug_name}")

    recommendation_text = f"""
Recommendation: switch to {new_drug_name}.
Why: {reason}
Suggested starting dose (from guideline): {dose_text}

Kidney function check: estimated creatinine clearance is {estimated_crcl} mL/min
(based on age {age}, weight {weight_kg} kg, serum creatinine {serum_creatinine} mg/dL).
What that means for dosing: {renal_guidance}
""".strip()

    if conditions:
        conditions_text = ", ".join(conditions)
        recommendation_text = f"{recommendation_text}\nOther conditions to keep in mind: {conditions_text}."

    return recommendation_text