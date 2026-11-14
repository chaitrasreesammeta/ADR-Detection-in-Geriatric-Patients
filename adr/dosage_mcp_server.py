from fastmcp import FastMCP

mcp = FastMCP("Dosage Calculator")

FEMALE_FACTOR = 0.85

SEVERE_CRCL_LIMIT = 30
MODERATE_CRCL_LIMIT = 50
MILD_CRCL_LIMIT = 60

CITATION = (
    "Cockcroft DW, Gault MH. Prediction of creatinine clearance from serum creatinine. "
    "Nephron. 1976;16(1):31-41."
)


def calculate_creatinine_clearance(
    age: int,
    weight_kg: float,
    serum_creatinine_mg_dl: float,
    is_female: bool,
) -> float:
    """
    Estimate creatinine clearance (mL/min) with the Cockcroft-Gault formula:
    ((140 - age) x weight) / (serum creatinine x 72), times 0.85 if female.
    """
    crcl = ((140 - age) * weight_kg) / (serum_creatinine_mg_dl * 72)

    if is_female:
        crcl = crcl * FEMALE_FACTOR

    return round(crcl, 1)


def get_renal_guidance(crcl: float) -> str:
    """
    Turn the creatinine clearance into a kidney function level and what it means for dosing.
    """
    if crcl < SEVERE_CRCL_LIMIT:
        return "Severe impairment: reduce dose substantially and extend intervals; consult a dosing reference."

    if crcl < MODERATE_CRCL_LIMIT:
        return "Moderate impairment: a lower dose or longer dosing interval may be needed; consult a dosing reference."

    if crcl < MILD_CRCL_LIMIT:
        return "Mild impairment: usually no adjustment, but check the drug's renal dosing guidance."

    return "Normal kidney function: no renal dose adjustment needed."


@mcp.tool
def estimate_renal_dose_adjustment(
    age: int,
    weight_kg: float,
    serum_creatinine_mg_dl: float,
    is_female: bool,
) -> dict:
    """
    Estimate kidney function (Cockcroft-Gault) and say what it means for dosing.
    It does not pick a starting dose; that comes from the guideline.
    """
    if serum_creatinine_mg_dl <= 0:
        return {
            "estimated_crcl_ml_min": None,
            "guidance": "Serum creatinine must be greater than 0 to estimate kidney function.",
            "note": CITATION,
        }

    crcl = calculate_creatinine_clearance(
        age=age,
        weight_kg=weight_kg,
        serum_creatinine_mg_dl=serum_creatinine_mg_dl,
        is_female=is_female,
    )

    guidance = get_renal_guidance(crcl=crcl)

    return {
        "estimated_crcl_ml_min": crcl,
        "guidance": guidance,
        "note": CITATION,
    }


if __name__ == "__main__":
    mcp.run(
        show_banner=False,
        log_level="WARNING",
    )