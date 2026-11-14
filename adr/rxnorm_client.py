import requests

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
REQUEST_TIMEOUT_SECONDS = 10
INGREDIENT_TERM_TYPE = "IN"


def get_json(
    url: str,
    params: dict,
) -> dict | None:
    """
    Send a GET request to the RxNorm API and return the JSON response.
    Returns None if the request fails or RxNorm does not answer with 200 (OK).
    """
    try:
        response = requests.get(
            url=url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    return response.json()


def get_rxcui(drug_name: str) -> str | None:
    """
    Find the RxNorm code (RxCUI) for a drug name that RxNorm recognizes exactly.
    Returns None if there is no exact match.
    """
    url = f"{RXNORM_BASE_URL}/rxcui.json"

    params = {
        "name": drug_name,
        "search": 2,
    }

    data = get_json(
        url=url,
        params=params,
    )

    if not data:
        return None

    id_group = data.get("idGroup", {})
    rxcui_list = id_group.get("rxnormId", [])

    if not rxcui_list:
        return None

    return rxcui_list[0]


def get_approximate_rxcui(drug_name: str) -> str | None:
    """
    Find the closest RxNorm code for a drug name that may have a typo, like "Loratadie".
    Returns None if RxNorm finds nothing close.
    """
    url = f"{RXNORM_BASE_URL}/approximateTerm.json"

    params = {
        "term": drug_name,
        "maxEntries": 1,
    }

    data = get_json(
        url=url,
        params=params,
    )

    if not data:
        return None

    approximate_group = data.get("approximateGroup", {})
    candidates = approximate_group.get("candidate", [])

    if not candidates:
        return None

    first_candidate = candidates[0]

    return first_candidate.get("rxcui")


def get_ingredient_rxcui(rxcui: str) -> tuple[str | None, str | None]:
    """
    Find the ingredient (like "loratadine") that an RxNorm code belongs to.
    Returns (ingredient RxCUI, ingredient name), or (None, None) if there isn't one.
    """
    url = f"{RXNORM_BASE_URL}/rxcui/{rxcui}/related.json"

    params = {
        "tty": INGREDIENT_TERM_TYPE,
    }

    data = get_json(
        url=url,
        params=params,
    )

    if not data:
        return None, None

    related_group = data.get("relatedGroup", {})
    concept_groups = related_group.get("conceptGroup", [])

    for group in concept_groups:
        if group.get("tty") != INGREDIENT_TERM_TYPE:
            continue

        properties = group.get("conceptProperties", [])

        if not properties:
            continue

        first_concept = properties[0]
        ingredient_rxcui = first_concept.get("rxcui")
        ingredient_name = first_concept.get("name")

        return ingredient_rxcui, ingredient_name

    return None, None


def get_concept_properties(rxcui: str) -> tuple[str | None, str | None]:
    """
    Ask RxNorm what an RxNorm code is.
    Returns (term type, name), for example ("IN", "sertraline"), or (None, None).
    """
    url = f"{RXNORM_BASE_URL}/rxcui/{rxcui}/properties.json"

    data = get_json(
        url=url,
        params={},
    )

    if not data:
        return None, None

    properties = data.get("properties", {})
    term_type = properties.get("tty")
    concept_name = properties.get("name")

    return term_type, concept_name


def normalize_drug_name(drug_name: str) -> tuple[str | None, str | None]:
    """
    Turn any drug text (brand name, typo, or name with a dose) into its RxNorm ingredient.
    Returns (ingredient RxCUI, ingredient name), or (None, None) if RxNorm can't confirm it.
    It never returns the typed text, so a typo is never saved as a drug name.
    """
    rxcui = get_rxcui(drug_name=drug_name)

    if not rxcui:
        rxcui = get_approximate_rxcui(drug_name=drug_name)

    if not rxcui:
        return None, None

    ingredient_rxcui, ingredient_name = get_ingredient_rxcui(rxcui=rxcui)

    if ingredient_name:
        return ingredient_rxcui, ingredient_name

    term_type, concept_name = get_concept_properties(rxcui=rxcui)

    if term_type == INGREDIENT_TERM_TYPE:
        return rxcui, concept_name

    return None, None


def main() -> None:
    drug_names = [
        "Diphenhydramine 25mg",
        "Valium 5mg",
        "Amitriptyline",
        "Loratadie",
        "Setaline",
        "Sertraline",
    ]

    for drug_name in drug_names:
        rxcui, ingredient_name = normalize_drug_name(drug_name=drug_name)
        print(f"{drug_name} -> rxcui={rxcui}, ingredient={ingredient_name}")


if __name__ == "__main__":
    main()