import logging

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10

NO_CACHE = {
    "Cache-Control": "no-cache",
}

HEADERS = {
    "Content-Type": "application/fhir+json",
    "Prefer": "return=representation",
}


def get_fhir_json(
    url: str,
    params: dict,
) -> dict:
    """
    Send a GET request to the FHIR server and return the answer as a dictionary.
    Returns an empty dictionary if the server can't be reached or doesn't answer 200 (OK),
    so the code that uses it never crashes.
    """
    try:
        response = requests.get(
            url=url,
            params=params,
            headers=NO_CACHE,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        logger.error(f"Could not reach the FHIR server at {url}. Is Docker running?")
        return {}

    if response.status_code != 200:
        logger.warning(f"FHIR answered {response.status_code} for {url}")
        return {}

    return response.json()


def post_fhir_resource(
    url: str,
    resource: dict,
) -> requests.Response | None:
    """
    Send a POST request to create a new FHIR resource.
    Returns the server's response, or None if the server can't be reached.
    """
    try:
        response = requests.post(
            url=url,
            json=resource,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        logger.error(f"Could not reach the FHIR server at {url}. Is Docker running?")
        return None

    return response


def put_fhir_resource(
    url: str,
    resource: dict,
) -> requests.Response | None:
    """
    Send a PUT request to replace a FHIR resource with an updated version.
    Returns the server's response, or None if the server can't be reached.
    """
    try:
        response = requests.put(
            url=url,
            json=resource,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        logger.error(f"Could not reach the FHIR server at {url}. Is Docker running?")
        return None

    return response


def delete_fhir_resource(url: str) -> requests.Response | None:
    """
    Send a DELETE request to remove one FHIR resource.
    Returns the server's response, or None if the server can't be reached.
    """
    try:
        response = requests.delete(
            url=url,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        logger.error(f"Could not reach the FHIR server at {url}. Is Docker running?")
        return None

    return response
