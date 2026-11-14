import asyncio
import logging
from pathlib import Path
from typing import Any

from fastmcp import Client

logger = logging.getLogger(__name__)

FHIR_SERVER_SCRIPT = Path(__file__).parent / "fhir_mcp_server.py"


async def call_fhir_tool(
    tool_name: str,
    arguments: dict,
) -> Any:
    """
    Start the FHIR MCP server, call one of its tools, and return the result.
    fastmcp's Client only works with async code, so this is the only async function.
    """
    logger.info(f"Calling FHIR MCP tool: {tool_name}")

    async with Client(FHIR_SERVER_SCRIPT) as client:
        result = await client.call_tool(tool_name, arguments)

    return result.data


def get_patient_summary(patient_id: str) -> dict:
    """
    Get name, age, sex, weight, creatinine and conditions for the patient.
    """
    arguments = {
        "patient_id": patient_id,
    }

    return asyncio.run(
        call_fhir_tool(
            tool_name="get_patient_summary",
            arguments=arguments,
        )
    )


def get_medications_for_patient(patient_id: str) -> list[dict]:
    """
    Get the patient's active medications.
    """
    arguments = {
        "patient_id": patient_id,
    }

    return asyncio.run(
        call_fhir_tool(
            tool_name="get_medications_for_patient",
            arguments=arguments,
        )
    )


def post_medication_request(
    patient_id: str,
    drug_name: str,
    dose_mg: float,
    note: str,
) -> dict:
    """
    Add a new active medication for the patient in FHIR.
    """
    arguments = {
        "patient_id": patient_id,
        "drug_name": drug_name,
        "dose_mg": dose_mg,
        "note": note,
    }

    return asyncio.run(
        call_fhir_tool(
            tool_name="post_medication_request",
            arguments=arguments,
        )
    )


def discontinue_medication(
    patient_id: str,
    drug_name: str,
) -> dict:
    """
    Set the patient's active medication with this name to "stopped" in FHIR.
    """
    arguments = {
        "patient_id": patient_id,
        "drug_name": drug_name,
    }

    return asyncio.run(
        call_fhir_tool(
            tool_name="discontinue_medication",
            arguments=arguments,
        )
    )