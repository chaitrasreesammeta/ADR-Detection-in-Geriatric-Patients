import asyncio
import logging
from pathlib import Path
from typing import Any

from fastmcp import Client

logger = logging.getLogger(__name__)

DOSAGE_SERVER_SCRIPT = Path(__file__).parent / "dosage_mcp_server.py"


async def call_dosage_tool(
    tool_name: str,
    arguments: dict,
) -> Any:
    """
    Start the dosage MCP server, call one of its tools, and return the result.
    fastmcp's Client only works with async code, so this is the only async function.
    """
    logger.info(f"Calling dosage MCP tool: {tool_name}")

    async with Client(DOSAGE_SERVER_SCRIPT) as client:
        result = await client.call_tool(tool_name, arguments)

    return result.data


def estimate_renal_dose_adjustment(
    age: int,
    weight_kg: float,
    serum_creatinine_mg_dl: float,
    is_female: bool,
) -> dict:
    """
    Get the patient's estimated creatinine clearance and what it means for dosing.
    """
    arguments = {
        "age": age,
        "weight_kg": weight_kg,
        "serum_creatinine_mg_dl": serum_creatinine_mg_dl,
        "is_female": is_female,
    }

    return asyncio.run(
        call_dosage_tool(
            tool_name="estimate_renal_dose_adjustment",
            arguments=arguments,
        )
    )