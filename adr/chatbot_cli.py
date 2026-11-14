import logging
import sqlite3
from datetime import datetime

import typer
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from load_vector_store import get_vector_store, get_chat_llm_model
from config import CLEANED_MARKDOWN_FILE, load_config
from fhir_client import get_medications_for_patient
from rxnorm_client import get_ingredient_rxcui, normalize_drug_name
from fhir_mcp_client import post_medication_request, discontinue_medication, get_patient_summary
from recommendation_tool import recommend_with_dose
from dosage_client import estimate_renal_dose_adjustment
from input_helpers import ask_for_number, find_patient_id

load_config()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a clinical decision support assistant helping a physician assess
adverse drug reaction (ADR) risk in geriatric patients.

You may be given two kinds of information: (1) a Patient Record section
listing the patient's actual medications, which is a known fact you
should state directly and confidently, and (2) a Guideline Context
section with retrieved clinical guideline content.

Rules:
- If asked what medications a patient is taking, answer directly from
  the Patient Record section. Do not hedge or say information is
  missing if the Patient Record section is provided.
- For questions about adverse drug reactions, risks, or interactions,
  answer using only the Guideline Context section. Do not use outside
  knowledge.
- If the Guideline Context does not contain enough information to
  answer an ADR-related question, say so plainly.
- Do not make a diagnosis.
- Answer in terms of specific adverse drug reactions (named symptoms or
  conditions), not abstract mechanism descriptions.
- For each drug, list only the adverse drug reactions that matter for
  this patient because of their age, kidney function, known conditions,
  or another drug they take. List at most 3 per drug, one short line
  each. Do not copy every side effect from the Guideline Context.
- Be concise but complete.
- Only list an interaction if the Guideline Context text names both
  drugs, or names one drug and the exact name of the other.
- Only list an interaction if both drugs are in the Patient Record's
  'Currently taking' list. Ignore interactions in the Guideline Context
  that mention drugs the patient does not take.
- If a drug is not in the knowledge base, do not assign a severity for
  it; state that severity cannot be assessed.
- Each drug in the Patient Record is labeled (in knowledge base) or
  (not in knowledge base). Trust these labels. For a drug labeled (not
  in knowledge base), give no ADRs, no dose check and no severity of its
  own; write '<drug>: not in the knowledge base.' But if the Guideline
  Context of a drug that is in the knowledge base names it as an
  interacting drug, list that interaction.
- Only flag a dose if it is above a maximum dose, or above a reduced
  dose the Guideline Context requires for this patient's age, weight or
  kidney function. A dose above the usual starting dose is not a
  problem by itself. Also flag a dose that is below the dose the
  Guideline Context states for this patient.
- When the patient takes two or more medications, always check for
  interactions between them, even if the question does not ask about
  interactions. Put them first, under the heading "Interactions",
  before the per-drug ADRs. Only list drug pairs that the Guideline
  Context says interact. Do not list pairs with no interaction.
- Put dose problems next, under the heading "Dose checks".
- Check each dose in the Patient Record against the Dosing
  Considerations section of the Guideline Context. For drugs that need
  a kidney dose change, compare against the Estimated CrCl. Flag any
  dose that is higher than the Guideline Context allows for this
  patient's age, weight or kidney function.
- Give every ADR risk you flag a severity of Mild, Moderate, or Severe.
  Use only these three words.
- Severity definitions: Severe means the drug, dose or combination
  should generally be avoided in older adults because the risk likely
  outweighs the benefit. Moderate means it needs monitoring, a dose
  change, or a careful risk-benefit decision. Mild means low clinical
  significance, usually routine monitoring.
- Base each severity only on the Guideline Context. If it does not
  support a rating, write "Severity: not enough information."
- End with one line in this format: "Overall severity: Mild, Moderate,
  or Severe - one sentence of reason from the Guideline Context." If
  every drug in the answer is not in the knowledge base, end instead
  with "Overall severity: cannot be assessed - not in the knowledge
  base."
- If the patient is younger than 65, say that geriatric-specific
  criteria do not apply, and only mention general interaction risks.
- If the Patient Record says 'Geriatric criteria: do not apply', say so
  in the first line of the answer, do not use elderly doses or geriatric
  criteria, and only mention general interaction risks.
- When you flag a drug as Severe or Moderate, and the Alternatives
  section of the Guideline Context for that drug states an alternative,
  name it. Do not name an alternative that is not stated there. Do not
  give a dose for it unless the Guideline Context states it. Do not call
  recommend_with_dose for this.
- Only call recommend_with_dose when the user asks for an alternative,
  a replacement, or a recommendation.
- When calling recommend_with_dose, set dose_text to the starting dose
  stated in the Guideline Context's Dosing Considerations section for the
  recommended drug. If no dose is stated there, set dose_text to "not
  specified in the guideline, confirm with a dosing reference." Never
  state a dose that is not in the Guideline Context.
""".strip()

TOP_K = 7

INTERACTION_SECTIONS = [
    "Interactions",
    "Adverse Drug Reaction Profile",
    "Mechanism",
    "Alternatives",
    "Dosing Considerations",
    "Monitoring",
    "Urgent Signs",
]

MUST_HAVE_SECTIONS = [
    "Dosing Considerations",
    "Interactions",
]

MUST_HAVE_QUERY = "dose adjustment for age, weight and kidney function, and drug interactions"

MUST_HAVE_TOP_K = 4

GERIATRIC_AGE = 65

APIXABAN_AGE_LIMIT = 80

APIXABAN_WEIGHT_LIMIT_KG = 60

APIXABAN_CREATININE_LIMIT = 1.5

APIXABAN_CRITERIA_FOR_REDUCED_DOSE = 2

APIXABAN_REDUCED_DOSE = "2.5 mg twice daily"

APIXABAN_STANDARD_DOSE = "5 mg twice daily"

DETAIL_WORDS = [
    "detail",
    "record",
    "info",
    "who is",
    "demographic",
]

RISK_WORDS = [
    "adr",
    "risk",
    "interaction",
    "severe",
    "severity",
    "side effect",
    "recommend",
    "alternative",
    "switch",
    "safe",
    "dose",
]

EXIT_COMMANDS = [
    "exit",
    "quit",
]

APPLY_COMMANDS = [
    "/apply",
    "apply",
    "\\apply",
]


def find_contained_medication(medication_request: dict) -> dict | None:
    """
    Find the Medication stored inside the MedicationRequest ("contained" format).
    Returns None if there isn't one.
    """
    contained_resources = medication_request.get("contained", [])
    medication_reference = medication_request.get("medicationReference", {})
    reference = medication_reference.get("reference", "")
    contained_id = reference.lstrip("#")

    for contained_resource in contained_resources:
        if contained_resource.get("id") == contained_id:
            return contained_resource

    return None


def get_name_from_contained_medication(medication: dict) -> str:
    """
    Get the ingredient name from a contained Medication using its RxNorm code.
    """
    code = medication.get("code", {})
    codings = code.get("coding", [])

    if not codings:
        return ""

    first_coding = codings[0]
    rxcui = first_coding.get("code")
    display = first_coding.get("display", "")

    if not rxcui:
        return display.capitalize()

    ingredient_rxcui, ingredient_name = get_ingredient_rxcui(rxcui=rxcui)

    if not ingredient_name:
        return display.capitalize()

    return ingredient_name.capitalize()


def get_name_from_medication_text(medication_request: dict) -> str:
    """
    Get the ingredient name from the free-text drug name, cleaned up through RxNorm.
    """
    medication = medication_request.get("medicationCodeableConcept", {})
    text = medication.get("text", "")

    if not text:
        return ""

    ingredient_rxcui, ingredient_name = normalize_drug_name(drug_name=text)

    if not ingredient_name:
        return text.capitalize()

    return ingredient_name.capitalize()


def get_all_medication_names(patient_id: str) -> list[str]:
    """
    Return the ingredient names of the patient's active meds, like ["Diazepam", "Amitriptyline"].
    """
    medications = get_medications_for_patient(patient_id=patient_id)

    names = []

    for entry in medications:
        medication_request = entry["resource"]
        contained_medication = find_contained_medication(medication_request=medication_request)

        if contained_medication:
            name = get_name_from_contained_medication(medication=contained_medication)
        else:
            name = get_name_from_medication_text(medication_request=medication_request)

        if name:
            names.append(name)

    return names


def get_reason_text(medication_request: dict) -> str:
    """
    Get what the drug is for from the first reasonCode's text, like "Atrial fibrillation".
    Returns an empty string if the MedicationRequest has no reasonCode.
    """
    reason_list = medication_request.get("reasonCode", [])

    if not reason_list:
        return ""

    first_reason = reason_list[0]

    return first_reason.get("text", "")


def get_dose_text(medication_request: dict) -> str:
    """
    Get the drug, strength, how to take it and what it is for,
    like "Apixaban 5 mg - 1 tablet by mouth twice daily (for Atrial fibrillation)".
    Leaves out the "(for ...)" part if the MedicationRequest has no reasonCode.
    """
    medication = medication_request.get("medicationCodeableConcept", {})
    medication_text = medication.get("text", "")

    dose_line = medication_text

    dosage_list = medication_request.get("dosageInstruction", [])

    if dosage_list:
        first_dosage = dosage_list[0]
        dosage_text = first_dosage.get("text", "")

        if dosage_text:
            dose_line = f"{medication_text} - {dosage_text}"

    if not dose_line:
        return dose_line

    reason_text = get_reason_text(medication_request=medication_request)

    if reason_text:
        dose_line = f"{dose_line} (for {reason_text})"

    return dose_line


def get_all_medication_doses(patient_id: str) -> list[str]:
    """
    Return one dose line for each of the patient's active meds.
    """
    medications = get_medications_for_patient(patient_id=patient_id)

    dose_lines = []

    for entry in medications:
        medication_request = entry["resource"]
        dose_text = get_dose_text(medication_request=medication_request)

        if dose_text:
            dose_lines.append(dose_text)

    return dose_lines


def get_crcl_text(patient_summary: dict) -> str:
    """
    Calculate creatinine clearance (Cockcroft-Gault) for the Patient Record.
    Says so plainly if weight or creatinine is missing.
    """
    age = patient_summary.get("age")
    is_female = patient_summary.get("is_female")
    weight_kg = patient_summary.get("weight_kg")
    serum_creatinine = patient_summary.get("serum_creatinine_mg_dl")

    if weight_kg is None:
        return "not available (no weight on file)"

    if serum_creatinine is None:
        return "not available (no serum creatinine on file)"

    dose_check = estimate_renal_dose_adjustment(
        age=age,
        weight_kg=weight_kg,
        serum_creatinine_mg_dl=serum_creatinine,
        is_female=is_female,
    )

    estimated_crcl = dose_check["estimated_crcl_ml_min"]

    if estimated_crcl is None:
        return "not available (serum creatinine must be greater than 0)"

    return f"{estimated_crcl} mL/min (Cockcroft-Gault)"


def find_patient_id_in_history(messages: list) -> str | None:
    """
    Check the user's messages from newest to oldest.
    Returns the most recent patient ID mentioned.
    """
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue

        patient_id = find_patient_id(user_input=str(message.content))

        if patient_id:
            return patient_id

    return None


def contains_any_word(
    text: str,
    words: list[str],
) -> bool:
    """
    Check if any of the words appears in the text.
    """
    for word in words:
        if word in text:
            return True

    return False


def is_details_only(user_input: str) -> bool:
    """
    Check if the question asks only for patient details,
    and nothing about risks or recommendations.
    """
    text = user_input.lower()

    asks_details = contains_any_word(
        text=text,
        words=DETAIL_WORDS,
    )

    asks_risk = contains_any_word(
        text=text,
        words=RISK_WORDS,
    )

    if asks_details and not asks_risk:
        return True

    return False


def load_knowledge_base_drugs() -> list[str]:
    """
    Read DRUGS.MD and return the drug names from its "# " heading lines, like ["Diphenhydramine", "Apixaban"].
    """
    markdown_text = CLEANED_MARKDOWN_FILE.read_text(encoding="utf-8")
    lines = markdown_text.splitlines()

    knowledge_base_drugs = []

    for line in lines:
        if line.startswith("# "):
            drug_name = line.removeprefix("# ").strip()
            knowledge_base_drugs.append(drug_name)

    return knowledge_base_drugs


def label_drug_names(
    drug_names: list[str],
    knowledge_base_drugs: list[str],
) -> list[str]:
    """
    Add a label after each drug name, like "Apixaban (in knowledge base)" or "Losartan (not in knowledge base)".
    Returns the labeled names in the same order.
    """
    labeled_names = []

    for drug_name in drug_names:
        if drug_name in knowledge_base_drugs:
            labeled_name = f"{drug_name} (in knowledge base)"
        else:
            labeled_name = f"{drug_name} (not in knowledge base)"

        labeled_names.append(labeled_name)

    return labeled_names


def get_geriatric_text(age: int | None) -> str:
    """
    Say whether geriatric criteria apply to this patient, based on age.
    Returns "apply (age 65 or older)", "do not apply (age under 65)" or "unknown (no age on file)".
    """
    if age is None:
        return "unknown (no age on file)"

    if age >= GERIATRIC_AGE:
        return f"apply (age {GERIATRIC_AGE} or older)"

    return f"do not apply (age under {GERIATRIC_AGE})"


def get_apixaban_criteria_met(
    age: int | None,
    weight_kg: float | None,
    serum_creatinine: float | None,
) -> list[str]:
    """
    Check the three apixaban dose-reduction criteria: age 80 or older, weight 60 kg or less, serum creatinine 1.5 mg/dL or more.
    A missing value does not count. Returns the criteria that are met, like ["age 80 or older", "weight 60 kg or less"].
    """
    criteria_met = []

    if age is not None and age >= APIXABAN_AGE_LIMIT:
        criteria_met.append(f"age {APIXABAN_AGE_LIMIT} or older")

    if weight_kg is not None and weight_kg <= APIXABAN_WEIGHT_LIMIT_KG:
        criteria_met.append(f"weight {APIXABAN_WEIGHT_LIMIT_KG} kg or less")

    if serum_creatinine is not None and serum_creatinine >= APIXABAN_CREATININE_LIMIT:
        criteria_met.append(f"serum creatinine {APIXABAN_CREATININE_LIMIT} mg/dL or more")

    return criteria_met


def count_apixaban_criteria(
    age: int | None,
    weight_kg: float | None,
    serum_creatinine: float | None,
) -> int:
    """
    Count how many apixaban dose-reduction criteria the patient meets.
    Returns a number from 0 to 3.
    """
    criteria_met = get_apixaban_criteria_met(
        age=age,
        weight_kg=weight_kg,
        serum_creatinine=serum_creatinine,
    )

    criteria_count = len(criteria_met)

    return criteria_count


def get_apixaban_correct_dose(criteria_count: int) -> str:
    """
    Choose the apixaban dose for atrial fibrillation from the number of criteria met.
    Returns "2.5 mg twice daily" if 2 or more are met, otherwise "5 mg twice daily".
    """
    if criteria_count >= APIXABAN_CRITERIA_FOR_REDUCED_DOSE:
        return APIXABAN_REDUCED_DOSE

    return APIXABAN_STANDARD_DOSE


def get_apixaban_text(
    age: int | None,
    weight_kg: float | None,
    serum_creatinine: float | None,
) -> str:
    """
    Build the apixaban line for the Patient Record.
    Returns text like "1 of 3 (age 80 or older). Correct dose for this patient: 5 mg twice daily."
    """
    criteria_met = get_apixaban_criteria_met(
        age=age,
        weight_kg=weight_kg,
        serum_creatinine=serum_creatinine,
    )

    criteria_count = count_apixaban_criteria(
        age=age,
        weight_kg=weight_kg,
        serum_creatinine=serum_creatinine,
    )

    correct_dose = get_apixaban_correct_dose(criteria_count=criteria_count)

    if criteria_met:
        criteria_text = ", ".join(criteria_met)
    else:
        criteria_text = "none"

    apixaban_text = (
        f"{criteria_count} of 3 ({criteria_text}). "
        f"Correct dose for this patient: {correct_dose}."
    )

    return apixaban_text


def build_patient_record(
    patient_id: str,
    drug_names: list[str],
) -> str:
    """
    Build the Patient Record text (name, age, sex, weight, conditions, current meds).
    """
    patient_summary = get_patient_summary(patient_id=patient_id)

    if patient_summary.get("is_female"):
        sex = "female"
    else:
        sex = "male"

    conditions = patient_summary.get("conditions", [])

    if conditions:
        conditions_text = ", ".join(conditions)
    else:
        conditions_text = "none on file"

    name = patient_summary.get("name")
    age = patient_summary.get("age")
    weight_kg = patient_summary.get("weight_kg")
    serum_creatinine = patient_summary.get("serum_creatinine_mg_dl")
    crcl_text = get_crcl_text(patient_summary=patient_summary)
    geriatric_text = get_geriatric_text(age=age)

    apixaban_line = ""

    if "Apixaban" in drug_names:
        apixaban_text = get_apixaban_text(
            age=age,
            weight_kg=weight_kg,
            serum_creatinine=serum_creatinine,
        )
        apixaban_line = f"\n  Apixaban dose-reduction criteria met (atrial fibrillation dosing): {apixaban_text}"

    knowledge_base_drugs = load_knowledge_base_drugs()
    labeled_names = label_drug_names(
        drug_names=drug_names,
        knowledge_base_drugs=knowledge_base_drugs,
    )
    medications_text = ", ".join(labeled_names)

    dose_lines = get_all_medication_doses(patient_id=patient_id)
    doses_text = ""

    for dose_line in dose_lines:
        doses_text = doses_text + f"\n    - {dose_line}"

    patient_record = f"""
Patient Record:
  Name: {name}
  Patient ID: {patient_id}
  Age: {age}
  Geriatric criteria: {geriatric_text}
  Sex: {sex}
  Weight: {weight_kg} kg
  Serum creatinine: {serum_creatinine} mg/dL
  Estimated CrCl: {crcl_text}{apixaban_line}
  Known conditions: {conditions_text}
  Currently taking: {medications_text}
  Doses:{doses_text}
""".strip()

    return patient_record


def build_metadata_filter(
    drug_name: str,
    interactions_only: bool,
) -> dict:
    """
    Build the Chroma filter for one drug.
    If interactions_only is True, only the interaction-related sections are searched.
    """
    drug_condition = {
        "header_1": {
            "$eq": drug_name
        }
    }

    if not interactions_only:
        return drug_condition

    section_condition = {
        "header_2": {
            "$in": INTERACTION_SECTIONS
        }
    }

    return {
        "$and": [
            drug_condition,
            section_condition,
        ]
    }


def build_must_have_filter(drug_name: str) -> dict:
    """
    Build the Chroma filter for one drug's Dosing Considerations and Interactions sections.
    """
    drug_condition = {
        "header_1": {
            "$eq": drug_name
        }
    }

    section_condition = {
        "header_2": {
            "$in": MUST_HAVE_SECTIONS
        }
    }

    return {
        "$and": [
            drug_condition,
            section_condition,
        ]
    }


def retrieve_documents(
    semantic_query: str,
    metadata_filter: dict | None,
    top_k: int = TOP_K,
) -> list:
    """
    Search Chroma for the top chunks, using the filter if there is one.
    """
    vector_store = get_vector_store()

    search_kwargs: dict = {
        "k": top_k,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vector_store.as_retriever(
        search_kwargs=search_kwargs
    )

    return retriever.invoke(semantic_query)


def retrieve_for_patient(
    drug_names: list[str],
    user_question: str,
) -> list:
    """
    Search Chroma twice for each of the patient's drugs:
    once with the user's question, and once for the must-have sections  so a general question
    still finds dose and interaction problems.
    With 2+ drugs, the question search only covers the interaction-related sections.
    """
    interactions_only = len(drug_names) >= 2

    all_documents = []
    seen_chunk_ids = []

    for drug_name in drug_names:
        metadata_filter = build_metadata_filter(
            drug_name=drug_name,
            interactions_only=interactions_only,
        )

        question_documents = retrieve_documents(
            semantic_query=user_question,
            metadata_filter=metadata_filter,
        )

        must_have_filter = build_must_have_filter(drug_name=drug_name)

        must_have_documents = retrieve_documents(
            semantic_query=MUST_HAVE_QUERY,
            metadata_filter=must_have_filter,
            top_k=MUST_HAVE_TOP_K,
        )

        add_new_documents(
            all_documents=all_documents,
            new_documents=question_documents,
            seen_chunk_ids=seen_chunk_ids,
        )

        add_new_documents(
            all_documents=all_documents,
            new_documents=must_have_documents,
            seen_chunk_ids=seen_chunk_ids,
        )

    return all_documents


def add_new_documents(
    all_documents: list,
    new_documents: list,
    seen_chunk_ids: list[str],
) -> None:
    """
    Add each new chunk to all_documents, skipping chunks that are already there.
    Both searches can return the same chunk, and the LLM only needs it once.
    """
    for document in new_documents:
        chunk_id = document.metadata.get("chunk_id")

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.append(chunk_id)
        all_documents.append(document)


def get_section_names(documents: list) -> list[str]:
    """
    List the drug and section of each chunk .
    """
    section_names = []

    for document in documents:
        header_1 = document.metadata.get("header_1")
        header_2 = document.metadata.get("header_2")
        section_names.append(f"{header_1} / {header_2}")

    return section_names


def format_context(documents: list) -> str:

    context_parts = []

    for doc in documents:
        header_1 = doc.metadata.get("header_1")
        header_2 = doc.metadata.get("header_2")
        content = doc.page_content

        part = f"""
{header_1}
{header_2}
{content}
""".strip()

        context_parts.append(part)

    return "\n\n".join(context_parts)


def are_all_drugs_out_of_scope(drug_names: list[str]) -> bool:
    """
    Check if every drug the patient takes is missing from the knowledge base (DRUGS.MD).
    Returns True only when none of the drugs is in the knowledge base.
    """
    knowledge_base_drugs = load_knowledge_base_drugs()

    for drug_name in drug_names:
        if drug_name in knowledge_base_drugs:
            return False

    return True


def build_out_of_scope_reply(
    patient_record: str,
    drug_names: list[str],
) -> str:
    """
    Build the reply for a patient whose drugs are all outside the knowledge base, without the LLM.
    Returns the Patient Record, one "not in the knowledge base" line per drug, and the overall severity line.
    """
    reply = patient_record + "\n"

    for drug_name in drug_names:
        reply = reply + f"\n{drug_name}: not in the knowledge base."

    reply = reply + "\n\nOverall severity: cannot be assessed - not in the knowledge base."

    return reply


def make_reply(text: str) -> dict:

    return {
        "messages": [
            AIMessage(content=text)
        ]
    }


def chatbot_node(state: MessagesState) -> dict:
    """
    Answer one question: find the patient, get their data and guideline text, then ask the LLM.
    """
    all_messages = state["messages"]
    user_input = str(all_messages[-1].content)

    patient_id = find_patient_id_in_history(messages=all_messages)

    drug_names = []
    patient_record = ""

    if patient_id:
        drug_names = get_all_medication_names(patient_id=patient_id)

        if not drug_names:
            return make_reply(text=f"No medications found for patient {patient_id}. If that's unexpected, check that the FHIR server (Docker) is running.")

        patient_record = build_patient_record(
            patient_id=patient_id,
            drug_names=drug_names,
        )

        if is_details_only(user_input=user_input):
            return make_reply(text=patient_record)

        if are_all_drugs_out_of_scope(drug_names=drug_names):
            out_of_scope_reply = build_out_of_scope_reply(
                patient_record=patient_record,
                drug_names=drug_names,
            )
            return make_reply(text=out_of_scope_reply)

    if drug_names:
        documents = retrieve_for_patient(
            drug_names=drug_names,
            user_question=user_input,
        )
    else:
        documents = retrieve_documents(
            semantic_query=user_input,
            metadata_filter=None,
        )

    logger.info(f"Patient {patient_id}: retrieved {len(documents)} chunks")

    section_names = get_section_names(documents=documents)
    logger.info(f"Retrieved sections: {section_names}")

    if not documents:
        return make_reply(text="No relevant guideline information found.")

    context = format_context(documents=documents)

    question_with_context = f"""
{patient_record}

Guideline Context:
{context}

Question: {user_input}
""".strip()

    earlier_messages = all_messages[:-1]

    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)]
    messages_for_llm.extend(earlier_messages)
    messages_for_llm.append(HumanMessage(content=question_with_context))

    model = get_chat_llm_model()
    model_with_tool = model.bind_tools([recommend_with_dose])

    response = model_with_tool.invoke(messages_for_llm)

    if not response.tool_calls:
        return {"messages": [response]}

    tool_call = response.tool_calls[0]
    tool_result = recommend_with_dose.invoke(tool_call["args"])

    return make_reply(text=tool_result)


def ask_for_drug_name(prompt: str) -> str:
    """
    Ask for a drug name and check it with RxNorm.
    Fixes small typos, and asks again if the drug is not found.
    """
    while True:
        typed_name = input(prompt).strip()

        ingredient_rxcui, ingredient_name = normalize_drug_name(drug_name=typed_name)

        if ingredient_name:
            return ingredient_name.capitalize()

        print(f"  Couldn't find '{typed_name}' in RxNorm, check the spelling.")


def confirm_and_apply_recommendation(
    patient_id: str,
    old_drug_name: str,
    new_drug_name: str,
    dose_mg: float,
    reason: str,
) -> None:
    """
    Show the proposed change. Only after the user types y,
    stop the old drug and add the new one in FHIR.
    """
    print(
        f"\nChatbot proposes: stop {old_drug_name} and start {new_drug_name} "
        f"at {dose_mg} mg for patient {patient_id}. "
        f"Reason: {reason}"
    )

    answer = input("Apply this to FHIR? (y/n): ").strip().lower()

    if answer != "y":
        print("Not applied.\n")
        return

    stop_result = discontinue_medication(
        patient_id=patient_id,
        drug_name=old_drug_name,
    )

    if not stop_result.get("success"):
        failure_reason = stop_result.get("reason", stop_result)
        print(f"Could not stop {old_drug_name}: {failure_reason}\n")
        return

    start_result = post_medication_request(
        patient_id=patient_id,
        drug_name=new_drug_name,
        dose_mg=dose_mg,
        note=reason,
    )

    if not start_result.get("success"):
        print(f"{old_drug_name} was stopped, but {new_drug_name} could not be added: {start_result}\n")
        return

    print(f"Change applied to FHIR: {old_drug_name} discontinued, {new_drug_name} {dose_mg} mg started.\n")


def run_apply_command() -> None:
    """
    Collect the details for a medication change and pass them to the confirm step.
    """
    patient_id = input("  Patient ID: ").strip()
    old_drug_name = ask_for_drug_name(prompt="  Drug to stop: ")
    new_drug_name = ask_for_drug_name(prompt="  New drug name: ")
    dose_mg = ask_for_number(prompt="  Dose (mg): ")
    reason = input("  Reason: ").strip()

    confirm_and_apply_recommendation(
        patient_id=patient_id,
        old_drug_name=old_drug_name,
        new_drug_name=new_drug_name,
        dose_mg=dose_mg,
        reason=reason,
    )


def setup_logging() -> None:
    """
    Save log messages to adr.log so they don't clutter the chat screen.
    """
    logging.basicConfig(
        filename="adr.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_graph() -> CompiledStateGraph:
    """
    Set up the LangGraph flow.
    Every conversation is saved to chat_memory.db.
    """
    connection = sqlite3.connect(
        "chat_memory.db",
        check_same_thread=False,
    )

    memory = SqliteSaver(connection)

    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    return workflow.compile(checkpointer=memory)


def make_new_thread_id() -> str:
    """
    Make a new thread ID from the current date and time, like "chat-2026-10-24-1-33-5".
    So every run starts a fresh conversation.
    """
    now = datetime.now()

    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second

    return f"chat-{year}-{month}-{day}-{hour}-{minute}-{second}"


def chat(
    thread_id: str = typer.Option(
        "",
        help="Conversation thread ID to resume. Leave empty to start a new conversation.",
    ),
) -> None:
    """
    Read what the user types and send it to exit, /apply, or the chatbot.
    """
    if not thread_id:
        thread_id = make_new_thread_id()

    setup_logging()

    graph = build_graph()

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    print(f"Chatbot: Hi, how can I help you today? (thread: {thread_id})\n")

    while True:
        user_input = input("You: ")
        command = user_input.strip().lower()

        if not command:
            continue

        if command in EXIT_COMMANDS:
            print("Thank you. Goodbye!")
            break

        if command in APPLY_COMMANDS:
            run_apply_command()
            continue

        output = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )

        output_messages = output["messages"]
        latest_reply = output_messages[-1]

        print(f"Chatbot: {latest_reply.content}\n")


if __name__ == "__main__":
    typer.run(chat)