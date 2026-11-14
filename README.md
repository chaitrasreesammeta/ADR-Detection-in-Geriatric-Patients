# ADR Detection in Geriatric Patients

A command-line chatbot that screens older adults for adverse drug reaction (ADR)
risk. It reads a patient's medications from a FHIR R4 server, retrieves matching
guideline text from a drug knowledge base (RAG), and reports drug interactions,
dose problems and ADRs with a severity rating. The chatbot is evaluated with RAGAS
on 40 synthetic test patients.

## How it works

1. The user asks about a patient, for example `Check patient 1244 for ADR risks`.
2. The chatbot reads the patient's medications, age, weight and serum creatinine
   from HAPI FHIR and calculates creatinine clearance (Cockcroft-Gault).
3. It retrieves the relevant sections of the knowledge base (`adr/data/DRUGS.MD`,
   16 drugs) from a Chroma vector store.
4. gpt-4o writes the answer from the patient record and the retrieved guideline
   text only. Drugs that are not in the knowledge base are reported as such.

## Project structure

```
adr/                 chatbot, FHIR and dosage clients, RAGAS scripts
adr/data/            knowledge base (DRUGS.MD) and synthetic test patients
evaluation/          RAGAS dataset, experiment results and results.md
src/                 earlier LangChain learning examples
```

## Setup

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, an OpenAI API key.

1. Install the dependencies: `uv sync`
2. Copy `.env.example` to `.env` and add your OpenAI API key.
3. Start a HAPI FHIR R4 server on `http://localhost:8080/fhir` (Docker).
4. From inside `adr/`:
   - `uv run python create_test_patients.py` loads the 40 synthetic test patients
   - `uv run python load_vector_store.py` builds the Chroma vector store

## Run the chatbot

```
cd adr
uv run python chatbot_cli.py
```

## Run the evaluation

```
cd adr
uv run python create_ragas_dataset.py
uv run python run_ragas_experiment.py
```

The experiment scores Faithfulness, Context Precision, Context Recall and
Response Relevancy and saves a CSV in `evaluation/experiments/`. A full run calls
the OpenAI API for all 40 patients. Results are summarised in
[`evaluation/results.md`](evaluation/results.md).

All patients in this repository are synthetic. No real patient data is used.
