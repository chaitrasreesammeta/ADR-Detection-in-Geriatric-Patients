# RAGAS evaluation results

Chatbot model: gpt-4o (temperature 0). Evaluator: gpt-4o with text-embedding-3-small.
Dataset: `datasets/adr_test_dataset.csv`, 40 synthetic test patients (TP01-TP40).

| Metric | Before fixes (0-58) | After fixes (11-59) |
|---|---|---|
| Faithfulness | 0.618 | 0.622 |
| Context Precision | 0.844 | 0.846 |
| Context Recall | 0.967 | 0.954 |
| Response Relevancy | 0.475 | 0.455 |

- Before: `experiments/ragas-gpt-4o-2026-11-14-0-58.csv`
- After: `experiments/ragas-gpt-4o-2026-11-14-11-59.csv`
